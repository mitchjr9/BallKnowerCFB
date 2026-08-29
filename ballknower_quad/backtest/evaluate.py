"""
ballknower_quad.backtest.evaluate
=================================

Honest, leakage-free **walk-forward** evaluation — the part that keeps everyone
honest. For each held-out season S (from ``first_eval_season`` onward), we fit on
everything strictly before S and predict S, so no future game ever informs a past
prediction. Reports the metrics that matter for a *content* model:

    accuracy      directional hit rate (inflated in CFB by talent mismatches —
                  read it alongside the others, never alone)
    brier         calibration + sharpness of the probability
    log_loss      penalizes confident misses
    ece           expected calibration error (10 bins)
    ats           against-the-spread hit rate vs. the margin model (if provided)

A reliability table (predicted vs. actual by probability bucket) is printed so you
can see, the way we did for Gridiron's V3.1, whether the tails are honest.

Usage:
    python -m ballknower_quad.backtest.evaluate
    python -m ballknower_quad.backtest.evaluate --first-eval 2019 --with-margin
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

from ballknower_quad.data.bundle import load_bundle, season_range
from ballknower_quad.models.cfb_margin_v1 import train_margin_v1
from ballknower_quad.models.cfb_model_v1 import train_v1
from ballknower_quad.models.feature_builder import FeatureBuilder
from ballknower_quad.settings import settings


def _ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(probs[m].mean() - y[m].mean())
    return float(ece)


def _tier_table(probs: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """
    Newsletter tier report: for each confidence label, how often it fires and how
    often it's right.

    The reliability table answers "is 70% really 70%". This answers the two
    questions that actually govern the newsletter: does a "Lock" hit at a rate
    that earns the word, and does it fire rarely enough to mean anything? A tier
    that covers a third of the slate is a label, not a signal — and thresholds
    tuned on one model version don't transfer to another, because promoting a
    model reshapes the probability distribution underneath them.
    """
    from ballknower_quad.content_utils import confidence_tier

    fav_prob = np.maximum(probs, 1 - probs)
    picked_home = probs >= 0.5
    correct = (picked_home == y.astype(bool)).astype(int)
    tiers = [confidence_tier(abs(p - 0.5) * 2.0) for p in probs]

    df = pd.DataFrame({"tier": tiers, "fav_prob": fav_prob, "correct": correct})
    rows = []
    order = [label for _, label in settings.confidence_tiers]
    for label in order:
        m = df["tier"] == label
        if not m.any():
            continue
        sub = df[m]
        rows.append({
            "tier": label,
            "picks": int(len(sub)),
            "share": round(len(sub) / len(df), 3),
            "avg_conf": round(float(sub["fav_prob"].mean()), 3),
            "hit_rate": round(float(sub["correct"].mean()), 3),
            "gap": round(float(sub["correct"].mean() - sub["fav_prob"].mean()), 3),
        })
    return pd.DataFrame(rows)


def _reliability_table(probs: np.ndarray, y: np.ndarray, n_bins: int = 10
                       ) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bucket": f"{bins[b]:.1f}-{bins[b + 1]:.1f}",
            "n": int(m.sum()),
            "pred": round(float(probs[m].mean()), 3),
            "actual": round(float(y[m].mean()), 3),
        })
    return pd.DataFrame(rows)


def walk_forward(games: pd.DataFrame, first_eval_season: int,
                 with_margin: bool = False, version: str = "v1",
                 game_ppa=None, season_ppa=None, preseason=None) -> dict:
    seasons = sorted(games["season"].dropna().unique().astype(int))
    eval_seasons = [s for s in seasons if s >= first_eval_season]
    if not eval_seasons:
        raise SystemExit("No seasons available at/after first_eval_season.")

    all_probs, all_y = [], []
    ats_hits = ats_total = 0
    per_season = []

    for s in eval_seasons:
        train_games = games[games["season"] < s]
        test_games = games[games["season"] == s]
        if train_games["season"].nunique() < 2:
            continue

        model, _ = train_v1(train_games, calibrate=True, version=version,
                            game_ppa=game_ppa, season_ppa=season_ppa,
                            preseason=preseason)

        # build test features on a fresh builder warmed on history only
        fb = FeatureBuilder(version=version, game_ppa=game_ppa,
                            season_ppa=season_ppa, preseason=preseason)
        fb.build(train_games, fit_elo=True)            # warm to end of history
        Xte, ywin, ymar = fb.build(test_games, fit_elo=True)  # walk the test yr
        mask = ywin.notna()
        Xte, yt = Xte[mask].reset_index(drop=True), ywin[mask].astype(int).to_numpy()
        if not len(Xte):
            continue
        probs = model.predict_proba(Xte)["p_blended"].to_numpy()

        acc = float(((probs >= 0.5).astype(int) == yt).mean())
        brier = float(np.mean((probs - yt) ** 2))
        per_season.append({"season": s, "n": len(yt), "acc": round(acc, 3),
                           "brier": round(brier, 4)})
        all_probs.append(probs)
        all_y.append(yt)

        if with_margin:
            mm = train_margin_v1(train_games, version=version,
                                 game_ppa=game_ppa, season_ppa=season_ppa,
                                 preseason=preseason)
            pm = mm.predict_margin(Xte)
            actual_margin = ymar[mask].to_numpy()
            # ATS vs. a synthetic "market" = Elo-implied margin (no lines offline)
            implied = Xte["elo_diff_with_hca"].to_numpy() / settings.elo_points_per_400
            pick_home = (pm - implied) > 0
            cover_home = (actual_margin - implied) > 0
            ats_hits += int(np.sum(pick_home == cover_home))
            ats_total += int(len(pm))

    probs = np.concatenate(all_probs)
    y = np.concatenate(all_y)
    eps = 1e-9
    out = {
        "n": int(len(y)),
        "seasons": f"{eval_seasons[0]}–{eval_seasons[-1]}",
        "accuracy": float(((probs >= 0.5).astype(int) == y).mean()),
        "brier": float(np.mean((probs - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(probs + eps)
                                   + (1 - y) * np.log(1 - probs + eps))),
        "ece": _ece(probs, y),
        "home_win_rate": float(y.mean()),
        "per_season": per_season,
        "reliability": _reliability_table(probs, y),
        "tiers": _tier_table(probs, y),
    }
    if with_margin and ats_total:
        out["ats"] = ats_hits / ats_total
        out["ats_n"] = ats_total
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Walk-forward backtest for BallKnower Quad.")
    p.add_argument("--first-eval", type=int, default=settings.first_train_season + 3)
    p.add_argument("--seasons-back", type=int, default=10,
                   help="how many seasons of history to load")
    p.add_argument("--with-margin", action="store_true")
    p.add_argument("--version", default=None,
                   help="feature-set version to evaluate (v1 or v1.1)")
    args = p.parse_args(argv)

    version = (args.version or settings.active_model_version).lower()
    seasons = season_range(args.seasons_back)
    print(f"Loading seasons {seasons[0]}–{seasons[-1]} "
          f"({'CFBD' if settings.has_cfbd_key else 'SYNTHETIC'}) for {version}…")
    bundle = load_bundle(seasons, version=version)
    games = bundle.games
    print(f"  {len(games):,} games, {games['home_team'].nunique()} teams\n")

    res = walk_forward(games, args.first_eval, with_margin=args.with_margin,
                       version=version, game_ppa=bundle.game_ppa,
                       season_ppa=bundle.season_ppa,
                       preseason=bundle.preseason)
    print("=" * 64)
    print(f"  Walk-forward results  ({res['seasons']}, n={res['n']:,})")
    print("=" * 64)
    print(f"  accuracy       {res['accuracy']:.4f}")
    print(f"  brier          {res['brier']:.4f}")
    print(f"  log_loss       {res['log_loss']:.4f}")
    print(f"  ece            {res['ece']:.4f}")
    print(f"  home_win_rate  {res['home_win_rate']:.4f}")
    if "ats" in res:
        print(f"  ATS (vs Elo)   {res['ats']:.4f}  (n={res['ats_n']:,})")
    print("\n  Reliability (predicted vs actual):")
    print(res["reliability"].to_string(index=False))
    print("\n  Newsletter tiers  (gap = actual minus advertised; + is good):")
    print(res["tiers"].to_string(index=False))
    locks = res["tiers"][res["tiers"]["tier"].str.contains("Lock")]
    if len(locks):
        share = float(locks["share"].iloc[0])
        if share > 0.20:
            print(f"    ⚠ Locks fire on {share*100:.0f}% of games — too often to "
                  f"mean anything. Raise the threshold in settings.")
        if float(locks["gap"].iloc[0]) < -0.02:
            print("    ⚠ Locks are hitting BELOW their advertised rate.")
    print("\n  Per season:")
    print(pd.DataFrame(res["per_season"]).to_string(index=False))
    print("\n" + settings.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
