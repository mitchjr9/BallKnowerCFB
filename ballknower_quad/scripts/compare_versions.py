"""
ballknower_quad.scripts.compare_versions
========================================

**The ablation harness.** Runs two model versions through the *same* walk-forward
folds on the *same* games, then applies an explicit, pre-declared kill criterion —
so V1.1 ships on evidence, not vibes.

    python -m ballknower_quad.scripts.compare_versions
    python -m ballknower_quad.scripts.compare_versions --a v1 --b v1.1 --first-eval 2019

Why a *paired* test: both versions predict the identical set of games, so the
per-game Brier scores are paired observations. Testing the mean paired difference
is far more sensitive than comparing two independent Brier averages — it cancels
out "this was a weird season" noise that hits both models equally.

The verdict rule (declare before you look, per BallKnower house style):

    SHIP   if  ΔBrier improvement > 2 standard errors  (roughly 95% confidence)
    WATCH  if  improvement is positive but inside 2 SE  (real but unproven)
    KILL   if  improvement ≤ 0

This is the same discipline that killed NFL V4's weather features: gain
importance said 11.95%, permutation said 0.00%, and the honest test won. A tie
is a kill — extra features are free overfitting room for future retrains.
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

from ballknower_quad.data.bundle import load_bundle, season_range
from ballknower_quad.models.cfb_model_v1 import train_v1
from ballknower_quad.models.feature_builder import FeatureBuilder
from ballknower_quad.settings import settings


def _per_game_scores(bundle, version: str, first_eval: int) -> pd.DataFrame:
    """Walk-forward; return per-game (key, prob, y, brier) for one version."""
    games = bundle.games
    seasons = sorted(games["season"].dropna().unique().astype(int))
    out = []

    for s in [x for x in seasons if x >= first_eval]:
        train_games = games[games["season"] < s]
        test_games = games[games["season"] == s]
        if train_games["season"].nunique() < 2 or test_games.empty:
            continue

        model, _ = train_v1(train_games, version=version, calibrate=True,
                            game_ppa=bundle.game_ppa, season_ppa=bundle.season_ppa,
                         preseason=bundle.preseason)

        # warm a fresh builder on history only, then walk the test season
        fb = FeatureBuilder(version=version, game_ppa=bundle.game_ppa,
                            season_ppa=bundle.season_ppa,
                            preseason=bundle.preseason)
        fb.build(train_games, fit_elo=True)
        Xte, ywin, _ = fb.build(test_games, fit_elo=True)

        mask = ywin.notna().to_numpy()
        if not mask.any():
            continue
        Xte = Xte[mask].reset_index(drop=True)
        yt = ywin[mask].astype(int).to_numpy()
        meta = test_games.reset_index(drop=True)[mask]
        probs = model.predict_proba(Xte)["p_blended"].to_numpy()
        out.append(pd.DataFrame({
            "key": meta["game_id"].astype(str).to_numpy(),
            "season": s, "prob": probs, "y": yt,
            "brier": (probs - yt) ** 2,
            "neutral": meta["neutral_site"].astype(bool).to_numpy(),
            "postseason": (meta["season_type"].astype(str).str.lower()
                           != "regular").to_numpy(),
            "week": pd.to_numeric(meta["week"], errors="coerce").to_numpy(),
        }))

    if not out:
        raise SystemExit("No evaluable folds — lower --first-eval or load more seasons.")
    return pd.concat(out, ignore_index=True)


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def _brier_decomposition(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict:
    """
    Murphy (1973) decomposition:  Brier = Reliability - Resolution + Uncertainty

    * Reliability  — calibration error. LOWER is better. This is what ECE tracks.
    * Resolution   — how far the model's predictions move away from the base
                     rate, i.e. sharpness / discriminating power. HIGHER is better.
    * Uncertainty  — base-rate variance. Fixed by the data; not the model's doing.

    This is the decomposition that explains an ECE improvement showing up
    alongside a Brier *regression*: a model can get better at being honest while
    getting worse at being decisive, and Brier charges it for the second.
    """
    ybar = float(y.mean())
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rel = res = 0.0
    n = len(y)
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        nk = m.sum()
        pk, yk = p[m].mean(), y[m].mean()
        rel += nk * (pk - yk) ** 2
        res += nk * (yk - ybar) ** 2
    return {"reliability": rel / n, "resolution": res / n,
            "uncertainty": ybar * (1 - ybar)}


def _summary(df: pd.DataFrame) -> dict:
    p, y = df["prob"].to_numpy(), df["y"].to_numpy()
    eps = 1e-9
    out = {
        "n": len(y),
        "accuracy": float(((p >= 0.5).astype(int) == y).mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
        "ece": _ece(p, y),
    }
    out.update(_brier_decomposition(p, y))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Paired ablation between two versions.")
    ap.add_argument("--a", default="v1", help="baseline version")
    ap.add_argument("--b", default="v1.1", help="challenger version")
    ap.add_argument("--first-eval", type=int, default=2019)
    ap.add_argument("--seasons-back", type=int, default=11)
    ap.add_argument("--b-anchor-source", default=None,
                   choices=["cfbd", "self"],
                   help="override the PPA prior-season anchor for the CHALLENGER "
                        "only: 'cfbd' uses /ppa/teams, 'self' carries our own "
                        "end-of-season ratings forward.")
    ap.add_argument("--b-preseason-blend", type=float, default=None,
                   help="override the preseason Elo blend for the CHALLENGER "
                        "only (0.0 = features present but Elo unseeded, 1.0 = "
                        "ignore last season). Ablates V2's core mechanism.")
    ap.add_argument("--b-portal-metric", default=None,
                   choices=["in", "net", "top_in"],
                   help="override which portal quantity feeds the preseason "
                        "composite, for the CHALLENGER only.")
    ap.add_argument("--b-fcs-mode", default=None,
                   choices=["naive", "skip", "pooled"],
                   help="override how FBS-vs-FCS games feed PPA, for the "
                        "CHALLENGER only.")
    ap.add_argument("--b-ppa-strength", type=float, default=None,
                   help="override PPA opponent-adjustment strength for the "
                        "CHALLENGER only (0.0 = raw PPA, 1.0 = full). Use this "
                        "to ablate the adjustment itself without editing settings.")
    ap.add_argument("--subset", default="all",
                   choices=["all", "neutral", "non_neutral", "postseason",
                            "regular", "early", "late", "close"],
                   help="evaluate on a SUBPOPULATION. Essential for features "
                        "that only touch some games: is_postseason can only "
                        "move ~7%% of the slate, so a whole-slate test dilutes "
                        "its effect ~14x and will read as noise either way.")
    ap.add_argument("--exclude-seasons", type=int, nargs="*", default=None,
                   help="drop seasons from the evaluation, e.g. --exclude-seasons "
                        "2020 for the COVID year")
    args = ap.parse_args(argv)

    seasons = season_range(args.seasons_back)
    print(f"Loading {seasons[0]}–{seasons[-1]} …")
    # load with the richer version so both share one games frame
    bundle = load_bundle(seasons, version=args.b)
    print(f"  source: {bundle.source} | {len(bundle.games):,} games"
          + (f" | {len(bundle.game_ppa):,} team-game PPA rows"
             if bundle.game_ppa is not None else ""))

    print(f"\nWalk-forward: {args.a} …")
    da = _per_game_scores(bundle, args.a, args.first_eval)

    overrides = {}
    if args.b_ppa_strength is not None:
        overrides["ppa_adjust_strength"] = args.b_ppa_strength
    if args.b_anchor_source is not None:
        overrides["ppa_anchor_source"] = args.b_anchor_source
    if args.b_fcs_mode is not None:
        overrides["ppa_fcs_mode"] = args.b_fcs_mode
    if args.b_preseason_blend is not None:
        overrides["preseason_blend"] = args.b_preseason_blend
    if args.b_portal_metric is not None:
        overrides["preseason_portal_metric"] = args.b_portal_metric

    if overrides:
        # Applied only while scoring the challenger, then restored — this is what
        # makes "does the opponent adjustment actually help?" a one-command test.
        prior = {k: getattr(settings, k) for k in overrides}
        for k, v in overrides.items():
            setattr(settings, k, v)
        desc = ", ".join(f"{k}={v}" for k, v in overrides.items())
        print(f"Walk-forward: {args.b} ({desc}) …")
        try:
            db = _per_game_scores(bundle, args.b, args.first_eval)
        finally:
            for k, v in prior.items():
                setattr(settings, k, v)
    else:
        print(f"Walk-forward: {args.b} …")
        db = _per_game_scores(bundle, args.b, args.first_eval)

    merged = da.merge(db, on=["key", "season"], suffixes=("_a", "_b"))

    if args.subset != "all":
        sel = {
            "neutral":    merged["neutral_a"],
            "postseason": merged["postseason_a"],
            "regular":    ~merged["postseason_a"].astype(bool),
            "early":      merged["week_a"] <= 4,   # the cold-start window V2 targets
            "late":       merged["week_a"] >= 10,
            "non_neutral": ~merged["neutral_a"].astype(bool),
            "close":      (merged["prob_a"] - 0.5).abs() <= 0.15,
        }[args.subset]
        before = len(merged)
        merged = merged[sel.astype(bool)]
        keep = set(merged["key"])
        da = da[da["key"].isin(keep)]
        db = db[db["key"].isin(keep)]
        print(f"  subset '{args.subset}': {len(merged):,} of {before:,} games")
    if args.exclude_seasons:
        drop = set(args.exclude_seasons)
        before = len(merged)
        merged = merged[~merged["season"].isin(drop)]
        da = da[~da["season"].isin(drop)]
        db = db[~db["season"].isin(drop)]
        print(f"  excluded season(s) {sorted(drop)}: "
              f"{before - len(merged):,} games dropped")
    if merged.empty:
        raise SystemExit("Could not pair games between versions (check game_id).")

    sa, sb = _summary(da), _summary(db)

    # paired difference: positive = challenger is BETTER (lower Brier)
    d = merged["brier_a"].to_numpy() - merged["brier_b"].to_numpy()
    delta = float(d.mean())
    se = float(d.std(ddof=1) / np.sqrt(len(d)))
    t = delta / se if se > 0 else 0.0

    label_a, label_b = args.a, args.b
    if label_a == label_b:
        # same version string on both sides (e.g. a PPA-strength ablation) —
        # disambiguate so the table and verdict aren't nonsense.
        label_a += " [base]"
        bits = []
        if args.b_ppa_strength is not None:
            bits.append(f"strength={args.b_ppa_strength}")
        if args.b_anchor_source is not None:
            bits.append(f"anchor={args.b_anchor_source}")
        if args.b_fcs_mode is not None:
            bits.append(f"fcs={args.b_fcs_mode}")
        if args.b_preseason_blend is not None:
            bits.append(f"blend={args.b_preseason_blend}")
        if args.b_portal_metric is not None:
            bits.append(f"portal={args.b_portal_metric}")
        label_b += f" [{', '.join(bits)}]" if bits else " [variant]"

    print("\n" + "=" * 68)
    print(f"  ABLATION  {label_a}  vs  {label_b}"
          f"   (paired n={len(merged):,}, {merged['season'].min()}–"
          f"{merged['season'].max()})")
    print("=" * 68)
    print(f"  {'metric':<14}{label_a:>16}{label_b:>16}{'Δ':>12}")
    for k in ("accuracy", "brier", "log_loss", "ece"):
        print(f"  {k:<14}{sa[k]:>16.4f}{sb[k]:>16.4f}{sb[k] - sa[k]:>+12.4f}")
    print(f"  {'-- Brier decomposition ' + '-' * 44}")
    for k, arrow in (("reliability", "lower better"),
                     ("resolution", "HIGHER better"),
                     ("uncertainty", "fixed")):
        print(f"  {k:<14}{sa[k]:>16.4f}{sb[k]:>16.4f}{sb[k] - sa[k]:>+12.4f}"
              f"   ({arrow})")

    print(f"\n  Paired ΔBrier (improvement) : {delta:+.5f}")
    print(f"  Standard error              : {se:.5f}")
    print(f"  t-statistic                 : {t:+.2f}")

    print("\n  " + "-" * 64)
    # Two different questions share this harness, and they read in OPPOSITE
    # directions:
    #   ADD test      (a != b)   challenger = a new version you might ship.
    #   KNOCKOUT test (a == b + overrides)  challenger = the SAME version with a
    #                            mechanism switched OFF. Here a negative delta
    #                            means "removing it hurt", i.e. KEEP it — the
    #                            add-test wording ("KILL the challenger") would
    #                            say the opposite of what you mean.
    knockout = (args.a == args.b) and bool(overrides)

    if knockout:
        what = ", ".join(f"{k}={v}" for k, v in overrides.items())
        if delta < 0 and t < -2.0:
            verdict = (f"✅ KEEP the mechanism — disabling it ({what}) costs "
                       f"{abs(delta):.5f} Brier at t={t:.2f}. It's earning its place.")
        elif delta < 0:
            verdict = (f"↩️  LEAN KEEP — disabling it ({what}) looks worse "
                       f"({abs(delta):.5f} Brier, t={t:.2f}) but inside noise. "
                       f"Nothing here argues for removing it.")
        elif t > 2.0:
            verdict = (f"❌ REMOVE the mechanism — the model is BETTER without it "
                       f"({what}), by more than 2 SE.")
        else:
            verdict = (f"⚠️  INCONCLUSIVE — disabling it ({what}) changes little "
                       f"(t={t:.2f}). The mechanism isn't clearly doing work.")
    elif delta <= 0:
        verdict = (f"❌ KILL — {label_b} is not better than {label_a}. "
                   f"Don't ship extra features that don't earn their seat.")
    elif t > 2.0:
        verdict = (f"✅ SHIP — {label_b} beats {label_a} by more than 2 SE. "
                   f"Promote it (set active_model_version).")
    else:
        verdict = (f"⚠️  WATCH — {label_b} is better but within noise "
                   f"(needs t>2.0). Keep {label_a} active; re-test with more data.")
    print(f"  VERDICT: {verdict}")
    print("  " + "-" * 64)

    print("\n  Per-season paired ΔBrier (positive = challenger better):")
    per = (merged.assign(d=merged["brier_a"] - merged["brier_b"])
                 .groupby("season")["d"].agg(["mean", "size"])
                 .rename(columns={"mean": "delta", "size": "n"}))
    per["delta"] = per["delta"].round(5)
    print(per.to_string())

    if not settings.has_cfbd_key:
        print("\n  ⚠️  SYNTHETIC data — this is a PLUMBING test, not evidence.")
        print("     Set CFBD_API_KEY and re-run for the verdict that counts.")

    print("\n" + settings.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
