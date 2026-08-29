"""
ballknower_quad.scripts.diagnose
================================

Answers "*why* did the ablation land where it did?" instead of guessing. Four
checks, each aimed at a specific failure mode we've hit before in this family of
models:

1. **Non-FBS contamination.** FCS opponents must be pooled at the Elo floor, not
   treated as ordinary programs. If North Dakota State is carrying its own 1500-
   base rating, every FBS-vs-FCS game is mispriced and the training data is
   polluted. Symptom in the wild: a Week 1 slate showing ~50% for an FBS team
   hosting an FCS team.

2. **PPA vs Elo divergence, by conference tier.** Opponent adjustment is the
   whole point of the PPA block. If Group of Five teams rank systematically
   *higher* on net PPA than on Elo, the adjustment is too weak and PPA is partly
   measuring schedule softness rather than quality.

3. **Feature redundancy.** The PPA block only earns its seat if it carries
   information Elo and rolling margin don't already have. This is the exact test
   that condemned the NFL QB-rating feature. |r| > 0.9 against an existing
   feature is a red flag.

4. **Neutral-site composition.** ``is_neutral`` is a single flag covering both
   bowl games (opt-outs, month-long layoffs, flat motivation) and Week 1 kickoff
   classics (full rosters, peak intensity). If the training data is mostly bowls,
   the model learns bowl behavior and applies it to season openers.

    python -m ballknower_quad.scripts.diagnose --version v1.1
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

from ballknower_quad.data.bundle import load_bundle, season_range
from ballknower_quad.models.feature_builder import (FeatureBuilder,
                                                    PPA_FEATURE_COLUMNS,
                                                    has_ppa_features)
from ballknower_quad.settings import settings

P5 = {"SEC", "Big Ten", "Big 12", "ACC", "Pac-12", "Pac-10"}


def check_non_fbs(games: pd.DataFrame) -> None:
    print("\n  1. Non-FBS contamination")
    have_class = ("home_classification" in games.columns
                  and games["home_classification"].notna().any())
    if not have_class:
        print("    ⚠ no classification column — re-pull games (delete "
              "data_cache/cfbd) so the loader captures it.")
        return

    long = pd.concat([
        games[["home_team", "home_classification"]].rename(
            columns={"home_team": "team", "home_classification": "cls"}),
        games[["away_team", "away_classification"]].rename(
            columns={"away_team": "team", "away_classification": "cls"}),
    ])
    by_team = long.groupby("team")["cls"].agg(
        lambda s: s.dropna().mode().iat[0] if s.notna().any() else None)
    non_fbs = sorted([t for t, c in by_team.items()
                      if c is not None and str(c).lower() != "fbs"])
    n_games = int(((games["home_classification"].str.lower() != "fbs")
                   | (games["away_classification"].str.lower() != "fbs")).sum())

    print(f"    teams in frame       : {len(by_team)}")
    print(f"    non-FBS teams        : {len(non_fbs)}")
    print(f"    games w/ a non-FBS side: {n_games:,} "
          f"({n_games / max(len(games), 1) * 100:.1f}%)")
    if non_fbs:
        print(f"    sample               : {', '.join(non_fbs[:6])}"
              + (" …" if len(non_fbs) > 6 else ""))
    return set(non_fbs)


def verify_pooling(fb, non_fbs: set) -> None:
    """The check that actually matters: did the FCS floor DO its job?

    Detecting non-FBS teams in the raw feed is expected and harmless. The failure
    mode is one of them carrying its own persisted Elo, which means it farmed and
    surrendered rating points like a real program.
    """
    leaked = sorted(non_fbs & set(fb.elo.ratings))
    if leaked:
        print(f"    ✗ {len(leaked)} non-FBS team(s) carry their own Elo rating: "
              f"{', '.join(leaked[:6])}")
        print("      → the FCS floor is NOT firing; classification normalization "
              "failed.")
    else:
        print(f"    ✓ pooled correctly — 0 of {len(non_fbs)} non-FBS teams hold "
              f"an individual Elo rating")
        print(f"      ({len(fb.elo.ratings)} rated teams = FBS only)")


def check_ppa_vs_elo(fb: FeatureBuilder, games: pd.DataFrame) -> None:
    print("\n  3. PPA vs Elo divergence by conference tier")
    if fb.ppa is None:
        print("    – skipped (v1 has no PPA)")
        return

    conf = {}
    for r in games.itertuples(index=False):
        conf[r.home_team] = r.home_conference
        conf[r.away_team] = r.away_conference

    rows = []
    for team, elo in fb.elo.ratings.items():
        c = conf.get(team)
        if c is None or str(c).upper() == "FCS":
            continue
        off, dfn = fb.ppa.get(team)
        rows.append({"team": team, "conf": c, "elo": elo, "net_ppa": off - dfn})
    df = pd.DataFrame(rows)
    if len(df) < 20:
        print("    – skipped (too few teams)")
        return

    df["elo_rank"] = df["elo"].rank(ascending=False)
    df["ppa_rank"] = df["net_ppa"].rank(ascending=False)
    # positive gap = PPA rates the team BETTER than Elo does
    df["gap"] = df["elo_rank"] - df["ppa_rank"]
    df["tier"] = np.where(df["conf"].isin(P5), "P5", "G5")

    rho = df["elo_rank"].corr(df["ppa_rank"], method="spearman")
    print(f"    Spearman(Elo rank, PPA rank) : {rho:.3f}")
    if rho > 0.95:
        print("      → nearly identical ordering; PPA is largely redundant with Elo.")
    elif rho < 0.55:
        print("      → very low agreement; check the PPA join before trusting it.")

    tiers = df.groupby("tier")["gap"].agg(["mean", "size"]).round(2)
    print(f"\n    Mean rank gap (PPA better than Elo = positive):")
    print("      " + tiers.to_string().replace("\n", "\n      "))
    g5 = tiers.loc["G5", "mean"] if "G5" in tiers.index else 0.0
    if g5 > 8:
        print(f"\n      ⚠ G5 teams rank ~{g5:.0f} spots higher on PPA than Elo — "
              f"opponent adjustment looks TOO WEAK.")
        print("        Try: settings.ppa_opponent_adjust=False to isolate its "
              "effect, or strengthen the adjustment.")

    print("\n    Biggest PPA-over-Elo risers (adjustment stress cases):")
    print("      " + df.nlargest(6, "gap")[["team", "conf", "elo", "net_ppa", "gap"]]
          .to_string(index=False).replace("\n", "\n      "))


def check_ppa_scale(fb, bundle) -> None:
    fb_offset = fb.ppa.fcs_offset if fb.ppa is not None else float("nan")
    print("\n  4. PPA scale & centering")
    if fb.ppa is None:
        print("    – skipped (v1 has no PPA)")
        return
    rep = fb.ppa.centering_report()
    if not rep:
        print("    – no PPA state")
        return

    if bundle.game_ppa is not None and len(bundle.game_ppa):
        gp = bundle.game_ppa
        # Split FBS-vs-FBS from FBS-vs-FCS. Only the former can be expected to
        # balance: CFBD returns one side of a cupcake game, so those rows are
        # inherently one-directional and will never sum symmetrically.
        counts = gp.groupby(["season", "game_id"])["team"].transform("size")
        two_sided = gp[counts >= 2]
        one_sided = gp[counts < 2]
        print(f"    raw, both sides    : off {two_sided['off_ppa'].mean():.3f}  "
              f"def {two_sided['def_ppa'].mean():.3f}   (these MUST match)")
        if len(one_sided):
            print(f"    raw, one-sided rows: off {one_sided['off_ppa'].mean():.3f}  "
                  f"def {one_sided['def_ppa'].mean():.3f}   "
                  f"({len(one_sided):,} rows — CFBD returns only the FBS side "
                  f"of FBS-vs-FCS games)")
            emp = (one_sided["off_ppa"].mean() - two_sided["off_ppa"].mean())
            print(f"    → empirical FCS offset ≈ {abs(emp):.3f}  "
                  f"| in use: {fb_offset:.3f}"
                  f"{'  (auto)' if settings.ppa_fcs_offset is None else '  (pinned)'}")
    if bundle.season_ppa is not None and len(bundle.season_ppa):
        sp = bundle.season_ppa
        print(f"    season anchors     : off {sp['off_ppa_season'].mean():.3f}  "
              f"def {sp['def_ppa_season'].mean():.3f}")
    print(f"    mode               : fcs={settings.ppa_fcs_mode}, "
          f"anchor={settings.ppa_anchor_source}, "
          f"strength={settings.ppa_adjust_strength}")
    print(f"    adjusted ratings   : off {rep['adj_mean_off']:.3f}  "
          f"def {rep['adj_mean_def']:.3f}")
    gap = rep["centering_gap"]
    print(f"    centering gap      : {gap:+.3f}")
    if abs(gap) > 0.05:
        print("      ⚠ offense and defense are centered on DIFFERENT means. In a")
        print("        closed system that's impossible — every offensive play is")
        print("        some defense's play — so the adjustment has pulled the two")
        print("        scales apart and net_ppa carries schedule-dependent bias.")
    else:
        print("      ✓ offense and defense share a scale")

    n_cur = rep["teams_with_current_games"]
    print(f"\n    teams w/ current-season games : {n_cur} of {rep['n_teams']}")
    if n_cur < rep["n_teams"] * 0.2:
        print("      ⚠ the ratings you're reading are ~ENTIRELY the prior-season")
        print("        anchor (CFBD's own /ppa/teams numbers), not this engine's")
        print("        in-season adjustment. Judging the leaderboard now judges")
        print("        CFBD. Use --through-week to inspect mid-season instead.")


def check_redundancy(X: pd.DataFrame, version: str) -> None:
    print("\n  5. Feature redundancy (PPA vs existing signal)")
    if not has_ppa_features(version):
        print("    – skipped (v1 has no PPA)")
        return
    base = [c for c in X.columns if c not in PPA_FEATURE_COLUMNS]
    corr = X.corr()
    flagged = False
    for p in PPA_FEATURE_COLUMNS:
        sub = corr.loc[p, base].abs().sort_values(ascending=False)
        top, val = sub.index[0], sub.iloc[0]
        mark = "  ⚠ REDUNDANT" if val > 0.90 else ("  · high" if val > 0.75 else "")
        flagged = flagged or val > 0.90
        print(f"    {p:<16} max |r| vs base = {val:.3f}  (vs {top}){mark}")
    if flagged:
        print("      → a feature above 0.90 is mostly re-expressing Elo/margin. "
              "That's the NFL QB-rating failure mode.")


def check_neutral(games: pd.DataFrame) -> None:
    print("\n  6. Neutral-site composition")
    n = games[games["neutral_site"].astype(bool)]
    if n.empty:
        print("    – no neutral-site games")
        return
    post = n[n["season_type"].astype(str).str.lower() != "regular"]
    early = n[(n["season_type"].astype(str).str.lower() == "regular")
              & (pd.to_numeric(n["week"], errors="coerce") <= 2)]
    print(f"    neutral games        : {len(n):,} "
          f"({len(n) / len(games) * 100:.1f}% of all games)")
    print(f"      postseason (bowls) : {len(post):,} "
          f"({len(post) / len(n) * 100:.0f}%)")
    print(f"      Week 1-2 openers   : {len(early):,} "
          f"({len(early) / len(n) * 100:.0f}%)")
    if len(post) / len(n) > 0.6:
        print("      ⚠ `is_neutral` is dominated by bowls, so the model learns "
              "bowl behavior and applies it to Week 1 kickoff classics.")
        print("        Fix: split into `is_postseason` + `is_neutral` (v1.2).")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose model/data health.")
    ap.add_argument("--version", default="v1.1")
    ap.add_argument("--seasons-back", type=int, default=11)
    ap.add_argument("--through-week", type=int, default=None,
                   help="inspect ratings as of this week of the LAST season "
                        "(e.g. 10) instead of at the season boundary, where "
                        "ratings are pure prior-season anchor")
    args = ap.parse_args(argv)

    seasons = season_range(args.seasons_back)
    bundle = load_bundle(seasons, version=args.version)
    games = bundle.games

    print("=" * 70)
    print(f"  DIAGNOSTICS — {args.version}  ({bundle.source}, "
          f"{len(games):,} games)")
    print("=" * 70)

    non_fbs = check_non_fbs(games)

    fb = FeatureBuilder(version=args.version, game_ppa=bundle.game_ppa,
                        season_ppa=bundle.season_ppa,
                        preseason=bundle.preseason)
    completed = games[games["completed"] & games["home_points"].notna()]

    if args.through_week is not None:
        last = int(completed["season"].max())
        prior = completed[completed["season"] < last]
        cur = completed[(completed["season"] == last)
                        & (pd.to_numeric(completed["week"], errors="coerce")
                           <= args.through_week)]
        completed = pd.concat([prior, cur], ignore_index=True)
        print(f"\n  (inspecting as of {last} week {args.through_week}: "
              f"{len(cur)} games into that season)")

    X, _, _ = fb.build(completed, fit_elo=True)

    print("\n  2. Non-FBS Elo pooling")
    verify_pooling(fb, non_fbs)

    print("\n  " + "-" * 60)
    check_ppa_vs_elo(fb, completed)
    check_ppa_scale(fb, bundle)
    check_redundancy(X, args.version)
    check_neutral(completed)

    if fb.ppa is not None:
        print("\n  Top 15 net PPA (games = current-season sample, "
              "anchor_w = share from last season):")
        print("    " + fb.ppa.top(15).to_string(index=False)
              .replace("\n", "\n    "))

    print("\n" + settings.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
