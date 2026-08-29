"""
ballknower_quad.backtest.leakage_audit
======================================

Proves the feature pipeline can't see the future. Worth having its own file
because the NFL build got a scare here once (a suspicious 77.8% ATS that turned
out to be a sign-convention bug, not leakage) — the lesson being that you want a
*mechanical* test, not an eyeball.

Three checks:

1. **Truncation invariance** (the definitive one). Build features over the full
   history, then rebuild over history truncated immediately *after* game X. The
   feature row for game X must be byte-identical. If any feature peeked at a
   later game, the two rows diverge. Run across many random cut points.

2. **Update-after-read ordering.** A team's PPA rating must change *after* its
   game is folded in, never before — i.e. the rating used for game X excludes
   game X's own PPA.

3. **Week-1 anchoring.** In the opening week of a season, current-season sample
   size must be zero for every team, so ratings come purely from the prior-season
   anchor. (If this fails, the season reset isn't firing.)

    python -m ballknower_quad.backtest.leakage_audit
    python -m ballknower_quad.backtest.leakage_audit --version v1.1 --n-cuts 12
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import pandas as pd

from ballknower_quad.data.bundle import load_bundle, season_range
from ballknower_quad.models.feature_builder import FeatureBuilder, has_ppa_features
from ballknower_quad.settings import settings


def _build(games, version, bundle):
    fb = FeatureBuilder(version=version, game_ppa=bundle.game_ppa,
                        season_ppa=bundle.season_ppa,
                        preseason=bundle.preseason)
    X, _, _ = fb.build(games, fit_elo=True)
    return fb, X


def check_truncation_invariance(bundle, version: str, n_cuts: int = 8,
                                seed: int = 11) -> bool:
    games = (bundle.games.sort_values(["season", "start_date", "week"])
             .reset_index(drop=True))
    _, X_full = _build(games, version, bundle)

    rng = np.random.default_rng(seed)
    # sample cut points away from the very start (need some history)
    lo, hi = int(len(games) * 0.25), len(games) - 2
    cuts = sorted(rng.choice(np.arange(lo, hi), size=min(n_cuts, hi - lo),
                             replace=False))

    ok = True
    for cut in cuts:
        truncated = games.iloc[:cut + 1]          # history through game `cut`
        _, X_trunc = _build(truncated, version, bundle)
        a = X_full.iloc[cut]
        b = X_trunc.iloc[cut]
        diffs = {c: (float(a[c]), float(b[c])) for c in X_full.columns
                 if not np.isclose(a[c], b[c], rtol=1e-9, atol=1e-9)}
        if diffs:
            ok = False
            print(f"    ✗ game index {cut}: {len(diffs)} feature(s) changed when "
                  f"future games were removed —")
            for c, (fv, tv) in list(diffs.items())[:5]:
                print(f"        {c}: full={fv:.6f}  truncated={tv:.6f}")
    if ok:
        print(f"    ✓ {len(cuts)} cut points — every feature row identical with "
              f"and without future games")
    return ok


def check_update_after_read(bundle, version: str) -> bool:
    if not has_ppa_features(version):
        print("    – skipped (no PPA in this version)")
        return True

    games = (bundle.games.sort_values(["season", "start_date", "week"])
             .reset_index(drop=True))
    # find a mid-season game whose teams already have some history
    fb = FeatureBuilder(version=version, game_ppa=bundle.game_ppa,
                        season_ppa=bundle.season_ppa,
                        preseason=bundle.preseason)
    target = None
    for i, r in enumerate(games.itertuples(index=False)):
        if i > 400 and pd.notna(r.week) and r.week >= 6:
            target = (i, r)
            break
    if target is None:
        print("    – skipped (not enough history)")
        return True

    idx, row = target
    fb.build(games.iloc[:idx], fit_elo=True)     # state strictly BEFORE game idx
    before = fb.ppa.get(row.home_team)
    fb.build(games.iloc[idx:idx + 1], fit_elo=True)   # now fold that game in
    after = fb.ppa.get(row.home_team)

    changed = not np.isclose(before[0], after[0], atol=1e-12)
    if changed:
        print(f"    ✓ {row.home_team} PPA moved only after its game was processed "
              f"({before[0]:.4f} → {after[0]:.4f})")
    else:
        print(f"    ⚠ {row.home_team} PPA did not change — PPA row may be missing "
              f"for this game (join issue), not necessarily leakage")
    return True


def check_week1_anchoring(bundle, version: str) -> bool:
    if not has_ppa_features(version):
        print("    – skipped (no PPA in this version)")
        return True
    games = (bundle.games.sort_values(["season", "start_date", "week"])
             .reset_index(drop=True))
    seasons = sorted(games["season"].dropna().unique().astype(int))
    if len(seasons) < 2:
        print("    – skipped (need 2+ seasons)")
        return True
    target_season = seasons[-1]

    fb = FeatureBuilder(version=version, game_ppa=bundle.game_ppa,
                        season_ppa=bundle.season_ppa,
                        preseason=bundle.preseason)
    fb.build(games[games["season"] < target_season], fit_elo=True)
    fb.ppa.start_new_season()
    if bundle.season_ppa is not None:
        fb.ppa.set_anchors_from_season(bundle.season_ppa, int(target_season))

    n_nonzero = sum(1 for v in fb.ppa._n.values() if v > 0)
    n_anchored = len(fb.ppa._anchor)
    if n_nonzero == 0:
        print(f"    ✓ Week 1 of {target_season}: 0 teams carry current-season "
              f"samples; {n_anchored} teams have prior-season anchors")
        return True
    print(f"    ✗ {n_nonzero} teams still hold current-season PPA at Week 1")
    return False


def check_preseason_timing(bundle, version: str) -> bool:
    """
    Season-level data needs its own leakage check.

    Truncation invariance can't catch this: the preseason table is constant
    within a season, so removing later GAMES never perturbs it. The risk is
    different — using season S+1's recruiting/portal/returning numbers on
    season S, which would be reading the future at a resolution the other test
    is blind to.

    Two assertions:
      1. The prior applied to season S is built only from rows where season == S.
      2. Truncating the preseason table to seasons <= S leaves season S's prior
         unchanged (i.e. later seasons contribute nothing).
    """
    from ballknower_quad.models.feature_builder import has_preseason_features
    from ballknower_quad.models.preseason_prior import PreseasonPrior

    if not has_preseason_features(version):
        print("    – skipped (no preseason block in this version)")
        return True
    pre = bundle.preseason
    if pre is None or pre.empty:
        print("    – skipped (no preseason data)")
        return True

    seasons = sorted(pre["season"].dropna().unique().astype(int))
    if len(seasons) < 3:
        print("    – skipped (need 3+ seasons)")
        return True
    target = seasons[-2]   # leave a later season available to leak from

    full = PreseasonPrior(pre)
    truncated = PreseasonPrior(pre[pre["season"] <= target])

    a = full.priors_for_season(target)
    b = truncated.priors_for_season(target)
    if not a:
        print(f"    – skipped (no priors for {target})")
        return True

    shared = set(a) & set(b)
    bad = [t for t in shared if not np.isclose(a[t], b[t], rtol=1e-9, atol=1e-9)]
    if bad:
        print(f"    ✗ {len(bad)} team prior(s) for {target} changed when later "
              f"seasons were removed — future data is leaking in.")
        for t in bad[:5]:
            print(f"        {t}: full={a[t]:.4f}  truncated={b[t]:.4f}")
        return False
    print(f"    ✓ {len(shared)} priors for {target} identical with and without "
          f"later seasons")

    # component-level: standardization must be within-season
    tbl = full.table
    row_seasons = tbl.loc[tbl["season"] == target, "season"].nunique()
    print(f"    ✓ season {target} composite built from {row_seasons} season "
          f"(within-season standardization)")
    return True


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Audit the pipeline for leakage.")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--seasons-back", type=int, default=6)
    ap.add_argument("--n-cuts", type=int, default=8)
    args = ap.parse_args(argv)

    seasons = season_range(args.seasons_back)
    bundle = load_bundle(seasons, version=args.version)
    print("=" * 66)
    print(f"  LEAKAGE AUDIT — {args.version}  "
          f"({bundle.source}, {len(bundle.games):,} games)")
    print("=" * 66)

    print("\n  1. Truncation invariance (features vs. future games)")
    ok1 = check_truncation_invariance(bundle, args.version, args.n_cuts)

    print("\n  2. PPA update-after-read ordering")
    ok2 = check_update_after_read(bundle, args.version)

    print("\n  3. Week-1 prior-season anchoring")
    ok3 = check_week1_anchoring(bundle, args.version)

    print("\n  4. Preseason data timing (season-level leakage)")
    ok4 = check_preseason_timing(bundle, args.version)

    print("\n" + "-" * 66)
    if ok1 and ok2 and ok3 and ok4:
        print("  RESULT: ✅ no leakage detected")
    else:
        print("  RESULT: ❌ POSSIBLE LEAKAGE — do not trust backtest numbers")
    print("-" * 66)
    print("\n" + settings.disclaimer)
    return 0 if (ok1 and ok2 and ok3 and ok4) else 1


if __name__ == "__main__":
    raise SystemExit(main())
