"""
ballknower_quad.scripts.repro_check
===================================

Runs the *identical* walk-forward twice and reports whether the results are
bit-identical.

Why this exists: the effects we're adjudicating are ~0.0005 Brier. If the
pipeline itself wobbles by that much between runs, then every SHIP/WATCH/KILL
verdict is partly reading noise, and comparing a number from today against one
from last week is meaningless. This is the check that tells you whether a
baseline that "moved" actually moved.

    python -m ballknower_quad.scripts.repro_check --version v1.1
    python -m ballknower_quad.scripts.repro_check --version v1.2 --first-eval 2022

If it reports drift, the usual culprit is thread-nondeterministic floating-point
summation in XGBoost. Pin ``n_jobs`` to 1 in ``settings.xgb_clf_params`` and
re-run: slower, but every ablation becomes exactly reproducible.
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np

from ballknower_quad.data.bundle import load_bundle, season_range
from ballknower_quad.scripts.compare_versions import _per_game_scores, _summary
from ballknower_quad.settings import settings


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Check pipeline determinism.")
    ap.add_argument("--version", default="v1.1")
    ap.add_argument("--first-eval", type=int, default=2022)
    ap.add_argument("--seasons-back", type=int, default=11)
    args = ap.parse_args(argv)

    seasons = season_range(args.seasons_back)
    bundle = load_bundle(seasons, version=args.version)
    print(f"Loading {seasons[0]}–{seasons[-1]} ({bundle.source}) …")
    print(f"  n_jobs = {settings.xgb_clf_params.get('n_jobs')}, "
          f"random_state = {settings.random_state}\n")

    print(f"Run 1: {args.version} …")
    d1 = _per_game_scores(bundle, args.version, args.first_eval)
    print(f"Run 2: {args.version} (identical config) …")
    d2 = _per_game_scores(bundle, args.version, args.first_eval)

    m = d1.merge(d2, on=["key", "season"], suffixes=("_1", "_2"))
    s1, s2 = _summary(d1), _summary(d2)

    print("\n" + "=" * 62)
    print(f"  REPRODUCIBILITY — {args.version}  (n={len(m):,})")
    print("=" * 62)
    print(f"  {'metric':<14}{'run 1':>14}{'run 2':>14}{'Δ':>14}")
    for k in ("accuracy", "brier", "log_loss", "ece"):
        print(f"  {k:<14}{s1[k]:>14.6f}{s2[k]:>14.6f}{s2[k] - s1[k]:>+14.6f}")

    dp = np.abs(m["prob_1"].to_numpy() - m["prob_2"].to_numpy())
    n_diff = int((dp > 1e-12).sum())
    print(f"\n  games with any probability difference : {n_diff:,} of {len(m):,}")
    print(f"  max |Δprob|                           : {dp.max():.3e}")
    brier_delta = abs(s2["brier"] - s1["brier"])
    print(f"  |ΔBrier| between identical runs       : {brier_delta:.6f}")

    print("\n  " + "-" * 58)
    if n_diff == 0:
        print("  ✅ DETERMINISTIC — identical configs give identical results.")
        print("     Baseline numbers are comparable across runs, so any movement")
        print("     you see between sessions is a real config change.")
        ok = True
    else:
        print("  ⚠️  NON-DETERMINISTIC — identical configs diverge.")
        print(f"     Run-to-run wobble ({brier_delta:.5f} Brier) must be treated as")
        print("     a noise floor: no ablation smaller than this means anything.")
        print("     Fix: set n_jobs=1 in settings.xgb_clf_params / xgb_reg_params.")
        ok = False
    print("  " + "-" * 58)
    print("\n" + settings.disclaimer)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
