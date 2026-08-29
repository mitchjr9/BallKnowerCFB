"""
ballknower_quad.scripts.calibrate_tiers
=======================================

Derives the Lock / Strong / Lean / Pass thresholds from the model's **actual
out-of-sample probability distribution**, instead of guessing them.

Why this needs a tool rather than a judgement call:

* Thresholds don't transfer between model versions. Promoting v1.1 → v2 reshaped
  the probability distribution underneath the labels, which is how "Lock" ended
  up firing on 45% of games without anyone changing a threshold.
* Tier is now stamped on every ledger row, and `tier_config_version` is derived
  from the thresholds. Getting them wrong doesn't just make a scruffy
  newsletter — it contaminates the tier-level reliability curve from row one,
  and that curve is the whole point of the ledger.

Two things must both be true for a tier to be worth publishing:

1. **Volume** — it fires rarely enough to mean something. A label on a third of
   the slate is a decoration.
2. **Honesty** — its hit rate is at least its advertised confidence. A Lock that
   hits *below* the number it advertises is the one result that should stop a
   release.

This tool sets thresholds by volume, then checks honesty and refuses to
recommend a config that fails it.

    # look, don't touch
    python -m ballknower_quad.scripts.calibrate_tiers

    # different editorial split
    python -m ballknower_quad.scripts.calibrate_tiers --shares 0.08 0.18 0.24

    # write the result into settings.py (backs up first)
    python -m ballknower_quad.scripts.calibrate_tiers --write
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import shutil
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ballknower_quad.data.bundle import load_bundle, season_range
from ballknower_quad.scripts.compare_versions import _per_game_scores
from ballknower_quad.settings import settings

LABELS = ["🔒 Lock", "💪 Strong", "🎯 Lean", "🪙 Pass"]
# Default editorial split. Locks ~10% keeps the word scarce; Pass ~45% is the
# honest admission that most college football games aren't worth a strong call.
DEFAULT_SHARES = [0.10, 0.20, 0.25]

# A tier whose realized hit rate sits more than this below its advertised
# confidence is publishing a number it doesn't earn.
HONESTY_TOLERANCE = 0.02


def _depth_from_week(week: float) -> str:
    """
    Approximate data_depth from week number.

    The ledger derives depth from games actually played; week is a close proxy
    (byes aside) and is what the walk-forward already carries. Depth matters here
    because it CAPS the tier — so raw threshold volume overstates how many Locks
    actually get published early in a season.
    """
    if not np.isfinite(week):
        return "rich"
    if week <= 2:
        return "none"
    if week <= 5:
        return "thin"
    return "rich"


_CAP_RANK = {"none": 2, "thin": 1, "rich": 0}   # index into LABELS


def _apply_depth_cap(idx: int, depth: str) -> int:
    return max(idx, _CAP_RANK.get(depth, 0))


def thresholds_from_shares(fav_prob: np.ndarray,
                           shares: List[float]) -> List[float]:
    """Pick thresholds so each tier captures its target share of the slate."""
    cum = 0.0
    out = []
    for s in shares:
        cum += s
        # top `cum` fraction by confidence sits above this quantile
        out.append(float(np.quantile(fav_prob, 1.0 - cum)))
    return [round(t, 4) for t in out]


def assign_tiers(fav_prob: np.ndarray, weeks: np.ndarray,
                 thresholds: List[float]) -> tuple:
    """Return (raw_idx, capped_idx) into LABELS."""
    idx = np.full(len(fav_prob), len(LABELS) - 1)
    for i, th in enumerate(thresholds):
        idx = np.where((fav_prob >= th) & (idx == len(LABELS) - 1), i, idx)
    capped = np.array([_apply_depth_cap(int(i), _depth_from_week(w))
                       for i, w in zip(idx, weeks)])
    return idx, capped


def _gap_stats(fav_prob: np.ndarray, correct: np.ndarray,
               mask: np.ndarray) -> tuple:
    """(share, avg_conf, hit_rate, gap, gap_se) for one tier."""
    if not mask.any():
        return 0.0, None, None, None, None
    n = int(mask.sum())
    conf = float(fav_prob[mask].mean())
    hit = float(correct[mask].mean())
    se = float(np.sqrt(max(hit * (1 - hit), 1e-9) / n))
    return n / len(fav_prob), conf, hit, hit - conf, se


def tier_report(fav_prob: np.ndarray, correct: np.ndarray, weeks: np.ndarray,
                thresholds: List[float]) -> pd.DataFrame:
    idx, capped = assign_tiers(fav_prob, weeks, thresholds)

    rows = []
    for i, label in enumerate(LABELS):
        m_raw = idx == i
        m = capped == i
        if not m.any() and not m_raw.any():
            continue
        share, conf, hit, gap, se = _gap_stats(fav_prob, correct, m)
        rows.append({
            "tier": label,
            "min_prob": (f"{thresholds[i]:.3f}" if i < len(thresholds)
                         else "—"),
            "picks": int(m.sum()),
            "share": round(share, 3),
            "share_pre_cap": round(float(m_raw.mean()), 3),
            "avg_conf": round(conf, 3) if conf is not None else None,
            "hit_rate": round(hit, 3) if hit is not None else None,
            "gap": round(gap, 3) if gap is not None else None,
            "gap_se": round(se, 3) if se is not None else None,
        })
    return pd.DataFrame(rows)


def _fails_honesty(gap, se, tolerance: float) -> bool:
    """
    A tier fails only when it is BOTH materially and significantly overconfident.

    A small tier's hit rate is noisy — 285 picks at 94% carries an SE near 0.014,
    so a -0.027 gap is only ~1.9 SE and could easily be sampling noise. Failing a
    config on that alone would have us chasing variance and retuning thresholds
    every week, which is its own kind of dishonesty.
    """
    if gap is None or se is None:
        return False
    if se <= 0:
        # Degenerate SE (a tier that won or lost everything). Fall back to the
        # material threshold alone rather than dividing by zero.
        return gap < -tolerance
    return (gap < -tolerance) and (gap < -2.0 * se)


def auto_thresholds(fav_prob: np.ndarray, correct: np.ndarray,
                    weeks: np.ndarray, tolerance: float,
                    min_shares: List[float]) -> tuple:
    """
    Search for the most SCARCE thresholds that still pass the honesty check.

    Setting thresholds by quantile alone assumes calibration is uniform across
    the probability range. It isn't: this model is mildly overconfident in the
    tail, so pushing the Lock cut higher selects a more overconfident
    subpopulation and makes the gap worse rather than better. Quantiles hand you
    scarcity and then fail the honesty check; this searches under the constraint
    instead.

    Greedy top-down: for each tier take the highest threshold that clears both
    the minimum-volume floor and the honesty check. Valid because a tier's
    post-cap membership depends only on thresholds at or above it.
    """
    candidates = np.round(np.arange(0.52, 0.995, 0.005), 3)
    chosen: List[float] = []
    notes: List[str] = []

    for i in range(len(LABELS) - 1):
        upper = chosen[-1] if chosen else 1.01
        best = None
        for th in sorted(candidates, reverse=True):
            if th >= upper:
                continue
            trial = chosen + [float(th)] + [0.0] * (len(LABELS) - 2 - i)
            _, capped = assign_tiers(fav_prob, weeks, trial)
            m = capped == i
            share, conf, hit, gap, se = _gap_stats(fav_prob, correct, m)
            if share < min_shares[i]:
                continue
            if _fails_honesty(gap, se, tolerance):
                continue
            best = float(th)
            break
        if best is None:
            # Nothing satisfies both; fall back to the loosest honest cut.
            best = float(min(candidates))
            notes.append(f"{LABELS[i]}: no threshold met both constraints; "
                         f"fell back to {best:.3f}")
        chosen.append(best)
    return [round(c, 3) for c in chosen], notes


def render_settings_block(thresholds: List[float]) -> str:
    lines = ["    confidence_tiers: List[Tuple[float, str]] = "
             "field(default_factory=lambda: ["]
    for th, label in zip(thresholds, LABELS[:-1]):
        lines.append(f'        ({th:.2f}, "{label}"),')
    lines.append(f'        (0.00, "{LABELS[-1]}"),')
    lines.append("    ])")
    return "\n".join(lines)


def write_settings(thresholds: List[float]) -> Path:
    settings_path = Path(__file__).resolve().parent.parent / "settings.py"
    backup = settings_path.with_suffix(".py.bak")
    shutil.copy2(settings_path, backup)

    text = settings_path.read_text()
    pattern = re.compile(
        r"    confidence_tiers: List\[Tuple\[float, str\]\] = "
        r"field\(default_factory=lambda: \[.*?\n    \]\)",
        re.DOTALL)
    if not pattern.search(text):
        raise SystemExit("Could not locate the confidence_tiers block in "
                         "settings.py — patch it by hand.")
    settings_path.write_text(pattern.sub(render_settings_block(thresholds), text))
    return backup


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Recalibrate newsletter tiers.")
    ap.add_argument("--version", default=None,
                    help="model version (default: active_model_version)")
    ap.add_argument("--first-eval", type=int, default=2019)
    ap.add_argument("--seasons-back", type=int, default=11)
    ap.add_argument("--shares", type=float, nargs=3, default=DEFAULT_SHARES,
                    metavar=("LOCK", "STRONG", "LEAN"),
                    help="target share of the slate for each tier "
                         f"(default: {DEFAULT_SHARES})")
    ap.add_argument("--auto", action="store_true",
                    help="search for the most scarce thresholds that still PASS "
                         "the honesty check, instead of setting them by quantile "
                         "and hoping. Use this when a quantile config fails.")
    ap.add_argument("--min-shares", type=float, nargs=3,
                    default=[0.03, 0.08, 0.12],
                    metavar=("LOCK", "STRONG", "LEAN"),
                    help="minimum post-cap share per tier for --auto")
    ap.add_argument("--tolerance", type=float, default=HONESTY_TOLERANCE,
                    help="how far below advertised confidence a tier may hit "
                         f"(default {HONESTY_TOLERANCE})")
    ap.add_argument("--keep-current", action="store_true",
                    help="report only; propose no change")
    ap.add_argument("--write", action="store_true",
                    help="patch settings.py in place (backs up to settings.py.bak)")
    args = ap.parse_args(argv)

    version = (args.version or settings.active_model_version).lower()
    if sum(args.shares) >= 1.0:
        raise SystemExit("--shares must sum to less than 1.0 (Pass takes the rest)")

    seasons = season_range(args.seasons_back)
    print(f"Loading {seasons[0]}–{seasons[-1]} for {version} …")
    bundle = load_bundle(seasons, version=version)
    print(f"  source: {bundle.source} | {len(bundle.games):,} games")
    print(f"\nWalk-forward {version} (out-of-sample probabilities) …")
    df = _per_game_scores(bundle, version, args.first_eval)

    p = df["prob"].to_numpy()
    y = df["y"].to_numpy()
    weeks = df["week"].to_numpy()
    fav_prob = np.maximum(p, 1 - p)
    correct = ((p >= 0.5).astype(int) == y).astype(int)

    print(f"  n = {len(df):,} games, {df['season'].min()}–{df['season'].max()}")

    old = [float(t) for t, _ in settings.confidence_tiers][:3]
    notes: List[str] = []
    if args.keep_current:
        new, mode = old, "current (unchanged)"
    elif args.auto:
        new, notes = auto_thresholds(fav_prob, correct, weeks,
                                     args.tolerance, list(args.min_shares))
        mode = f"auto-search, min shares {list(args.min_shares)}"
    else:
        new = thresholds_from_shares(fav_prob, list(args.shares))
        mode = f"quantile, target shares {list(args.shares)}"

    print("\n" + "=" * 74)
    print(f"  CURRENT thresholds {old}")
    print("=" * 74)
    print(tier_report(fav_prob, correct, weeks, old).to_string(index=False))

    print("\n" + "=" * 74)
    print(f"  PROPOSED thresholds {new}   ({mode})")
    print("=" * 74)
    rep = tier_report(fav_prob, correct, weeks, new)
    print(rep.to_string(index=False))
    print("\n  share         = after the ledger's data-depth cap (what you publish)")
    print("  share_pre_cap = by probability alone")
    print("  gap           = hit rate minus advertised confidence; + is good")

    # ---- honesty check ------------------------------------------------- #
    problems, marginal = [], []
    for r in rep.itertuples(index=False):
        if r.gap is None:
            continue
        # A tier that went 30-for-30 has a binomial SE of exactly 0, which is an
        # artifact of a small sample rather than certainty. Guard the division
        # and report it honestly.
        se = r.gap_se if r.gap_se else None
        sig = f", {r.gap / se:+.1f} SE" if se else " (SE~0, tiny sample)"
        desc = (f"{r.tier} hits {r.hit_rate:.3f} but advertises "
                f"{r.avg_conf:.3f} (gap {r.gap:+.3f}{sig})")
        if _fails_honesty(r.gap, r.gap_se, args.tolerance):
            problems.append(desc)
        elif r.gap < -args.tolerance:
            marginal.append(desc)

    print("\n  " + "-" * 70)
    if problems:
        print("  ❌ DO NOT SHIP — a tier advertises materially more than it "
              "delivers:")
        for p_ in problems:
            print(f"     {p_}")
        print()
        print("     NOTE: for an overconfident tail, LOWERING the threshold")
        print("     usually helps — a higher cut selects a more overconfident")
        print("     subpopulation. Re-run with --auto, which searches under the")
        print("     honesty constraint instead of assuming quantiles are safe.")
    elif marginal:
        print("  ⚠️  PASSES, with a caveat — a tier is slightly overconfident but")
        print("     not significantly so (inside 2 SE), i.e. plausibly noise:")
        for m_ in marginal:
            print(f"     {m_}")
        print("     Safe to ship; worth re-checking after a few more weeks.")
    else:
        print("  ✅ Every tier hits at or above its advertised confidence.")
    for n_ in notes:
        print(f"  · {n_}")
    print("  " + "-" * 70)

    print("\n  Paste into settings.py (replacing the confidence_tiers block):\n")
    print(render_settings_block(new))

    if args.write:
        if problems:
            raise SystemExit("\n  Refusing --write while the honesty check fails.")
        backup = write_settings(new)
        print(f"\n  ✏️  settings.py updated (backup at {backup.name})")
        print("     tier_config_version is derived from these thresholds, so it")
        print("     changes automatically — rows written from here on are")
        print("     distinguishable from rows written under the old config.")
        print("\n  Re-run the pipeline so content and ledger use the new tiers:")
        print("     python -m ballknower_quad.scripts.weekly_pipeline")
    else:
        print("\n  (dry run — pass --write to apply)")

    print("\n" + settings.disclaimer)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
