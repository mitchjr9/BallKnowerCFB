"""
ballknower_quad.ledger.score
============================

The public calibration surface. Joins forecasts to resolutions and computes the
scores **on demand** — nothing here is stored back into the ledger.

    python -m ballknower_quad.ledger.score
    python -m ballknower_quad.ledger.score --write        # -> calibration.md
    python -m ballknower_quad.ledger.score --include-backfill

Two rules inherited from the schema:

* **Scores are a view, not a column.** Freezing Brier into the data would lock in
  one metric choice forever and make rescoring a history rewrite. Everything
  below is recomputed from (forecast, resolution) every time it runs.
* **Backfill is quarantined by default.** Rows committed after the event are
  legitimate for model development and worthless as evidence of foresight.
  Mixing them into a public curve is the single easiest way to accidentally
  publish a track record that isn't one, so `--include-backfill` is opt-in and
  labelled wherever it appears.

Reported per family, never averaged across them:

* probabilistic — Brier, log-loss, ECE, accuracy, and the tier table
* distributional — CRPS against the forecast Normal, plus interval coverage
  (what share of realized margins actually landed inside ±1σ and ±2σ)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ballknower_quad.settings import settings


# --------------------------------------------------------------------------- #
# Scoring primitives
# --------------------------------------------------------------------------- #
def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def crps_normal(mu: float, sigma: float, y: float) -> float:
    """
    Closed-form CRPS for a Normal forecast.

    The distributional analog of Brier: it rewards a sharp distribution centered
    on the truth and penalizes both bias and overconfidence, in the same units as
    the quantity being predicted (points). Lower is better.
    """
    if sigma <= 0:
        return float("inf")
    z = (y - mu) / sigma
    return sigma * (z * (2 * _norm_cdf(z) - 1) + 2 * _norm_pdf(z)
                    - 1.0 / math.sqrt(math.pi))


def _ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


# --------------------------------------------------------------------------- #
# Load + join
# --------------------------------------------------------------------------- #
def load_ledger(include_backfill: bool = False) -> pd.DataFrame:
    root = settings.content_root
    f_rows, r_rows = [], []
    for d in sorted(root.iterdir()):
        led = d / "ledger"
        fp, rp = led / "forecasts.jsonl", led / "resolutions.jsonl"
        if not fp.exists():
            continue
        for line in fp.read_text().splitlines():
            if line:
                rec = json.loads(line)
                rec["batch"] = d.name
                f_rows.append(rec)
        if rp.exists():
            for line in rp.read_text().splitlines():
                if line:
                    r_rows.append(json.loads(line))

    if not f_rows:
        return pd.DataFrame()

    f = pd.DataFrame(f_rows)
    if not include_backfill:
        f = f[f["provenance"] == "live"]
    if not r_rows:
        f["resolution_status_r"] = None
        return f

    r = pd.DataFrame(r_rows)[
        ["forecast_id", "resolution_status", "outcome_binary", "outcome_value",
         "outcome_text", "resolved_ts"]].rename(
        columns={"resolution_status": "resolution_status_r"})
    return f.merge(r, on="forecast_id", how="left", suffixes=("", "_r"))


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #
def probabilistic_report(df: pd.DataFrame) -> dict:
    d = df[(df["record_family"] == "probabilistic")
           & (df["resolution_status_r"] == "resolved")].copy()
    if d.empty:
        return {}
    p = d["prob"].astype(float).to_numpy()
    y = d["outcome_binary"].astype(float).to_numpy()
    eps = 1e-9
    return {
        "n": int(len(d)),
        "accuracy": float((y == 1).mean()),   # subject is always the pick
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p + eps)
                                   + (1 - y) * np.log(1 - p + eps))),
        "ece": _ece(p, y),
        "avg_confidence": float(p.mean()),
        "frame": d,
    }


def tier_report(d: pd.DataFrame, by_depth: bool = True) -> pd.DataFrame:
    """
    Tier reliability, segmented by data depth.

    Pooling across depths quietly corrupts this table. The depth cap means a
    "Lean" in Week 1 (depth `none`, capped down from a 94% forecast) and a
    "Lean" in Week 8 (depth `rich`, genuinely ~75%) are different animals sharing
    a label. Averaging them produces a row that describes neither, and the more
    weeks accumulate the more misleading it gets. `tier_config_version` catches
    threshold changes but not this — the cap is a second dimension, so the report
    has to carry it.
    """
    depth_order = {"rich": 0, "thin": 1, "none": 2}
    rows = []
    groups = (["rich", "thin", "none"] if by_depth else [None])
    for depth in groups:
        for tier in ["Lock", "Strong", "Lean", "Pass"]:
            m = d["tier"] == tier
            if depth is not None:
                m = m & (d["data_depth"] == depth)
            if not m.any():
                continue
            p = d.loc[m, "prob"].astype(float).to_numpy()
            y = d.loc[m, "outcome_binary"].astype(float).to_numpy()
            hit, conf = float(y.mean()), float(p.mean())
            se = math.sqrt(max(hit * (1 - hit), 1e-9) / len(y))
            row = {"tier": tier, "picks": len(y),
                   "advertised": round(conf, 3), "hit_rate": round(hit, 3),
                   "gap": round(hit - conf, 3), "se": round(se, 3)}
            if depth is not None:
                row = {"depth": depth, **row}
            rows.append(row)
    df = pd.DataFrame(rows)
    if by_depth and len(df):
        df = df.sort_values(["depth", "tier"],
                            key=lambda s: (s.map(depth_order)
                                           if s.name == "depth" else s))
    return df.reset_index(drop=True)


def distributional_report(df: pd.DataFrame) -> dict:
    d = df[(df["record_family"] == "distributional")
           & (df["resolution_status_r"] == "resolved")].copy()
    if d.empty:
        return {}
    mu = d["mu"].astype(float).to_numpy()
    sg = d["sigma"].astype(float).to_numpy()
    y = d["outcome_value"].astype(float).to_numpy()
    crps = np.array([crps_normal(m, s, v) for m, s, v in zip(mu, sg, y)])
    z = np.abs((y - mu) / sg)
    return {
        "n": int(len(d)),
        "crps": float(crps.mean()),
        "mae": float(np.mean(np.abs(y - mu))),
        "bias": float(np.mean(mu - y)),
        "cover_1sd": float((z <= 1).mean()),
        "cover_2sd": float((z <= 2).mean()),
    }


def detail_table(df: pd.DataFrame, batch: Optional[str] = None) -> pd.DataFrame:
    """
    Per-pick results: what we said, what happened, hit or miss.

    The aggregate metrics answer "is the model honest". This answers "how did we
    do last week", which is the question a reader actually asks — and it's the
    only view that makes a miss concrete rather than a rounding error in a Brier
    score. Sorted by confidence so the picks we staked the most on are on top,
    where a bad one can't hide.
    """
    d = df[(df["record_family"] == "probabilistic")
           & (df["resolution_status_r"] == "resolved")].copy()
    if batch:
        d = d[d["batch"] == batch]
    if d.empty:
        return pd.DataFrame()
    d["prob"] = d["prob"].astype(float)
    d = d.sort_values("prob", ascending=False)
    return pd.DataFrame({
        "hit": np.where(d["outcome_binary"].astype(float) == 1, "✅", "❌"),
        "pick": d["proposition_text"].str.replace(r"\s*\(\d{4}-\d{2}-\d{2}\)",
                                                 "", regex=True),
        "said": (d["prob"] * 100).round(0).astype(int).astype(str) + "%",
        "tier": d["tier"],
        "depth": d["data_depth"],
        "result": d["outcome_text"],
    })


def margin_detail(df: pd.DataFrame, batch: Optional[str] = None) -> pd.DataFrame:
    d = df[(df["record_family"] == "distributional")
           & (df["resolution_status_r"] == "resolved")].copy()
    if batch:
        d = d[d["batch"] == batch]
    if d.empty:
        return pd.DataFrame()
    d["err"] = d["mu"].astype(float) - d["outcome_value"].astype(float)
    d = d.reindex(d["err"].abs().sort_values(ascending=False).index)
    return pd.DataFrame({
        "matchup": d["proposition_text"].str.replace(
            r"^Final margin, ", "", regex=True).str.replace(
            r"\s*\(\d{4}-\d{2}-\d{2}\).*", "", regex=True),
        "predicted": d["mu"].astype(float).round(1),
        "actual": d["outcome_value"].astype(float).round(0),
        "miss_by": d["err"].round(1),
    })


def render(prob: dict, dist: dict, df: pd.DataFrame,
           include_backfill: bool) -> str:
    out = ["# BallKnower Quad — Calibration\n"]
    scope = "live forecasts only" if not include_backfill else \
            "**includes backfill rows — not a track record**"
    out.append(f"_Pre-registered college football forecasts, {scope}. "
               f"Every row was committed before kickoff and graded afterward "
               f"whether it worked or not._\n")

    pending = int((df["resolution_status_r"].isna()).sum())
    void = int((df["resolution_status_r"] == "void").sum())

    if prob:
        out.append("## Win probability\n")
        out.append(f"| metric | value |\n|---|---:|")
        out.append(f"| forecasts graded | {prob['n']} |")
        out.append(f"| hit rate | {prob['accuracy']:.3f} |")
        out.append(f"| average confidence | {prob['avg_confidence']:.3f} |")
        out.append(f"| Brier | {prob['brier']:.4f} |")
        out.append(f"| log loss | {prob['log_loss']:.4f} |")
        out.append(f"| calibration error (ECE) | {prob['ece']:.4f} |")
        out.append("")
        gap = prob["accuracy"] - prob["avg_confidence"]
        out.append(
            f"The number that matters is the gap between the last two: we said "
            f"**{prob['avg_confidence'] * 100:.1f}%** on average and hit "
            f"**{prob['accuracy'] * 100:.1f}%** "
            f"({gap:+.1%}). Being right more often than we claimed is as much a "
            f"calibration miss as the reverse — it just flatters us.\n")

        t = tier_report(prob["frame"])
        if len(t):
            out.append("### By confidence tier\n")
            out.append("| data depth | tier | picks | advertised | hit rate | "
                       "gap | ±1 SE |")
            out.append("|---|---|---:|---:|---:|---:|---:|")
            for r in t.itertuples(index=False):
                out.append(f"| {r.depth} | {r.tier} | {r.picks} | "
                           f"{r.advertised:.3f} | {r.hit_rate:.3f} | "
                           f"{r.gap:+.3f} | {r.se:.3f} |")
            out.append("")
            out.append(
                "**Data depth** is how much football the ratings are built on. "
                "In the opening weeks a team's rating comes mostly from "
                "preseason roster signals rather than results, so the model caps "
                "how confident a label it will publish — which is why an "
                "early-season tier can advertise a probability well above its "
                "name. Depths are reported separately because pooling them "
                "would average two different things.\n")

    if dist:
        out.append("## Margin\n")
        out.append(f"| metric | value |\n|---|---:|")
        out.append(f"| forecasts graded | {dist['n']} |")
        out.append(f"| CRPS | {dist['crps']:.3f} |")
        out.append(f"| mean absolute error | {dist['mae']:.2f} pts |")
        out.append(f"| bias (predicted − actual) | {dist['bias']:+.2f} pts |")
        out.append(f"| inside ±1σ | {dist['cover_1sd']:.3f} (target 0.683) |")
        out.append(f"| inside ±2σ | {dist['cover_2sd']:.3f} (target 0.954) |")
        out.append("")
        out.append("Coverage well under target means the margin model is "
                   "overconfident — the stated uncertainty is too narrow. Over "
                   "target means it's hedging.\n")

    recent = sorted(df["batch"].unique())[-1] if len(df) else None
    det = detail_table(df, batch=recent) if recent else pd.DataFrame()
    if len(det):
        wins = int((det["hit"] == "✅").sum())
        out.append(f"## Last slate ({recent}) — {wins}-{len(det) - wins}\n")
        out.append("| | pick | said | tier | result |")
        out.append("|---|---|---:|---|---|")
        for r in det.itertuples(index=False):
            out.append(f"| {r.hit} | {r.pick} | {r.said} | {r.tier} | "
                       f"{r.result} |")
        out.append("")

    if pending or void:
        out.append(f"\n_{pending} forecast(s) still open, {void} void._\n")
    out.append(f"\n---\n_{settings.disclaimer}_\n")
    return "\n".join(out)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score the forecast ledger.")
    ap.add_argument("--include-backfill", action="store_true",
                    help="include rows committed after the event. Off by "
                         "default: they are model-development artifacts, not "
                         "evidence of foresight.")
    ap.add_argument("--write", action="store_true",
                    help="write content/cfb/calibration.md")
    ap.add_argument("--detail", action="store_true",
                    help="show every graded pick and what happened")
    ap.add_argument("--batch", default=None,
                    help="limit --detail to one batch date, e.g. 2026-08-29")
    args = ap.parse_args(argv)

    df = load_ledger(include_backfill=args.include_backfill)
    if df.empty:
        raise SystemExit(
            "No forecasts found. Run:\n"
            "  python -m ballknower_quad.ledger.emit\n"
            "  python -m ballknower_quad.ledger.resolve")

    prob = probabilistic_report(df)
    dist = distributional_report(df)

    print("=" * 66)
    print("  FORECAST LEDGER — CALIBRATION"
          + ("  (INCLUDES BACKFILL)" if args.include_backfill else ""))
    print("=" * 66)
    print(f"  batches       : {df['batch'].nunique()}")
    print(f"  forecast rows : {len(df)}")
    graded = int((df["resolution_status_r"] == "resolved").sum())
    print(f"  graded        : {graded}  "
          f"(open {int(df['resolution_status_r'].isna().sum())}, "
          f"void {int((df['resolution_status_r'] == 'void').sum())})")

    if not prob and not dist:
        print("\n  Nothing graded yet — run ledger.resolve after the games "
              "finish.")
        return 0

    if prob:
        print(f"\n  WIN PROBABILITY  (n={prob['n']})")
        print(f"    hit rate        {prob['accuracy']:.4f}")
        print(f"    avg confidence  {prob['avg_confidence']:.4f}")
        print(f"    Brier           {prob['brier']:.4f}")
        print(f"    log loss        {prob['log_loss']:.4f}")
        print(f"    ECE             {prob['ece']:.4f}")
        t = tier_report(prob["frame"])
        if len(t):
            print("\n    By tier and data depth:")
            print("      " + t.to_string(index=False).replace("\n", "\n      "))
            depths = set(prob["frame"]["data_depth"])
            if depths == {"none"} or depths == {"thin"}:
                print("\n      Every row is early-season, so the depth cap has")
                print("      pushed high-confidence forecasts down into lower")
                print("      tiers — that's why a tier can advertise ~0.94.")
            bad = t[(t["gap"] < -0.02) & (t["gap"] < -2 * t["se"])]
            if len(bad):
                print("\n    ⚠ a tier is hitting materially below what it "
                      "advertises:")
                for r in bad.itertuples(index=False):
                    print(f"      {r.tier}: says {r.advertised:.3f}, "
                          f"hits {r.hit_rate:.3f}")
                print("      Re-run calibrate_tiers once you have a few more "
                      "weeks of data.")

    if dist:
        print(f"\n  MARGIN  (n={dist['n']})")
        print(f"    CRPS            {dist['crps']:.3f}")
        print(f"    MAE             {dist['mae']:.2f} pts")
        print(f"    bias            {dist['bias']:+.2f} pts")
        print(f"    inside ±1σ      {dist['cover_1sd']:.3f}  (target 0.683)")
        print(f"    inside ±2σ      {dist['cover_2sd']:.3f}  (target 0.954)")

    if args.detail:
        which = args.batch or (sorted(df["batch"].unique())[-1]
                               if len(df) else None)
        det = detail_table(df, batch=which)
        if len(det):
            wins = int((det["hit"] == "✅").sum())
            print(f"\n  PICKS — {which}   ({wins}-{len(det) - wins})")
            print("  " + det.to_string(index=False).replace("\n", "\n  "))
        md = margin_detail(df, batch=which)
        if len(md):
            print(f"\n  MARGIN — biggest misses first")
            print("  " + md.head(10).to_string(index=False)
                  .replace("\n", "\n  "))

    if prob and prob["n"] < 100:
        print(f"\n  ⚠ Only {prob['n']} graded forecasts. One week is far too few "
              f"to read calibration from —")
        print("    a 70% pick losing twice in a row means nothing. Publish the "
              "page, don't interpret it yet.")

    if args.write:
        page = settings.content_root / "calibration.md"
        page.write_text(render(prob, dist, df, args.include_backfill))
        print(f"\n  → {page}")

    print("\n" + settings.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
