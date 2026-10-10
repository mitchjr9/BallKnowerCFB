"""
ballknower_quad.ledger.resolve
==============================

The grader. Looks up what actually happened and writes **resolution records**.

    python -m ballknower_quad.ledger.resolve                 # today's batch
    python -m ballknower_quad.ledger.resolve --date 2026-08-29
    python -m ballknower_quad.ledger.resolve --all           # every open batch

The one rule that governs this whole file: **it never modifies
``forecasts.jsonl``.** Resolutions go to a separate append-only
``resolutions.jsonl`` beside it, joined on ``forecast_id``. That separation is
what makes the track record auditable — if grading could edit the forecast, then
"we predicted 62%" and "we were right" would live in the same mutable row and
nobody outside would have reason to believe either. Anyone can re-run this and
get the same answer, or diff the resolutions against their own.

A second consequence: **scores are never stored here.** No Brier column, no
"correct" flag beyond the raw outcome. Resolution records what happened; scoring
is a pure function of (forecast, resolution) computed on demand by
``ledger.score``. Storing a score would freeze a metric choice into the data and
make it impossible to rescore later without rewriting history.

Statuses:
  * ``resolved`` — the game finished and the proposition is decided
  * ``void``     — cancelled, or the data says something impossible (a tie in
                   modern CFB means a data error, not a draw)
  * ``pending``  — not played yet, postponed, or not found upstream. Written
                   nowhere; the forecast simply stays open.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ballknower_quad.data.cfbd_loader import load_games
from ballknower_quad.ledger.emit_records import (RESOLUTION_POLICY_VERSION,
                                                 now_iso, slug)
from ballknower_quad.settings import settings

SOURCE = "cfbd:/games"


@dataclass
class Resolution:
    forecast_id: str
    event_id: str
    resolution_status: str          # resolved | void
    outcome_binary: Optional[int] = None    # probabilistic: 1 / 0
    outcome_value: Optional[float] = None   # distributional: realized value
    outcome_text: str = ""                  # human-readable, for auditing
    resolved_ts: str = field(default_factory=now_iso)
    source: str = SOURCE
    source_asof_ts: str = ""
    policy_version: str = RESOLUTION_POLICY_VERSION
    note: str = ""
    prev_hash: Optional[str] = None
    row_hash: str = ""

    def finalize(self, prev_hash: Optional[str]) -> "Resolution":
        self.prev_hash = prev_hash
        body = {k: v for k, v in asdict(self).items() if k != "row_hash"}
        self.row_hash = hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


# --------------------------------------------------------------------------- #
# Result lookup
# --------------------------------------------------------------------------- #
def _game_key(event_id: str) -> Optional[str]:
    """`cfb:event:401752701` -> `401752701`."""
    parts = str(event_id).split(":")
    return parts[-1] if len(parts) >= 3 else None


def composite_key(season, week, home: str, away: str) -> Optional[str]:
    """Mirror of the fallback branch in ``emit_records.event_id``."""
    if pd.isna(season) or pd.isna(week):
        return None
    return f"{int(season)}_w{int(week)}_{slug(away)}_at_{slug(home)}"


def build_results_index(seasons: List[int]) -> Dict[str, dict]:
    """
    Index results under EVERY key shape an event_id can take.

    ``event_id()`` uses the CFBD numeric game id when it has one and falls back
    to a ``{season}_w{week}_{away}_at_{home}`` composite when it doesn't. Early
    batches were emitted before the pipeline carried ``game_id`` through, so
    their rows are keyed the composite way — and because forecasts are
    append-only and those games have been played, they can't be re-emitted to
    fix the key (a fresh row would be refused as post-kickoff). The reader has to
    be tolerant of both instead. That's the usual shape of an append-only
    ledger: you widen the reader rather than rewrite history.
    """
    games = load_games(seasons)
    idx: Dict[str, dict] = {}
    n_num = n_comp = 0
    for r in games.itertuples(index=False):
        rec = {
            "home_team": r.home_team, "away_team": r.away_team,
            "home_points": r.home_points, "away_points": r.away_points,
            "completed": bool(r.completed),
        }
        if pd.notna(r.game_id):
            idx[str(int(r.game_id))] = rec
            n_num += 1
        ck = composite_key(r.season, r.week, r.home_team, r.away_team)
        if ck:
            idx.setdefault(ck, rec)
            n_comp += 1
    idx["__counts__"] = {"numeric": n_num, "composite": n_comp}  # for diagnostics
    return idx


# --------------------------------------------------------------------------- #
# Resolution logic
# --------------------------------------------------------------------------- #
def resolve_one(fc: dict, result: Optional[dict],
                source_asof: str) -> Optional[Resolution]:
    """Return a Resolution, or None to leave the forecast open."""
    ev = fc["event_id"]
    fid = fc["forecast_id"]

    if result is None:
        return None                              # not found upstream yet
    if not result["completed"]:
        return None                              # not played yet
    hp, ap = result["home_points"], result["away_points"]
    if hp is None or ap is None or pd.isna(hp) or pd.isna(ap):
        return None

    hp, ap = float(hp), float(ap)
    margin = hp - ap

    if margin == 0:
        # Modern CFB decides every game in overtime, so a reported tie means the
        # feed is wrong, not that the game drew. Void rather than guess.
        return Resolution(forecast_id=fid, event_id=ev, resolution_status="void",
                          outcome_text=f"reported tie {hp:.0f}-{ap:.0f}",
                          source_asof_ts=source_asof,
                          note="tie is impossible in modern CFB; treated as a "
                               "data error").finalize(None)

    if fc["record_family"] == "distributional":
        # `side` is always home for our spread rows: realized home-minus-away.
        return Resolution(
            forecast_id=fid, event_id=ev, resolution_status="resolved",
            outcome_value=margin,
            outcome_text=(f"{result['home_team']} {hp:.0f}, "
                          f"{result['away_team']} {ap:.0f} "
                          f"(margin {margin:+.0f})"),
            source_asof_ts=source_asof).finalize(None)

    # probabilistic: did the SUBJECT team win?
    winner = result["home_team"] if margin > 0 else result["away_team"]
    subject_slug = str(fc["subject_id"]).split(":")[-1]
    won = int(slug(winner) == subject_slug)

    # Guard against a silent identity mismatch — a subject that matches neither
    # side means the alias table is wrong, and scoring it would be worse than
    # leaving it open.
    if subject_slug not in (slug(result["home_team"]), slug(result["away_team"])):
        return Resolution(
            forecast_id=fid, event_id=ev, resolution_status="void",
            outcome_text=f"subject {subject_slug!r} not in this game",
            source_asof_ts=source_asof,
            note="entity identity mismatch — check the alias table"
            ).finalize(None)

    return Resolution(
        forecast_id=fid, event_id=ev, resolution_status="resolved",
        outcome_binary=won,
        outcome_text=(f"{winner} won {max(hp, ap):.0f}-{min(hp, ap):.0f}"),
        source_asof_ts=source_asof).finalize(None)


# --------------------------------------------------------------------------- #
# Batch handling
# --------------------------------------------------------------------------- #
def batch_dirs(date: Optional[str], every: bool) -> List[Path]:
    root = settings.content_root
    if every:
        return sorted(d / "ledger" for d in root.iterdir()
                      if d.is_dir() and (d / "ledger" / "forecasts.jsonl").exists())
    d = root / (date or dt.date.today().isoformat()) / "ledger"
    return [d] if (d / "forecasts.jsonl").exists() else []


def resolve_batch(ledger_dir: Path, results: Dict[str, dict],
                  source_asof: str) -> dict:
    forecasts = [json.loads(l) for l in
                 (ledger_dir / "forecasts.jsonl").read_text().splitlines() if l]
    res_path = ledger_dir / "resolutions.jsonl"

    existing: Dict[str, dict] = {}
    if res_path.exists():
        for line in res_path.read_text().splitlines():
            if line:
                r = json.loads(line)
                existing[r["forecast_id"]] = r

    prev_hash = None
    if existing:
        prev_hash = list(existing.values())[-1].get("row_hash")

    new: List[Resolution] = []
    still_open = 0
    for fc in forecasts:
        if fc["forecast_id"] in existing:
            continue                     # append-only: never regrade
        r = resolve_one(fc, results.get(_game_key(fc["event_id"])), source_asof)  # noqa: E501
        if r is None:
            still_open += 1
            continue
        r.finalize(prev_hash)
        prev_hash = r.row_hash
        new.append(r)

    if new:
        with res_path.open("a") as fh:
            for r in new:
                fh.write(r.to_json() + "\n")

    return {
        "dir": ledger_dir,
        "forecasts": len(forecasts),
        "already": len(existing),
        "new": len(new),
        "resolved": sum(1 for r in new if r.resolution_status == "resolved"),
        "void": sum(1 for r in new if r.resolution_status == "void"),
        "open": still_open,
        "sample_keys": [_game_key(f["event_id"]) for f in forecasts[:3]],
        "unmatched": [_game_key(f["event_id"]) for f in forecasts
                      if f["forecast_id"] not in existing
                      and results.get(_game_key(f["event_id"])) is None][:3],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Grade pre-registered forecasts.")
    ap.add_argument("--date", default=None, help="batch folder (default: today)")
    ap.add_argument("--all", action="store_true", dest="every",
                    help="grade every batch that still has open forecasts")
    ap.add_argument("--seasons", type=int, nargs="*", default=None,
                    help="seasons to pull results for (default: current year)")
    args = ap.parse_args(argv)

    dirs = batch_dirs(args.date, args.every)
    if not dirs:
        raise SystemExit(
            "No forecast batch found. Expected "
            f"{settings.content_root}/<date>/ledger/forecasts.jsonl\n"
            "Run: python -m ballknower_quad.ledger.emit")

    seasons = args.seasons or [dt.datetime.now().year]
    print(f"Pulling results for {seasons} "
          f"({'CFBD' if settings.has_cfbd_key else 'SYNTHETIC'}) …")
    results = build_results_index(seasons)
    source_asof = now_iso()
    counts = results.pop("__counts__", {})
    print(f"  {counts.get('numeric', 0):,} games indexed by CFBD id, "
          f"{counts.get('composite', 0):,} by season/week/teams\n")

    print("=" * 64)
    print("  RESOLUTION")
    print("=" * 64)
    totals = {"new": 0, "resolved": 0, "void": 0, "open": 0}
    unmatched_samples: List[str] = []
    for d in dirs:
        s = resolve_batch(d, results, source_asof)
        for k in totals:
            totals[k] += s[k]
        unmatched_samples.extend(s.get("unmatched", []))
        label = d.parent.name
        print(f"  {label}: {s['forecasts']:3d} forecasts · "
              f"{s['new']:3d} newly graded "
              f"({s['resolved']} resolved, {s['void']} void) · "
              f"{s['open']:3d} still open"
              + (f" · {s['already']} previously graded" if s["already"] else ""))

    if unmatched_samples:
        print("\n  ⚠ Some forecasts matched no game in the results index.")
        print(f"    Their event keys look like: {unmatched_samples[:3]}")
        print("    If those are season/week composites and the index above shows")
        print("    composite keys, check that the season passed with --seasons")
        print("    covers them. If they look like numeric ids, the results pull")
        print("    may be for the wrong season.")

    print("-" * 64)
    print(f"  newly graded {totals['new']} "
          f"({totals['resolved']} resolved, {totals['void']} void), "
          f"{totals['open']} still open")
    if totals["open"]:
        print("  Open forecasts are games not yet final upstream. Re-run after "
              "they finish; grading is append-only, so nothing regrades.")
    print("\n  forecasts.jsonl was NOT modified — resolutions live beside it.")
    print("\n  Next:")
    print("    python -m ballknower_quad.ledger.score")
    print(f"    git add {settings.content_root.name} && "
          f"git commit -m 'ledger: resolve {dt.date.today()}' && git push")
    print("\n" + settings.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
