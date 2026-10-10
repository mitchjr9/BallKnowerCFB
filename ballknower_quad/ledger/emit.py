"""
ballknower_quad.ledger.emit
===========================

Turns a Quad slate into **forecast ledger rows**: append-only JSONL plus a
manifest carrying the chain head, ready to be git-committed for third-party
timestamp attestation.

    # after weekly_pipeline has written predictions.json
    python -m ballknower_quad.ledger.emit
    python -m ballknower_quad.ledger.emit --date 2026-08-29
    python -m ballknower_quad.ledger.emit --provenance backfill

Each game emits **two** records:

  * ``moneyline``  — probabilistic, scored by Brier / log-loss
  * ``spread``     — distributional (mu = predicted margin, sigma = model RMSE),
                     scored by CRPS and interval coverage

Two rows rather than one because they resolve independently: a pick can be right
on the winner and wrong on the number. Merging them would make the reliability
curve uncomputable without unpacking, and the two families use different metrics
that must never be averaged into a single headline.

Rows that violate the leakage invariant — anything where kickoff has already
passed — are **refused and reported**, not silently dropped. If you run this
after Saturday's games start, you should see the refusals.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import List, Optional

from ballknower_quad.ledger.emit_records import (SCHEMA_VERSION, ForecastRecord,
                                                 LedgerError, build_manifest,
                                                 code_commit, config_hash,
                                                 compute_tier, data_depth_for,
                                                 event_id, moneyline_text,
                                                 now_iso, spread_text, team_id)
from ballknower_quad.settings import settings

# Fallback spread uncertainty when the trained margin model's RMSE isn't
# recorded alongside the slate. ~16 points matches the walk-forward RMSE.
DEFAULT_MARGIN_SIGMA = 16.0


def _load_predictions(date_str: str) -> tuple[List[dict], Path]:
    d = settings.content_root / date_str
    p = d / "predictions.json"
    if not p.exists():
        raise SystemExit(
            f"No predictions at {p}\n"
            f"Run: python -m ballknower_quad.scripts.weekly_pipeline")
    return json.loads(p.read_text()), d


def _margin_sigma() -> float:
    """Read RMSE off the trained margin model so sigma isn't a guess."""
    meta = (settings.models_dir_for(settings.active_model_version)
            / "margin_meta.json")
    try:
        m = json.loads(meta.read_text()).get("metrics", {})
        rmse = float(m.get("rmse", 0.0))
        if rmse > 0:
            return rmse
    except Exception:  # noqa: BLE001
        pass
    return DEFAULT_MARGIN_SIGMA


def records_for_game(g: dict, *, asof_ts: str, committed_ts: str,
                     provenance: str, sigma: float,
                     model_version: str) -> List[ForecastRecord]:
    home, away = g["home_team"], g["away_team"]
    # Use the real kickoff instant. The date-only fallback survives only for
    # batches written before `kickoff_ts` existed, and it deliberately errs
    # EARLY: an assumed kickoff before the real one can only cause a wrongful
    # refusal, never a wrongful claim of pre-registration. That's the right
    # direction for a track record to fail in — it just must not be the normal
    # path, which is what it silently was.
    kickoff = g.get("kickoff_ts") or ""
    if not kickoff:
        d = g.get("game_date") or ""
        kickoff = f"{d}T12:00:00+00:00" if len(d) == 10 else d
        g["_kickoff_guessed"] = True

    ev = event_id(g.get("game_id"), g.get("season"), g.get("week"), home, away)
    neutral = bool(g.get("neutral_site"))
    depth = data_depth_for(g.get("games_played_min"))

    p_home = float(g["p_blended"])
    favorite = home if p_home >= 0.5 else away
    fav_prob = max(p_home, 1.0 - p_home)
    common = dict(
        schema_version=SCHEMA_VERSION, event_id=ev,
        data_depth=depth, asof_ts=asof_ts, created_ts=now_iso(),
        committed_ts=committed_ts, event_start_ts=kickoff,
        model_version=model_version, config_hash=config_hash(),
        code_commit=code_commit(), provenance=provenance,
    )

    out: List[ForecastRecord] = []

    # 1) moneyline — probabilistic
    out.append(ForecastRecord(
        forecast_id="", record_family="probabilistic", market_type="moneyline",
        subject_id=team_id(favorite),
        opponent_id=team_id(away if favorite == home else home),
        side="home" if favorite == home else "away",
        line=None,
        proposition_text=moneyline_text(
            favorite, away if favorite == home else home, neutral, kickoff),
        prob=fav_prob,
        tier=compute_tier(fav_prob, depth),
        **common))

    # 2) margin — distributional
    mu = g.get("predicted_margin")
    if mu is not None:
        out.append(ForecastRecord(
            forecast_id="", record_family="distributional", market_type="spread",
            subject_id=team_id(home), opponent_id=team_id(away),
            side="home", line=g.get("spread_line"),
            proposition_text=spread_text(home, away, float(mu), kickoff),
            mu=float(mu), sigma=sigma,
            tier="Pass",              # tiers are a probabilistic-family concept
            **common))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Emit forecast-ledger records.")
    ap.add_argument("--date", default=dt.date.today().isoformat(),
                    help="content/cfb/<date>/ folder to read (default: today)")
    ap.add_argument("--provenance", default="live",
                    choices=["live", "backfill", "replay"],
                    help="'live' rows are the only ones a public calibration "
                         "surface may show")
    ap.add_argument("--asof", default=None,
                    help="ISO timestamp of the latest data the model saw "
                         "(default: read from the slate)")
    args = ap.parse_args(argv)

    preds, out_dir = _load_predictions(args.date)
    if not preds:
        raise SystemExit("predictions.json is empty")

    asof = args.asof or preds[0].get("asof_ts") or now_iso()
    committed = now_iso()
    sigma = _margin_sigma()
    model_version = settings.active_model_version

    ledger_dir = out_dir / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    jsonl = ledger_dir / "forecasts.jsonl"

    # Append-only. This used to rewrite forecasts.jsonl from scratch, so
    # re-running emit on a batch that had already been graded would have
    # destroyed the rows its resolutions point at. Existing rows are now
    # immutable: we only add forecast_ids not already present, and the hash
    # chain continues from the last row on disk.
    existing_ids: set = set()
    prev_hash = None
    if jsonl.exists():
        lines = [l for l in jsonl.read_text().splitlines() if l]
        for l in lines:
            existing_ids.add(json.loads(l)["forecast_id"])
        if lines:
            prev_hash = json.loads(lines[-1]).get("row_hash")

    records: List[ForecastRecord] = []
    refused: List[tuple] = []
    skipped_existing = 0
    guessed = 0

    for g in preds:
        try:
            recs = records_for_game(
                    g, asof_ts=asof, committed_ts=committed,
                    provenance=args.provenance, sigma=sigma,
                    model_version=model_version)
            guessed += int(bool(g.get("_kickoff_guessed")))
            for rec in recs:
                if rec.compute_id() in existing_ids:
                    skipped_existing += 1
                    continue
                rec.finalize(prev_hash)
                prev_hash = rec.row_hash
                records.append(rec)
        except LedgerError as e:
            refused.append((f"{g.get('away_team')} @ {g.get('home_team')}",
                            str(g.get("game_date", "")), str(e)))

    if records:
        with jsonl.open("a") as fh:
            for r in records:
                fh.write(r.to_json() + "\n")

    # Manifest reflects the WHOLE batch on disk, not just this run's additions.
    from types import SimpleNamespace
    all_rows = [SimpleNamespace(**json.loads(l)) for l in
                jsonl.read_text().splitlines() if l] if jsonl.exists() else []
    manifest = build_manifest(all_rows, batch_label=f"cfb-{args.date}")
    (ledger_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    n_games = len(preds)
    games_written = len({r.event_id for r in records})
    games_refused = len({name for name, _, _ in refused})

    print("=" * 62)
    print(f"  FORECAST LEDGER — cfb-{args.date}")
    print("=" * 62)
    print(f"  slate           : {n_games} games")
    print(f"  emitted         : {games_written} games "
          f"({len(records)} records)")
    if skipped_existing:
        print(f"  already on file : {skipped_existing} records "
              f"(append-only — left untouched)")
    if games_refused:
        print(f"  refused         : {games_refused} games")
    print(f"  batch total     : {manifest['record_count']} records  "
          f"({manifest['family_counts']})")
    print(f"  provenance      : {args.provenance}")
    print(f"  tiers           : {manifest['tier_counts']}")
    print(f"  asof_ts         : {asof}")
    print(f"  committed_ts    : {committed}")
    print(f"  chain head      : {manifest['chain_head'][:16]}…"
          if manifest["chain_head"] else "  chain head      : (none)")
    print(f"  config_hash     : {manifest['config_hash']}  "
          f"code_commit: {manifest['code_commit']}")
    print(f"\n  → {jsonl}")
    print(f"  → {ledger_dir / 'manifest.json'}")

    if refused:
        started = [r for r in refused if "already kicked off" in r[2]]
        other = [r for r in refused if r not in started]

        if started:
            share = len(started) / max(n_games, 1)
            loud = share > 0.20
            head = "⛔" if loud else "ⓘ"
            print(f"\n  {head} {len(started)} of {n_games} game(s) refused as "
                  f"already kicked off ({share:.0%}):")
            for name, when, _ in started[:8]:
                print(f"      {name}  (kickoff {when})")
            if len(started) > 8:
                print(f"      … and {len(started) - 8} more")
            if loud:
                # A handful of Thursday/Friday games is normal. A majority of
                # the slate is not — it means either the run really was late,
                # or the kickoff times are wrong. Don't reassure; make the user
                # check which.
                print("    That's most of the slate. Before accepting it, check")
                print("    the kickoff times above against the real schedule —")
                print("    if games listed here hadn't started when you ran this,")
                print("    the timestamps are wrong, not the timing.")
            else:
                print("    Normal for a few early-week games; a forecast "
                      "committed after kickoff isn't a forecast.")

        if other:
            print(f"\n  ⚠ {len(other)} game(s) REFUSED for other reasons:")
            for name, when, why in other[:6]:
                print(f"      {name} ({when}): {why}")

    if guessed:
        print(f"\n  ⚠ {guessed} game(s) had no kickoff time and fell back to a "
              f"noon-UTC guess.")
        print("    That guess refuses early rather than wrongly accepting, but it")
        print("    shouldn't happen on a fresh slate — re-run weekly_pipeline.")

    print("\n  Next: commit the manifest so the timestamp is third-party "
          "attested —")
    try:
        shown = ledger_dir.relative_to(settings.project_root)
    except ValueError:
        shown = ledger_dir          # content root configured outside the repo
    print(f"    git add {shown} && \\")
    print(f"    git commit -m 'ledger: cfb {args.date}' && git push")
    print("\n" + settings.disclaimer)
    # Non-zero exit when most of the slate was refused, so a script or a
    # glance at the terminal can't mistake it for a clean run.
    return 2 if (refused and len(refused) / max(n_games, 1) > 0.20) else 0


if __name__ == "__main__":
    raise SystemExit(main())
