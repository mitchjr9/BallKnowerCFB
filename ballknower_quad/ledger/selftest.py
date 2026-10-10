"""
ballknower_quad.ledger.selftest
===============================

Invariant suite for Quad's ledger emitter, in the same spirit as the schema's
own regression tests: every rule that matters is asserted here, including the
ones that are supposed to *fail*.

A validator that has never rejected anything is not evidence it works.

    python -m ballknower_quad.ledger.selftest
"""
from __future__ import annotations

from typing import List, Tuple

from ballknower_quad.ledger.emit_records import (ForecastRecord, LedgerError,
                                                 build_manifest, compute_tier,
                                                 data_depth_for, event_id,
                                                 team_id)

PASSED: List[str] = []
FAILED: List[Tuple[str, str]] = []


def check(name: str, fn) -> None:
    try:
        fn()
        PASSED.append(name)
    except AssertionError as e:
        FAILED.append((name, f"assertion: {e}"))
    except Exception as e:  # noqa: BLE001
        FAILED.append((name, f"{type(e).__name__}: {e}"))


def expect_reject(name: str, fn, must_contain: str = "") -> None:
    try:
        fn()
    except LedgerError as e:
        if must_contain and must_contain.lower() not in str(e).lower():
            FAILED.append((name, f"rejected, but not for the expected reason: {e}"))
        else:
            PASSED.append(name)
        return
    except Exception as e:  # noqa: BLE001
        FAILED.append((name, f"wrong exception type: {type(e).__name__}: {e}"))
        return
    FAILED.append((name, "row was ACCEPTED but should have been rejected"))


def _base(**over) -> ForecastRecord:
    d = dict(
        forecast_id="", schema_version="1", record_family="probabilistic",
        market_type="moneyline",
        subject_id=team_id("Ohio State"), opponent_id=team_id("Texas"),
        event_id=event_id(401752701, 2026, 1, "Ohio State", "Texas"),
        side="home", line=None,
        proposition_text="Ohio State defeats Texas (2026-08-30)",
        prob=0.663, tier=compute_tier(0.663, "none"), data_depth="none",
        asof_ts="2026-08-26T00:00:00+00:00",
        created_ts="2026-08-27T10:00:00+00:00",
        committed_ts="2026-08-27T10:00:00+00:00",
        event_start_ts="2026-08-30T16:00:00+00:00",
        model_version="v2", config_hash="abc123", code_commit="deadbee",
        provenance="live",
    )
    d.update(over)
    # Tier is a pure function, so the fixture must derive it too — hardcoding a
    # tier here would just be testing that the fixture agrees with itself.
    if "tier" not in over and d.get("record_family") == "probabilistic":
        d["tier"] = compute_tier(d["prob"], d["data_depth"])
    return ForecastRecord(**d)


def run() -> int:
    # --- identity ------------------------------------------------------- #
    check("team_id is namespaced and slugged",
          lambda: _assert(team_id("Ohio State") == "cfb:team:ohio_state"))
    check("event_id prefers the CFBD game id",
          lambda: _assert(event_id(123, 2026, 1, "A", "B") == "cfb:event:123"))
    check("event_id falls back deterministically",
          lambda: _assert(event_id(None, 2026, 1, "Ohio State", "Texas")
                          == "cfb:event:2026_w1_texas_at_ohio_state"))

    # --- happy path ----------------------------------------------------- #
    check("valid live row is accepted",
          lambda: _base().finalize(None))
    check("forecast_id is deterministic across runs",
          lambda: _assert(_base().finalize(None).forecast_id
                          == _base().finalize(None).forecast_id))
    check("changing content changes the id",
          lambda: _assert(_base().finalize(None).forecast_id
                          != _base(line=-3.5).finalize(None).forecast_id))
    check("hash chain links rows",
          lambda: _chain_links())

    # --- the invariants ------------------------------------------------- #
    expect_reject("leakage: asof after event_start",
                  lambda: _base(asof_ts="2026-09-01T00:00:00+00:00").finalize(None),
                  "leakage")
    expect_reject("pre-registration: committed after kickoff (live)",
                  lambda: _base(committed_ts="2026-08-31T00:00:00+00:00"
                                ).finalize(None),
                  "already kicked off")
    check("same row is legal as backfill",
          lambda: _base(committed_ts="2026-08-31T00:00:00+00:00",
                        provenance="backfill").finalize(None))
    expect_reject("backfill still cannot see the future",
                  lambda: _base(asof_ts="2026-09-05T00:00:00+00:00",
                                committed_ts="2026-09-06T00:00:00+00:00",
                                provenance="backfill").finalize(None),
                  "leakage")

    # --- families ------------------------------------------------------- #
    expect_reject("probabilistic row may not carry mu",
                  lambda: _base(mu=7.5).finalize(None), "must not carry mu")
    expect_reject("distributional row may not carry prob",
                  lambda: _base(record_family="distributional", mu=7.5,
                                sigma=16.0).finalize(None), "must not carry prob")
    check("valid distributional row is accepted",
          lambda: _base(record_family="distributional", market_type="spread",
                        prob=None, mu=7.5, sigma=16.0, tier="Pass"
                        ).finalize(None))
    expect_reject("sigma must be positive",
                  lambda: _base(record_family="distributional", prob=None,
                                mu=7.5, sigma=0.0).finalize(None), "sigma")
    expect_reject("prob must be inside (0,1)",
                  lambda: _base(prob=1.0).finalize(None), "prob")

    # --- tiering -------------------------------------------------------- #
    check("depth mapping",
          lambda: _assert(data_depth_for(0) == "none"
                          and data_depth_for(3) == "thin"
                          and data_depth_for(9) == "rich"))
    check("thin data caps the tier below Lock",
          lambda: _assert(compute_tier(0.97, "thin") != "Lock"))
    check("no data caps the tier at Lean",
          lambda: _assert(compute_tier(0.99, "none") == "Lean"))
    check("rich data allows Lock",
          lambda: _assert(compute_tier(0.97, "rich") == "Lock"))
    expect_reject("hand-set tier is rejected",
                  lambda: _base(tier="Lock").finalize(None), "hand-set")
    check("tier tracks the configured thresholds",
          lambda: _assert(_base(prob=0.95, data_depth="rich").finalize(None).tier
                          == "Lock"))

    # --- provenance / manifest ------------------------------------------ #
    expect_reject("unknown provenance is rejected",
                  lambda: _base(provenance="guess").finalize(None), "provenance")
    check("manifest carries chain head and counts", _manifest_ok)

    # --- resolution (the grader) ----------------------------------------- #
    check("subject won -> outcome 1", _resolve_win)
    check("subject lost -> outcome 0", _resolve_loss)
    check("distributional resolves to the realized margin", _resolve_margin)
    check("reported tie is voided, not scored", _resolve_tie)
    check("subject not in the game is voided", _resolve_mismatch)
    check("unfinished game stays open", _resolve_open)
    check("CRPS is minimized by a centred forecast", _crps_sane)
    check("grading is append-only (re-run adds nothing)", _append_only)
    check("composite event_ids resolve (pre-game_id batches)", _composite_key)
    check("game-day morning run keeps daytime games (real kickoff_ts)",
          _gameday_morning)
    check("emit is append-only (re-run never rewrites a batch)",
          _emit_append_only)
    check("dirty working tree is detected; clean tree is not", _dirty_guard)

    # --- every CLI parser must build ------------------------------------ #
    check("all CLI parsers build (argparse help strings are format-safe)",
          _all_parsers_build)

    # --- report --------------------------------------------------------- #
    print("=" * 62)
    print("  LEDGER EMITTER — INVARIANT SUITE")
    print("=" * 62)
    for n in PASSED:
        print(f"  ✓ {n}")
    for n, why in FAILED:
        print(f"  ✗ {n}\n      {why}")
    total = len(PASSED) + len(FAILED)
    print("-" * 62)
    print(f"  {len(PASSED)}/{total} passed")
    return 0 if not FAILED else 1


def _assert(cond: bool) -> None:
    if not cond:
        raise AssertionError("condition was false")


# --------------------------------------------------------------------------- #
# Resolution helpers
# --------------------------------------------------------------------------- #
def _fc(family="probabilistic", subject="Ohio State", **over):
    from ballknower_quad.ledger.emit_records import team_id
    d = {"forecast_id": "abc123", "event_id": "cfb:event:999",
         "record_family": family, "subject_id": team_id(subject),
         "side": "home"}
    d.update(over)
    return d


_RESULT = {"home_team": "Ohio State", "away_team": "Texas",
           "home_points": 31.0, "away_points": 17.0, "completed": True}


def _resolve_win() -> None:
    from ballknower_quad.ledger.resolve import resolve_one
    r = resolve_one(_fc(subject="Ohio State"), _RESULT, "ts")
    _assert(r is not None and r.resolution_status == "resolved")
    _assert(r.outcome_binary == 1)


def _resolve_loss() -> None:
    from ballknower_quad.ledger.resolve import resolve_one
    r = resolve_one(_fc(subject="Texas"), _RESULT, "ts")
    _assert(r.resolution_status == "resolved" and r.outcome_binary == 0)


def _resolve_margin() -> None:
    from ballknower_quad.ledger.resolve import resolve_one
    r = resolve_one(_fc(family="distributional"), _RESULT, "ts")
    _assert(r.resolution_status == "resolved")
    _assert(abs(r.outcome_value - 14.0) < 1e-9)
    _assert(r.outcome_binary is None)


def _resolve_tie() -> None:
    from ballknower_quad.ledger.resolve import resolve_one
    tied = dict(_RESULT, home_points=21.0, away_points=21.0)
    r = resolve_one(_fc(), tied, "ts")
    _assert(r.resolution_status == "void")


def _resolve_mismatch() -> None:
    from ballknower_quad.ledger.resolve import resolve_one
    r = resolve_one(_fc(subject="Michigan"), _RESULT, "ts")
    _assert(r.resolution_status == "void")


def _resolve_open() -> None:
    from ballknower_quad.ledger.resolve import resolve_one
    _assert(resolve_one(_fc(), dict(_RESULT, completed=False), "ts") is None)
    _assert(resolve_one(_fc(), None, "ts") is None)


def _crps_sane() -> None:
    from ballknower_quad.ledger.score import crps_normal
    centred = crps_normal(14.0, 16.0, 14.0)
    biased = crps_normal(-10.0, 16.0, 14.0)
    _assert(centred < biased)
    # closed form at y == mu is sigma * (2*phi(0) - 1/sqrt(pi))
    import math
    expected = 16.0 * (2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))
    _assert(abs(centred - expected) < 1e-9)


def _append_only() -> None:
    """Re-grading must be a no-op — the ledger is append-only by construction."""
    import json
    import tempfile
    from pathlib import Path
    from ballknower_quad.ledger.resolve import resolve_batch

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ledger"
        d.mkdir()
        rows = [_fc(forecast_id="f1"),
                _fc(forecast_id="f2", family="distributional")]
        (d / "forecasts.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n")
        results = {"999": _RESULT}

        first = resolve_batch(d, results, "ts")
        second = resolve_batch(d, results, "ts")
        _assert(first["new"] == 2)
        _assert(second["new"] == 0)
        lines = [l for l in (d / "resolutions.jsonl").read_text().splitlines() if l]
        _assert(len(lines) == 2)
        # forecasts file untouched
        _assert(len((d / "forecasts.jsonl").read_text().splitlines()) == 2)


def _composite_key() -> None:
    """
    Batches emitted before the pipeline carried `game_id` have event_ids of the
    form `cfb:event:2026_w1_away_at_home`. Those rows are already committed and
    their games are played, so they can never be re-emitted with a better key —
    a fresh row would be refused as post-kickoff. The grader must therefore
    resolve them as they are, forever.
    """
    import json
    import tempfile
    from pathlib import Path
    from ballknower_quad.ledger.emit_records import event_id, team_id
    from ballknower_quad.ledger.resolve import composite_key, resolve_batch

    home, away, season, week = "Ohio State", "Texas", 2026, 1
    ev = event_id(None, season, week, home, away)
    _assert(ev == "cfb:event:2026_w1_texas_at_ohio_state")

    fc = {"forecast_id": "f1", "event_id": ev, "record_family": "probabilistic",
          "subject_id": team_id(home), "side": "home"}
    rec = {"home_team": home, "away_team": away, "home_points": 31.0,
           "away_points": 17.0, "completed": True}

    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / "ledger"
        d.mkdir()
        (d / "forecasts.jsonl").write_text(json.dumps(fc) + "\n")

        # numeric-only index: the original failure — 0 of 1 graded
        _assert(resolve_batch(d, {"401752701": rec}, "ts")["new"] == 0)

        # with the composite key present it resolves
        idx = {"401752701": rec, composite_key(season, week, home, away): rec}
        s = resolve_batch(d, idx, "ts")
        _assert(s["new"] == 1 and s["resolved"] == 1)


def _pred(kickoff_ts=None, game_date="2026-09-26", home="Ohio State",
          away="Texas", gid="401900001"):
    return {"home_team": home, "away_team": away, "game_id": gid,
            "season": 2026, "week": 4, "neutral_site": False,
            "p_blended": 0.72, "predicted_margin": 7.5,
            "games_played_min": 3, "game_date": game_date,
            "kickoff_ts": kickoff_ts, "asof_ts": "2026-09-20T00:00:00+00:00"}


def _gameday_morning() -> None:
    """
    Reproduces the Week 4 incident: emit run at 8:01am Central on game day.
    With only a date, the emitter assumed a noon-UTC (7am Central) kickoff and
    refused an 11am game. With the real kickoff_ts it must accept it.
    """
    from ballknower_quad.ledger.emit import records_for_game
    committed = "2026-09-26T13:01:00+00:00"          # Sat 8:01am CT
    kw = dict(asof_ts="2026-09-20T00:00:00+00:00", committed_ts=committed,
              provenance="live", sigma=16.0, model_version="v2")

    # old shape (date only) -> guessed 12:00 UTC -> refused
    try:
        for r in records_for_game(_pred(kickoff_ts=None), **kw):
            r.finalize(None)
        raise AssertionError("date-only row should have been refused")
    except LedgerError:
        pass

    # new shape (real kickoff, 11am CT = 16:00 UTC) -> accepted
    recs = records_for_game(_pred(kickoff_ts="2026-09-26T16:00:00+00:00"), **kw)
    for r in recs:
        r.finalize(None)
    _assert(len(recs) == 2)


def _emit_append_only() -> None:
    """Running emit twice on one batch must not rewrite or duplicate rows."""
    import json
    import tempfile
    from pathlib import Path
    from unittest import mock
    from ballknower_quad.ledger import emit as E
    from ballknower_quad.settings import settings as S

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        day = root / "2026-09-26"
        day.mkdir()
        preds = [_pred(kickoff_ts="2030-09-26T16:00:00+00:00", gid="1"),
                 _pred(kickoff_ts="2030-09-26T19:30:00+00:00", gid="2",
                       home="Georgia", away="Clemson")]
        (day / "predictions.json").write_text(json.dumps(preds))

        with mock.patch.object(S, "content_root", root):
            E.main(["--date", "2026-09-26"])
            fp = day / "ledger" / "forecasts.jsonl"
            first = fp.read_text()
            E.main(["--date", "2026-09-26"])
            second = fp.read_text()

        _assert(first == second)                       # untouched
        _assert(len([l for l in second.splitlines() if l]) == 4)


def _dirty_guard() -> None:
    import subprocess
    import tempfile
    from pathlib import Path
    from ballknower_quad.ledger.emit import code_is_dirty

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        pkg = root / "ballknower_quad"
        pkg.mkdir()
        (pkg / "m.py").write_text("x = 1\n")
        env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
               "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
        for cmd in (["git", "init", "-q"], ["git", "add", "-A"],
                    ["git", "commit", "-qm", "init"]):
            subprocess.run(cmd, cwd=root, env=env, check=True,
                           capture_output=True)
        _assert(code_is_dirty(root) is None)          # clean
        (pkg / "m.py").write_text("x = 2\n")
        _assert(code_is_dirty(root) is not None)      # modified
        # changes OUTSIDE the package (content, logs) must not count
        subprocess.run(["git", "checkout", "--", "ballknower_quad/m.py"],
                       cwd=root, env=env, check=True, capture_output=True)
        (root / "content.txt").write_text("generated\n")
        _assert(code_is_dirty(root) is None)


def _all_parsers_build() -> None:
    """
    Build every command-line parser in the package.

    argparse treats `help=` as a %-format template, so a literal percent sign
    ("95%+ locks") is a latent crash. Python <=3.13 only blew up when the help
    was actually rendered; 3.14 validates at `add_argument` time, which turns a
    dormant typo in a flag nobody uses into an immediate failure of the whole
    command. Cheap to assert, annoying to rediscover at kickoff.
    """
    import importlib

    modules = [
        "ballknower_quad.scripts.train",
        "ballknower_quad.scripts.predict",
        "ballknower_quad.scripts.weekly_pipeline",
        "ballknower_quad.scripts.compare_versions",
        "ballknower_quad.scripts.diagnose",
        "ballknower_quad.scripts.calibrate_tiers",
        "ballknower_quad.scripts.check_api",
        "ballknower_quad.scripts.repro_check",
        "ballknower_quad.backtest.evaluate",
        "ballknower_quad.backtest.leakage_audit",
        "ballknower_quad.ledger.emit",
        "ballknower_quad.ledger.resolve",
        "ballknower_quad.ledger.score",
    ]
    broken = []
    for name in modules:
        try:
            mod = importlib.import_module(name)
            try:
                mod.main(["--help"])
            except SystemExit:
                pass          # --help always exits; that's success
        except ValueError as e:
            broken.append(f"{name}: {e}")
        except Exception as e:  # noqa: BLE001
            broken.append(f"{name}: {type(e).__name__}: {e}")
    if broken:
        raise AssertionError("; ".join(broken))


def _chain_links() -> None:
    a = _base().finalize(None)
    b = _base(line=-3.5).finalize(a.row_hash)
    _assert(b.prev_hash == a.row_hash and a.row_hash != b.row_hash)


def _manifest_ok() -> None:
    a = _base().finalize(None)
    b = _base(record_family="distributional", market_type="spread", prob=None,
              mu=7.5, sigma=16.0, tier="Pass").finalize(a.row_hash)
    m = build_manifest([a, b], "cfb-test")
    _assert(m["record_count"] == 2)
    _assert(m["chain_head"] == b.row_hash)
    _assert(m["family_counts"] == {"distributional": 1, "probabilistic": 1})


if __name__ == "__main__":
    raise SystemExit(run())
