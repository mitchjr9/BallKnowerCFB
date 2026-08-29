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
                  "pre-registration")
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
