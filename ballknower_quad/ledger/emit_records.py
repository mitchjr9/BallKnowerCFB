"""
ballknower_quad.ledger.records
==============================

Quad's **emitter** for the BallKnower forecast ledger. This does not define the
schema — that already exists (SQL DDL, Pydantic models, resolution policy,
invariant suite). This module maps Quad's output onto it and enforces the
invariants locally, so a malformed row fails *here* rather than at the ledger
boundary.

Design rules inherited from the schema, and why each one shows up in this file:

* **One record = one resolvable proposition.** A Quad game emits *two* records,
  not one: a probabilistic moneyline record and a distributional margin record.
  Packing both into one row makes reliability curves impossible to compute
  without unpacking, and the two families are scored by different metrics
  (Brier / log-loss vs. CRPS) that must never be averaged together.

* **Namespaced entity IDs.** ``cfb:team:ohio_state``, ``cfb:event:401752701``.

* **The leakage invariant is a rejected write, not a lint.**
  ``asof_ts <= committed_ts <= event_start_ts``. If Quad is run after kickoff,
  the row is refused rather than quietly recorded.

* **Tier is a pure function**, versioned, never hand-set. Here it also takes
  ``data_depth`` — which for Quad is the honest thing to do, because a Week 1
  rating is mostly preseason prior and a Week 10 rating is mostly played
  football. Thin data caps the tier.

* **Deterministic ``forecast_id``.** Hash of the identifying tuple, so re-running
  the pipeline produces the same IDs instead of duplicates, and any content
  change produces a different one.

* **Nothing derived is stored.** No Brier, no stake, no feature values — those
  are views or config references (``config_hash``, ``code_commit``).
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

SCHEMA_VERSION = "1"
RESOLUTION_POLICY_VERSION = "cfb-v1-draft"
MODEL_ID = "ballknower.quad"

LEAGUE = "cfb"


class LedgerError(ValueError):
    """Raised on any row that must not be written."""


# --------------------------------------------------------------------------- #
# Tier config version — DERIVED, never typed
# --------------------------------------------------------------------------- #
def tier_config_version() -> str:
    """
    Version string computed from the thresholds themselves.

    Hand-maintaining this is a trap: you retune the thresholds, forget to bump
    the string, and now rows scored under two different tier definitions are
    indistinguishable in the ledger — which silently corrupts every tier-level
    reliability curve that spans the change. Deriving it from a hash means the
    version cannot fail to change when the config does.
    """
    from ballknower_quad.settings import settings
    raw = json.dumps([[round(float(th), 6), str(lab)]
                      for th, lab in settings.confidence_tiers], sort_keys=True)
    return "quad-tier-" + hashlib.sha256(raw.encode()).hexdigest()[:8]


# --------------------------------------------------------------------------- #
# Entity identity
# --------------------------------------------------------------------------- #
def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
    return s.strip("_")


def team_id(name: str) -> str:
    return f"{LEAGUE}:team:{slug(name)}"


def event_id(game_id, season, week, home: str, away: str) -> str:
    """Prefer CFBD's game id; fall back to a deterministic composite."""
    if game_id is not None and str(game_id).strip() not in ("", "nan", "None"):
        return f"{LEAGUE}:event:{game_id}"
    return f"{LEAGUE}:event:{season}_w{week}_{slug(away)}_at_{slug(home)}"


# --------------------------------------------------------------------------- #
# Tier — pure function of (prob, data_depth), versioned
# --------------------------------------------------------------------------- #
def data_depth_for(games_played: Optional[int]) -> str:
    """
    How much *played* football backs this rating.

    Quad-specific and load-bearing: in Week 1 a team's rating is essentially the
    preseason prior, and calling that a "Lock" would be advertising confidence
    the model hasn't earned. Depth is a separate dimension from probability, and
    the tier function uses both.
    """
    if games_played is None:
        return "none"
    if games_played >= 5:
        return "rich"
    if games_played >= 2:
        return "thin"
    return "none"


_TIER_ORDER = ["Pass", "Lean", "Strong", "Lock"]
_DEPTH_CAP = {"rich": "Lock", "thin": "Strong", "none": "Lean"}


def compute_tier(prob: float, data_depth: str,
                 thresholds: Optional[List] = None) -> str:
    """
    Deterministic tier. Never set this by hand — that contaminates tier-level
    calibration with discretion, and the reliability curve ends up measuring
    judgement rather than the model.
    """
    from ballknower_quad.settings import settings
    thresholds = thresholds or settings.confidence_tiers

    fav = max(prob, 1.0 - prob)
    raw = "Pass"
    for threshold, label in thresholds:
        if fav >= threshold:
            raw = re.sub(r"[^A-Za-z]", "", label) or "Pass"
            break

    cap = _DEPTH_CAP.get(data_depth, "Lean")
    if _TIER_ORDER.index(raw) > _TIER_ORDER.index(cap):
        return cap
    return raw


def assert_tier(record: "ForecastRecord") -> None:
    expected = compute_tier(record.prob if record.prob is not None else 0.5,
                            record.data_depth)
    if record.tier != expected:
        raise LedgerError(
            f"tier {record.tier!r} was hand-set; {tier_config_version()} "
            f"computes {expected!r} for prob={record.prob} "
            f"depth={record.data_depth}")


# --------------------------------------------------------------------------- #
# Provenance / code identity
# --------------------------------------------------------------------------- #
def code_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "nogit"


def config_hash() -> str:
    """Hash the settings that actually change predictions."""
    from ballknower_quad.settings import settings
    payload = {
        "model_version": settings.active_model_version,
        "blend_elo": settings.default_blend_elo,
        "elo": [settings.elo_base, settings.elo_k, settings.elo_hca,
                settings.elo_season_carry, settings.elo_mov_cap],
        "ppa": [settings.ppa_ridge_games, settings.ppa_anchor_k,
                settings.ppa_adjust_strength, settings.ppa_fcs_mode,
                settings.ppa_anchor_source],
        "preseason": [settings.preseason_blend, settings.preseason_elo_sd,
                      sorted(settings.preseason_weights.items()),
                      settings.preseason_portal_metric],
        "tiers": [list(t) for t in settings.confidence_tiers],
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
@dataclass
class ForecastRecord:
    # identity
    forecast_id: str
    schema_version: str
    record_family: str            # "probabilistic" | "distributional"

    # proposition (self-describing — a stranger must be able to grade it)
    market_type: str              # moneyline | spread
    subject_id: str
    opponent_id: Optional[str]
    event_id: str
    side: str
    line: Optional[float]
    proposition_text: str

    # forecast payload
    prob: Optional[float] = None          # probabilistic family
    mu: Optional[float] = None            # distributional family
    sigma: Optional[float] = None

    # tiering / depth
    tier: str = "Pass"
    tier_config_version: str = field(default_factory=tier_config_version)
    data_depth: str = "none"

    # market comparison (null until a lines feed is wired in)
    market_source: Optional[str] = None
    market_prob_devig: Optional[float] = None
    market_line: Optional[float] = None
    market_asof_ts: Optional[str] = None
    devig_method: Optional[str] = None

    # time
    asof_ts: str = ""
    created_ts: str = ""
    committed_ts: str = ""
    event_start_ts: str = ""

    # model / provenance
    model_id: str = MODEL_ID
    model_version: str = "v2"
    config_hash: str = ""
    code_commit: str = ""
    provenance: str = "live"      # live | backfill | replay
    policy_version: str = RESOLUTION_POLICY_VERSION

    # chain
    prev_hash: Optional[str] = None
    row_hash: str = ""
    supersedes: Optional[str] = None

    # resolution (written later, by the grader — never by this package)
    resolution_status: str = "pending"

    def content_tuple(self) -> tuple:
        return (self.subject_id, self.event_id, self.market_type,
                self.line, self.side, self.model_id, self.model_version,
                self.asof_ts)

    def compute_id(self) -> str:
        raw = "|".join("" if v is None else str(v) for v in self.content_tuple())
        return hashlib.sha256(raw.encode()).hexdigest()[:24]

    def compute_row_hash(self) -> str:
        body = {k: v for k, v in asdict(self).items() if k != "row_hash"}
        raw = json.dumps(body, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Every check here is a rejected write, not a warning."""
        if self.record_family not in ("probabilistic", "distributional"):
            raise LedgerError(f"bad record_family {self.record_family!r}")

        if self.record_family == "probabilistic":
            if self.prob is None or not (0.0 < self.prob < 1.0):
                raise LedgerError(f"prob must be in (0,1), got {self.prob}")
            if self.mu is not None:
                raise LedgerError("probabilistic record must not carry mu")
        else:
            if self.mu is None or self.sigma is None or self.sigma <= 0:
                raise LedgerError("distributional record needs mu and sigma>0")
            if self.prob is not None:
                raise LedgerError("distributional record must not carry prob")

        for f in ("asof_ts", "committed_ts", "event_start_ts"):
            if not getattr(self, f):
                raise LedgerError(f"missing timestamp {f}")

        a = _parse(self.asof_ts)
        c = _parse(self.committed_ts)
        e = _parse(self.event_start_ts)

        # `asof_ts <= event_start_ts` is the LEAKAGE invariant and holds for
        # every row, always: a forecast may never be built from data that only
        # existed after the outcome was determinable.
        if a > e:
            raise LedgerError(
                "leakage invariant violated: asof_ts > event_start_ts "
                f"({self.asof_ts} > {self.event_start_ts}) — the model saw data "
                "from after the event")

        # `committed_ts <= event_start_ts` is the PRE-REGISTRATION invariant and
        # applies only to live rows. A backfilled backtest row is legitimately
        # committed today about a game played in 2019; that's why `provenance`
        # exists, and why public calibration surfaces filter to live only.
        if self.provenance == "live":
            if not (a <= c <= e):
                raise LedgerError(
                    "pre-registration invariant violated: require asof_ts <= "
                    f"committed_ts <= event_start_ts, got {self.asof_ts} / "
                    f"{self.committed_ts} / {self.event_start_ts}. A row "
                    "committed after kickoff is not a forecast — emit it with "
                    "--provenance backfill, which is quarantined from the "
                    "public calibration curve.")
        elif a > c:
            raise LedgerError(
                f"asof_ts ({self.asof_ts}) is after committed_ts "
                f"({self.committed_ts})")

        if self.provenance not in ("live", "backfill", "replay"):
            raise LedgerError(f"bad provenance {self.provenance!r}")

        if self.forecast_id != self.compute_id():
            raise LedgerError("forecast_id does not match its content tuple")

        if self.record_family == "probabilistic":
            assert_tier(self)

    def finalize(self, prev_hash: Optional[str]) -> "ForecastRecord":
        self.forecast_id = self.compute_id()
        self.prev_hash = prev_hash
        self.validate()
        self.row_hash = self.compute_row_hash()
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, default=str)


def _parse(ts: str) -> datetime:
    s = str(ts).replace("Z", "+00:00")
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Proposition text — generated, never hand-typed
# --------------------------------------------------------------------------- #
def moneyline_text(pick: str, opponent: str, neutral: bool, kickoff: str) -> str:
    where = "at a neutral site" if neutral else "against"
    tail = "" if neutral else f" {opponent}"
    return (f"{pick} wins {where}{tail} "
            f"({kickoff[:10]})") if neutral else \
           f"{pick} defeats {opponent} ({kickoff[:10]})"


def spread_text(home: str, away: str, mu: float, kickoff: str) -> str:
    return (f"Final margin, {home} minus {away} "
            f"({kickoff[:10]}); model mean {mu:+.1f}")


# --------------------------------------------------------------------------- #
# Manifest (chain head + count) — the thing that gets git-committed daily
# --------------------------------------------------------------------------- #
def build_manifest(records: List[ForecastRecord], batch_label: str) -> Dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "batch": batch_label,
        "emitter": MODEL_ID,
        "generated_ts": now_iso(),
        "record_count": len(records),
        "chain_head": records[-1].row_hash if records else None,
        "code_commit": code_commit(),
        "config_hash": config_hash(),
        "provenance_counts": _counts([r.provenance for r in records]),
        "family_counts": _counts([r.record_family for r in records]),
        "tier_counts": _counts([r.tier for r in records
                                if r.record_family == "probabilistic"]),
    }


def _counts(vals: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items()))
