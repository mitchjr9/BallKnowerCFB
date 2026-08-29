"""
ballknower_quad.settings
========================

Single source of truth for paths, hyperparameters, the CFBD API config, and the
newsletter confidence tiers. Mirrors the `settings.py` in Gridiron/Pitch/Links so
the loaders, models, backtester, CLI and pipeline all read one config object.

Set your free CollegeFootballData API key once (https://collegefootballdata.com/key)
via a `.env` file or an environment variable:

    # .env  (in the project root, next to this package folder)
    CFBD_API_KEY=your_key_here

Without a key the loader degrades to a deterministic **synthetic** history so the
whole package boots and validates offline — the same fallback convention as the
golf and tennis builds.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# settings.py lives at  <root>/ballknower_quad/settings.py , so the project root
# (where you run `python -m ballknower_quad.scripts.X`) is two parents up.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Optional .env loader (no hard dependency on python-dotenv)
# --------------------------------------------------------------------------- #
def _load_dotenv() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass
class Settings:
    # ----- identity ------------------------------------------------------- #
    app_name: str = "BallKnower Quad"
    sport: str = "college-football"
    disclaimer: str = (
        "For entertainment and educational purposes only — not financial or "
        "betting advice."
    )

    # ----- paths ---------------------------------------------------------- #
    project_root: Path = PROJECT_ROOT
    data_cache_root: Path = PROJECT_ROOT / "data_cache"
    models_store_root: Path = PROJECT_ROOT / "models_store"
    content_root: Path = PROJECT_ROOT / "content" / "cfb"

    # ----- CFBD API (v2) -------------------------------------------------- #
    # The v1 API was retired before the 2025 season. Since May 2025 BOTH
    # api.collegefootballdata.com and apinext.collegefootballdata.com point to
    # v2; api.* is the canonical host and what the current docs use, so we
    # target it here. Every call needs a Bearer token.
    cfbd_base_url: str = "https://api.collegefootballdata.com"
    cfbd_api_key: str = field(
        default_factory=lambda: os.environ.get("CFBD_API_KEY", "")
    )
    # Free tier is 1,000 calls/month — cache aggressively. These TTLs (hours)
    # govern the on-disk cache in cfbd_loader.
    cache_ttl_completed_hours: int = 24 * 30   # finished seasons ~never change
    cache_ttl_current_hours: int = 6           # in-season slate refresh
    http_max_retries: int = 4
    http_backoff_base_s: float = 1.5           # exponential: base * 2**attempt

    # Classification of which division we model. FBS is the core; FCS opponents
    # get a floored pooled rating (see cfb_elo.FCS_FLOOR).
    classification: str = "fbs"

    # ----- Elo hyperparameters ------------------------------------------- #
    elo_base: float = 1500.0
    elo_k: float = 42.0                # higher than NFL: fewer games/season
    elo_hca: float = 65.0             # home-field ≈ 2.5 pts; tunable, neutral-aware
    elo_mov_enabled: bool = True      # 538-style margin-of-victory multiplier
    elo_mov_cap: float = 5.0          # cap the log-margin multiplier (blowout damp)
    elo_season_carry: float = 0.66    # regress toward mean by ~1/3 between seasons
    elo_points_per_400: float = 25.0  # 400 Elo ≈ 25 points (margin scaling)

    # ----- model / blend -------------------------------------------------- #
    active_model_version: str = "v2"
    default_blend_elo: float = 0.30   # blend XGB win prob with Elo baseline
    margin_version: str = "v2"
    random_state: int = 7
    test_size: float = 0.15           # holdout fraction for quick metrics

    # ----- PPA (V1.1) ----------------------------------------------------- #
    # Opponent-adjusted Predicted Points Added — the CFB analog of nflverse EPA,
    # which drove most of the gain in Gridiron's NFL V3.
    ppa_ridge_games: float = 3.0   # phantom games toward league mean (small-sample)
    ppa_anchor_k: float = 4.0      # games-played ramp from prior-season anchor
    ppa_opponent_adjust: bool = True
    # Adjustment strength: 0.0 = raw PPA (no opponent adjustment), 1.0 = full.
    # Exposed because the adjustment is the least-proven part of the PPA block —
    # ablate it before trusting it.
    ppa_adjust_strength: float = 1.0
    # Hard cap on how far one opponent can move a single game's adjusted value.
    # Without this, the adjustment feeds on its own output across a season and
    # can drift the whole scale (symptom: adjusted offense and defense centering
    # on different means, which is impossible in a closed system).
    ppa_adjust_clamp: float = 0.25
    # Blend weight of the raw league mean when seeding the self-centering
    # reference (phantom observations).
    ppa_ref_prior_n: float = 50.0
    # Where the Week-1 anchor comes from:
    #   "cfbd" — CFBD's /ppa/teams season aggregate (their adjustment)
    #   "self" — carry forward OUR OWN end-of-season adjusted ratings
    # Worth testing: at week 10 our in-season ratings produce a cleaner top-15
    # than the CFBD aggregate does, so importing theirs each September may be
    # throwing away a better number. "self" also drops a /ppa/teams call per season.
    ppa_anchor_source: str = "cfbd"
    # How FBS-vs-FCS games feed the PPA ratings:
    #   "naive"  — treat the FCS opponent as an average FBS team (WRONG: hands
    #              out a free efficiency bonus, and CFBD only returns one side
    #              of the game so the league means come out asymmetric)
    #   "skip"   — drop those games entirely (unbiased, but throws away ~13% of
    #              the sample; measurably cost accuracy in testing)
    #   "pooled" — keep the game, but rate the FCS opponent at a pooled floor so
    #              the adjustment discounts it properly. Same idea as the Elo
    #              FCS floor. Keeps the data AND removes the bias.
    ppa_fcs_mode: str = "pooled"
    # How far below league average a pooled non-FBS team sits, in PPA units.
    # None = estimate it from the data (recommended). The estimate is the gap
    # between FBS teams' PPA in cupcake games and in FBS-vs-FBS games, which the
    # offense and defense sides agree on to three decimals. Getting this wrong
    # leaves a residual free bonus for every team that schedules an FCS opponent.
    ppa_fcs_offset: Optional[float] = None
    ppa_fcs_offset_fallback: float = 0.18
    # Regression toward the league mean when carrying our own ratings across a
    # season boundary — the PPA analog of elo_season_carry (rosters turn over).
    ppa_anchor_carry: float = 0.60

    # XGBoost classifier params (win probability)
    xgb_clf_params: dict = field(default_factory=lambda: {
        "n_estimators": 320,
        "max_depth": 4,
        "learning_rate": 0.035,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 6,
        "reg_lambda": 1.4,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": 0,
    })
    # XGBoost regressor params (margin)
    xgb_reg_params: dict = field(default_factory=lambda: {
        "n_estimators": 360,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 8,
        "reg_lambda": 1.6,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "n_jobs": 0,
    })

    # ----- newsletter confidence tiers (favorite win prob → label) -------- #
    # RECALIBRATED against the 2019-2025 walk-forward reliability table.
    # CFB's win-prob distribution is far more right-shifted than the NFL's:
    # with n=6,002, the share of games at or above each favorite-win-prob level
    # was roughly
    #     >=0.90 : 12%      >=0.80 : 33%      >=0.70 : 54%      >=0.60 : 75%
    # For comparison, Gridiron's NFL model put only ~18% of games above 0.70.
    # Keeping Lock at 0.80 would have labelled a THIRD of every Saturday a
    # "Lock", which destroys the word. These thresholds target roughly
    # 12% Lock / 21% Strong / 21% Lean / 46% Pass.
    confidence_tiers: List[Tuple[float, str]] = field(default_factory=lambda: [
        (0.94, "🔒 Lock"),
        (0.90, "💪 Strong"),
        (0.86, "🎯 Lean"),
        (0.00, "🪙 Pass"),
    ])

    # ----- V2: preseason cold-start prior --------------------------------- #
    # How much of a team's Week 1 rating comes from THIS year's roster signals
    # rather than last year's regressed Elo. 0.0 disables V2's Elo seeding
    # entirely (features still present); 1.0 would ignore last season.
    preseason_blend: float = 0.25
    # Spread of the Elo distribution, used to map a composite z-score onto the
    # Elo scale. ~150 matches the observed FBS spread.
    preseason_elo_sd: float = 150.0
    # Component weights in the composite. Normalized internally, so these are
    # relative. Talent leads because the 247 composite already folds in
    # recruiting AND transfers; returning production is the piece it misses.
    preseason_weights: dict = field(default_factory=lambda: {
        "talent": 0.40,
        "recruit_roll": 0.20,
        "returning_ppa_pct": 0.30,
        "portal": 0.10,
    })
    # Trailing years of recruiting classes that make up a roster.
    recruiting_window: int = 4
    # Which portal quantity feeds the composite:
    #   "in"     — summed rating of INCOMING transfers (talent added)
    #   "net"    — incoming minus outgoing
    #   "top_in" — summed rating of the best few incoming transfers
    # "net" is the intuitive choice and the wrong one: elite programs bleed
    # backups to the portal, so raw net counts bodies rather than impact and
    # ends up inversely correlated with program quality. "in" is the default.
    preseason_portal_metric: str = "in"
    preseason_portal_top_n: int = 5

    # ----- data horizon --------------------------------------------------- #
    first_train_season: int = 2014   # CFBD PPA coverage is solid from ~2014
    schedule_horizon_days: int = 9   # weekly pipeline look-ahead (Tue→next Mon)

    # ----- helpers -------------------------------------------------------- #
    def models_dir_for(self, version: str) -> Path:
        d = self.models_store_root / version.lower()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ensure_dirs(self) -> None:
        for d in (self.data_cache_root, self.models_store_root, self.content_root):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def has_cfbd_key(self) -> bool:
        return bool(self.cfbd_api_key)


settings = Settings()
settings.ensure_dirs()
