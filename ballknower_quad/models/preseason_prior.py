"""
ballknower_quad.models.preseason_prior
======================================

Turns preseason roster signals into a **starting Elo prior** for each season.

The problem this solves is the model's clearest structural weakness. At the start
of a season, ``CFBElo._carry_to_new_season`` regresses every team toward 1500 and
that's all it knows. A team that returned two starters and a team that returned
nineteen arrive at Week 1 looking identical apart from last year's record. For
the first month — a quarter of the schedule, and the part your newsletter covers
while readers are most engaged — the model is predicting last year's team.

Approach (the same shape SP+ uses for its preseason projection: recent recruiting,
returning production, recent history):

1. Build a composite z-score per team from talent, trailing 4-year recruiting,
   returning production and net portal value. Each component is standardized
   **within its own season**, so a scale change at the source (talent numbers
   drifting, portal volume exploding post-2021) can't smuggle in a trend.
2. Map that composite onto the Elo scale: ``prior = 1500 + z * elo_sd``.
3. Blend it with the regressed carryover rather than replacing it:

       ``start = (1 - w) * carryover + w * prior``

   Recruiting is correlated with last year's Elo, so adding them would double-
   count program strength. Blending keeps the scale honest, and ``w`` (default
   0.25) says how much of a fresh start the roster signal justifies.

The blend only fires at season boundaries. Within a season, ordinary Elo updates
take over and the prior's influence decays naturally as real results accumulate
— no hand-tuned decay schedule needed.

**Leakage:** every input is known before Week 1 of the season it applies to
(February signing day, spring portal close, prior-year production). The prior for
season S is built only from rows where ``season == S``; nothing from S+1 is ever
visible. ``backtest/leakage_audit`` asserts this directly.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ballknower_quad.data.preseason_loader import rolling_recruiting
from ballknower_quad.settings import settings

# Components and their default weights in the composite. Exposed in settings so
# they can be ablated rather than trusted.
COMPONENTS = ("talent", "recruit_roll", "returning_ppa_pct", "portal")


def _portal_column() -> str:
    return {"in": "portal_in", "net": "portal_net",
            "top_in": "portal_top_in"}.get(settings.preseason_portal_metric,
                                           "portal_in")


def _zscore(s: pd.Series) -> pd.Series:
    """Standardize within a season; constant/empty input returns zeros."""
    v = pd.to_numeric(s, errors="coerce")
    mu = v.mean()
    sd = v.std(ddof=0)
    if not np.isfinite(sd) or sd < 1e-9:
        return pd.Series(np.zeros(len(v)), index=v.index)
    return ((v - mu) / sd).fillna(0.0)


class PreseasonPrior:
    """Per-season, per-team preseason Elo priors plus the raw components."""

    def __init__(self, preseason: Optional[pd.DataFrame] = None):
        self.weights: Dict[str, float] = dict(settings.preseason_weights)
        self.blend: float = settings.preseason_blend
        self.elo_sd: float = settings.preseason_elo_sd
        # (season, team) -> prior Elo
        self._prior: Dict[Tuple[int, str], float] = {}
        # (season, team) -> dict of raw/standardized components
        self._components: Dict[Tuple[int, str], dict] = {}
        self.table: pd.DataFrame = pd.DataFrame()
        if preseason is not None and len(preseason):
            self.fit(preseason)

    # ------------------------------------------------------------------ #
    def fit(self, preseason: pd.DataFrame) -> "PreseasonPrior":
        df = rolling_recruiting(preseason, window=settings.recruiting_window)

        parts = []
        for season, grp in df.groupby("season"):
            g = grp.copy()
            zs = {}
            for comp in COMPONENTS:
                src = _portal_column() if comp == "portal" else comp
                col = g[src] if src in g.columns else pd.Series(
                    np.nan, index=g.index)
                zs[f"z_{comp}"] = _zscore(col)
            for k, v in zs.items():
                g[k] = v

            total_w = sum(abs(self.weights.get(c, 0.0)) for c in COMPONENTS) or 1.0
            g["composite_z"] = sum(
                self.weights.get(c, 0.0) * g[f"z_{c}"] for c in COMPONENTS
            ) / total_w
            g["prior_elo"] = settings.elo_base + g["composite_z"] * self.elo_sd
            parts.append(g)

        self.table = (pd.concat(parts, ignore_index=True) if parts
                      else pd.DataFrame())
        for r in self.table.itertuples(index=False):
            key = (int(r.season), r.team)
            self._prior[key] = float(r.prior_elo)
            self._components[key] = {
                "returning_ppa_pct": float(getattr(r, "returning_ppa_pct",
                                                   np.nan) or np.nan),
                "portal_net": float(getattr(r, "portal_net", np.nan) or np.nan),
                "composite_z": float(r.composite_z),
                "z_returning": float(getattr(r, "z_returning_ppa_pct", 0.0)),
                "z_portal": float(getattr(r, "z_portal", 0.0)),
            }
        return self

    # ------------------------------------------------------------------ #
    def prior_elo(self, season: int, team: str) -> Optional[float]:
        return self._prior.get((int(season), team))

    def component(self, season: int, team: str, name: str,
                  default: float = 0.0) -> float:
        c = self._components.get((int(season), team))
        if not c:
            return default
        v = c.get(name, default)
        return default if (v is None or not np.isfinite(v)) else float(v)

    def priors_for_season(self, season: int) -> Dict[str, float]:
        return {team: elo for (s, team), elo in self._prior.items()
                if s == int(season)}

    # ------------------------------------------------------------------ #
    def top(self, season: int, n: int = 25) -> pd.DataFrame:
        if self.table.empty:
            return pd.DataFrame()
        t = self.table[self.table["season"] == int(season)].copy()
        cols = [c for c in ("team", "prior_elo", "composite_z", "talent",
                            "recruit_roll", "returning_ppa_pct",
                            _portal_column())
                if c in t.columns]
        return (t.sort_values("prior_elo", ascending=False)[cols]
                .head(n).round(3).reset_index(drop=True))
