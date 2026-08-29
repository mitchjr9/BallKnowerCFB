"""
ballknower_quad.data.preseason_loader
=====================================

Loads the four CFBD feeds that describe **this year's roster before it plays a
snap** — the raw material for fixing the model's weakest stretch, Weeks 1-4,
where Elo is nothing but last season regressed toward the mean.

    /talent             composite roster talent (247 composite, team-season)
    /recruiting/teams   recruiting class rank + points (team-season)
    /player/returning   returning production (share of last year's PPA back)
    /player/portal      transfer portal, player-level (in and out)

**Timing / leakage.** Every one of these is known before Week 1 of the season it
describes: classes sign in February, the spring portal window closes in May,
returning production is computed off the *previous* roster, and the talent
composite reflects the roster as constituted for that season. So using season-S
values on season-S games is legitimate. What would NOT be legitimate is using
season S+1's values on season S — ``backtest/leakage_audit`` has a check for
exactly that.

Cost: one call per endpoint per season (~4/season, ~48 for 12 seasons). Cached
like everything else. No API key → deterministic synthetic fallback derived from
the same latent team strengths as the synthetic games, so the offline ablation
is a plumbing test rather than a rigged one.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from ballknower_quad.data.cfbd_loader import _cfbd_get, _synthetic_games
from ballknower_quad.settings import settings

log = logging.getLogger("ballknower_quad.preseason")

PRESEASON_COLS = ["season", "team", "talent", "recruit_points", "recruit_rank",
                  "returning_ppa_pct", "portal_in", "portal_out", "portal_net",
                  "portal_top_in"]


def _num(v, default=np.nan) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Individual endpoints
# --------------------------------------------------------------------------- #
def _load_talent(season: int, ttl: int) -> pd.DataFrame:
    rows = _cfbd_get("/talent", {"year": season}, ttl)
    return pd.DataFrame([{"season": season, "team": r.get("team"),
                          "talent": _num(r.get("talent"))} for r in rows])


def _load_recruiting(season: int, ttl: int) -> pd.DataFrame:
    rows = _cfbd_get("/recruiting/teams", {"year": season}, ttl)
    return pd.DataFrame([{"season": season, "team": r.get("team"),
                          "recruit_points": _num(r.get("points")),
                          "recruit_rank": _num(r.get("rank"))} for r in rows])


def _load_returning(season: int, ttl: int) -> pd.DataFrame:
    rows = _cfbd_get("/player/returning", {"year": season}, ttl)
    recs = []
    for r in rows:
        # headline metric is the share of last season's PPA coming back
        pct = r.get("percentPPA", r.get("percent_ppa"))
        if pct is None:
            pct = r.get("percentPassingPPA")
        recs.append({"season": season, "team": r.get("team"),
                     "returning_ppa_pct": _num(pct)})
    return pd.DataFrame(recs)


def _load_portal(season: int, ttl: int) -> pd.DataFrame:
    """Aggregate player-level portal moves into per-team in/out/net value."""
    rows = _cfbd_get("/player/portal", {"year": season}, ttl)
    if not rows:
        return pd.DataFrame(columns=["season", "team", "portal_in",
                                     "portal_out", "portal_net"])
    inc, out, inc_vals = {}, {}, {}
    for r in rows:
        # Prefer the 247-style composite rating; fall back to stars scaled to a
        # comparable range so one missing field doesn't zero out a transfer.
        val = _num(r.get("rating"))
        if not np.isfinite(val):
            stars = _num(r.get("stars"), 0.0)
            val = 0.0 if stars <= 0 else 0.70 + 0.06 * (stars - 2.0)
        if not np.isfinite(val):
            val = 0.0
        dest = r.get("destination")
        orig = r.get("origin")
        if dest:
            inc[dest] = inc.get(dest, 0.0) + val
            inc_vals.setdefault(dest, []).append(val)
        if orig:
            out[orig] = out.get(orig, 0.0) + val
    teams = set(inc) | set(out)
    top_n = settings.preseason_portal_top_n
    return pd.DataFrame([{
        "season": season, "team": t,
        "portal_in": inc.get(t, 0.0), "portal_out": out.get(t, 0.0),
        "portal_net": inc.get(t, 0.0) - out.get(t, 0.0),
        "portal_top_in": float(sum(sorted(inc_vals.get(t, []),
                                          reverse=True)[:top_n])),
    } for t in teams])


# --------------------------------------------------------------------------- #
# Public
# --------------------------------------------------------------------------- #
def load_preseason(seasons: List[int]) -> pd.DataFrame:
    """One row per (season, team) with every preseason signal we use."""
    if not settings.has_cfbd_key:
        log.info("No CFBD key — synthesizing preseason data for %s", seasons)
        return _synthetic_preseason(seasons)

    current_year = datetime.now(timezone.utc).year
    frames = []
    for season in seasons:
        ttl = (settings.cache_ttl_current_hours if season >= current_year
               else settings.cache_ttl_completed_hours)
        df = _load_talent(season, ttl)
        for loader in (_load_recruiting, _load_returning, _load_portal):
            other = loader(season, ttl)
            if other.empty:
                continue
            df = (other if df.empty
                  else df.merge(other, on=["season", "team"], how="outer"))
        if not df.empty:
            frames.append(df)

    if not frames:
        log.warning("No preseason data returned — falling back to synthetic.")
        return _synthetic_preseason(seasons)

    out = pd.concat(frames, ignore_index=True)
    for c in PRESEASON_COLS:
        if c not in out.columns:
            out[c] = np.nan
    return out[PRESEASON_COLS].dropna(subset=["team"]).reset_index(drop=True)


def rolling_recruiting(preseason: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """
    Trailing `window`-year mean recruiting points per team.

    A single signing class never made a roster — a program is the accumulation of
    the last four. Uses a shifted-inclusive window (the class for season S counts
    toward season S, since it signs in February) and only past seasons otherwise,
    so no future class ever leaks backward.
    """
    if preseason.empty:
        return preseason.assign(recruit_roll=np.nan)
    df = preseason.sort_values(["team", "season"]).copy()
    df["recruit_roll"] = (df.groupby("team")["recruit_points"]
                            .transform(lambda s: s.rolling(window, min_periods=1)
                                       .mean()))
    return df


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #
def _synthetic_preseason(seasons: List[int]) -> pd.DataFrame:
    """Derived from the same latent strengths as the synthetic games, with noise
    — strong programs recruit well and return production, but imperfectly."""
    span = [min(seasons) - 1] + list(seasons)
    games = _synthetic_games(span, with_strengths=True)
    rng = np.random.default_rng(settings.random_state + 202)

    strength = {}
    for r in games.itertuples(index=False):
        strength[(r.season, r.home_team)] = r.synth_home_strength
        strength[(r.season, r.away_team)] = r.synth_away_strength

    recs = []
    for (season, team), s in strength.items():
        if season not in seasons:
            continue
        z = s / 12.0
        recs.append({
            "season": season, "team": team,
            "talent": 700 + 60 * z + rng.normal(0, 18),
            "recruit_points": 180 + 24 * z + rng.normal(0, 9),
            "recruit_rank": max(1, int(70 - 25 * z + rng.normal(0, 8))),
            "returning_ppa_pct": float(np.clip(0.60 + 0.05 * z
                                               + rng.normal(0, 0.14), 0.1, 0.95)),
            "portal_in": max(0.0, 4.0 + 1.4 * z + rng.normal(0, 1.4)),
            "portal_out": max(0.0, 4.0 - 0.4 * z + rng.normal(0, 1.4)),
        })
    df = pd.DataFrame(recs)
    if df.empty:
        return pd.DataFrame(columns=PRESEASON_COLS)
    df["portal_net"] = df["portal_in"] - df["portal_out"]
    df["portal_top_in"] = df["portal_in"] * 0.6
    return df[PRESEASON_COLS].reset_index(drop=True)
