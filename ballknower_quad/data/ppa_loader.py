"""
ballknower_quad.data.ppa_loader
===============================

Loads **Predicted Points Added (PPA)** — CFBD's expected-points metric and the
direct analog of nflverse EPA, which was the single biggest source of lift in
Gridiron's NFL V3 (``net_epa_diff`` carried most of the gain).

Two endpoints, two different jobs — and the distinction matters for leakage:

``/ppa/games``  → **per-game, per-team** PPA (one row per team per game).
    This is what we roll forward chronologically inside a season. Because it's
    per-game, we can compute "PPA as of kickoff" with no peeking. It is *not*
    opponent-adjusted by CFBD — we do that ourselves in ``ppa_ratings.py``.
    Schema (v2 camelCase, v1-ish flat also handled):
        gameId, season, week, seasonType, team, conference, opponent,
        offense{overall,passing,rushing,firstDown,secondDown,thirdDown},
        defense{...same...}

``/ppa/teams``  → **season-aggregate, opponent-adjusted** PPA.
    A season aggregate is catastrophic to use *within* its own season (it
    contains the game you're predicting, plus every future game). We use it
    **only for prior seasons**, as the Week 1 anchor — precisely the
    prior-season fallback design from Gridiron's team-efficiency loader, whose
    absence made the NFL QB-rating feature so noisy early in the year.

Cost: 1 call per season per endpoint (~22 calls for 11 seasons). Cached.
No API key → deterministic synthetic PPA derived from the same latent team
strengths as the synthetic games, so the offline ablation is a *plumbing* test
rather than a rigged one.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

import numpy as np
import pandas as pd

from ballknower_quad.data.cfbd_loader import _cfbd_get, _synthetic_games
from ballknower_quad.settings import settings

log = logging.getLogger("ballknower_quad.ppa")

GAME_PPA_COLS = ["game_id", "season", "week", "season_type", "team", "opponent",
                 "off_ppa", "def_ppa"]
SEASON_PPA_COLS = ["season", "team", "off_ppa_season", "def_ppa_season"]


def _dig(d: dict, group: str, *flat_keys, default=np.nan):
    """Pull ``d[group]['overall']`` (nested) or a flat fallback key."""
    g = d.get(group)
    if isinstance(g, dict):
        v = g.get("overall")
        if v is not None:
            return float(v)
    for k in flat_keys:
        if d.get(k) is not None:
            try:
                return float(d[k])
            except (TypeError, ValueError):
                pass
    return default


def _normalize_game_ppa(rows: list) -> pd.DataFrame:
    recs = []
    for r in rows:
        def gv(*keys, default=None):
            for k in keys:
                if k in r and r[k] is not None:
                    return r[k]
            return default

        recs.append({
            "game_id": gv("gameId", "game_id"),
            "season": gv("season"),
            "week": gv("week"),
            "season_type": gv("seasonType", "season_type", default="regular"),
            "team": gv("team"),
            "opponent": gv("opponent"),
            "off_ppa": _dig(r, "offense", "offenseOverall", "off_overall"),
            "def_ppa": _dig(r, "defense", "defenseOverall", "def_overall"),
        })
    df = pd.DataFrame.from_records(recs, columns=GAME_PPA_COLS)
    if not df.empty:
        for c in ("season", "week", "off_ppa", "def_ppa"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_game_ppa(seasons: List[int]) -> pd.DataFrame:
    """Per-team, per-game PPA for the given seasons (long format)."""
    if not settings.has_cfbd_key:
        log.info("No CFBD key — synthesizing game PPA for %s", seasons)
        return _synthetic_game_ppa(seasons)

    current_year = datetime.now(timezone.utc).year
    frames = []
    for season in seasons:
        ttl = (settings.cache_ttl_current_hours if season >= current_year
               else settings.cache_ttl_completed_hours)
        rows = _cfbd_get("/ppa/games", {"year": season}, ttl)
        frames.append(_normalize_game_ppa(rows))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=GAME_PPA_COLS)
    if df.empty:
        log.warning("No PPA returned — falling back to synthetic.")
        return _synthetic_game_ppa(seasons)
    return df.dropna(subset=["team", "season"]).reset_index(drop=True)


def load_season_ppa(seasons: List[int]) -> pd.DataFrame:
    """
    Season-aggregate, opponent-adjusted PPA. Use ONLY as a prior-season anchor —
    never for the season it describes.
    """
    if not settings.has_cfbd_key:
        return _synthetic_season_ppa(seasons)

    frames = []
    for season in seasons:
        rows = _cfbd_get("/ppa/teams", {"year": season},
                         settings.cache_ttl_completed_hours)
        recs = []
        for r in rows:
            recs.append({
                "season": r.get("season", season),
                "team": r.get("team"),
                "off_ppa_season": _dig(r, "offense", "offenseOverall"),
                "def_ppa_season": _dig(r, "defense", "defenseOverall"),
            })
        frames.append(pd.DataFrame.from_records(recs, columns=SEASON_PPA_COLS))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=SEASON_PPA_COLS)
    if df.empty:
        return _synthetic_season_ppa(seasons)
    return df.dropna(subset=["team"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Synthetic fallback
# --------------------------------------------------------------------------- #
# Realism knobs. PPA's advantage over raw scoring margin is that it aggregates
# ~70 plays instead of one final score, so it's a *lower-variance* read on true
# team quality (garbage time and turnover luck wash out). We encode exactly that:
# PPA noise is small relative to its signal, whereas the synthetic scoreboard
# carries sd≈13.5 points of game-day noise.
_PPA_SCALE = 0.011      # latent strength units → PPA units
_PPA_BASE = 0.30        # league-average offensive PPA per play
_PPA_NOISE = 0.085      # per-game PPA noise


def _synthetic_game_ppa(seasons: List[int]) -> pd.DataFrame:
    games = _synthetic_games(seasons, with_strengths=True)
    rng = np.random.default_rng(settings.random_state + 101)
    recs = []
    for r in games.itertuples(index=False):
        hs = getattr(r, "synth_home_strength")
        as_ = getattr(r, "synth_away_strength")
        edge = (hs - as_) / 2.0
        # home team's offense good vs away defense, and vice versa
        h_off = _PPA_BASE + _PPA_SCALE * edge + rng.normal(0, _PPA_NOISE)
        a_off = _PPA_BASE - _PPA_SCALE * edge + rng.normal(0, _PPA_NOISE)
        # defensive PPA = PPA *allowed* (lower is better) = opponent's offense.
        # Mirror CFBD's real behaviour: FBS-vs-FCS games come back with ONLY the
        # FBS side, so the offline fixture exercises the same one-sided path the
        # live feed produces (and the offset estimator can be tested offline).
        home_fcs = str(r.home_conference).upper() == "FCS"
        away_fcs = str(r.away_conference).upper() == "FCS"
        if not home_fcs:
            recs.append({"game_id": r.game_id, "season": r.season, "week": r.week,
                         "season_type": r.season_type, "team": r.home_team,
                         "opponent": r.away_team, "off_ppa": h_off,
                         "def_ppa": a_off})
        if not away_fcs:
            recs.append({"game_id": r.game_id, "season": r.season, "week": r.week,
                         "season_type": r.season_type, "team": r.away_team,
                         "opponent": r.home_team, "off_ppa": a_off,
                         "def_ppa": h_off})
    return pd.DataFrame.from_records(recs, columns=GAME_PPA_COLS)


def _synthetic_season_ppa(seasons: List[int]) -> pd.DataFrame:
    g = _synthetic_game_ppa(seasons)
    if g.empty:
        return pd.DataFrame(columns=SEASON_PPA_COLS)
    agg = (g.groupby(["season", "team"], as_index=False)
             .agg(off_ppa_season=("off_ppa", "mean"),
                  def_ppa_season=("def_ppa", "mean")))
    return agg[SEASON_PPA_COLS]


def estimate_fcs_offset(game_ppa: pd.DataFrame) -> Optional[float]:
    """
    Estimate how far below FBS-average a non-FBS opponent plays, in PPA units.

    CFBD returns only the FBS side of an FBS-vs-FCS game, so those rows are
    exactly the "performance against a cupcake" sample. The gap between them and
    two-sided (FBS-vs-FBS) rows measures the opponent's weakness — and the
    offensive and defensive views give an independent read on the same quantity,
    so we average them and expect close agreement.
    """
    if game_ppa is None or game_ppa.empty:
        return None
    counts = game_ppa.groupby(["season", "game_id"])["team"].transform("size")
    two, one = game_ppa[counts >= 2], game_ppa[counts < 2]
    if len(one) < 50 or len(two) < 200:
        return None
    off_gap = float(one["off_ppa"].mean() - two["off_ppa"].mean())
    def_gap = float(two["def_ppa"].mean() - one["def_ppa"].mean())
    est = (off_gap + def_gap) / 2.0
    if not np.isfinite(est) or est <= 0:
        return None
    return round(est, 4)
