"""
ballknower_quad.data.bundle
===========================

One place that loads "everything the model needs for these seasons," so the
trainer, the CLI, the weekly pipeline and the backtester can't drift apart in
what they feed the feature builder.

For ``v1`` that's just games. For ``v1.1`` it also pulls per-game PPA and the
season-aggregate PPA used as the prior-season anchor — including one extra
season of history *before* the training window, so the first modelled season
still gets a real Week 1 anchor instead of falling back to league average.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from ballknower_quad.data.cfbd_loader import load_games
from ballknower_quad.data.ppa_loader import load_game_ppa, load_season_ppa
from ballknower_quad.data.preseason_loader import load_preseason
from ballknower_quad.models.feature_builder import (has_ppa_features,
                                                    has_preseason_features)
from ballknower_quad.settings import settings


@dataclass
class SeasonBundle:
    games: pd.DataFrame
    game_ppa: Optional[pd.DataFrame] = None
    season_ppa: Optional[pd.DataFrame] = None
    preseason: Optional[pd.DataFrame] = None
    seasons: Optional[List[int]] = None

    @property
    def source(self) -> str:
        return "CFBD" if settings.has_cfbd_key else "SYNTHETIC"


def season_range(seasons_back: int) -> List[int]:
    end = dt.datetime.now().year
    start = max(settings.first_train_season, end - seasons_back)
    return list(range(start, end + 1))


def fbs_teams_by_season(games: pd.DataFrame) -> dict:
    """season -> set of FBS team names, from CFBD's classification fields."""
    out: dict = {}
    for team_col, cls_col in (("home_team", "home_classification"),
                              ("away_team", "away_classification")):
        if cls_col not in games.columns:
            continue
        sub = games[["season", team_col, cls_col]].dropna(subset=["season",
                                                                  team_col])
        for season, team, cls in sub.itertuples(index=False):
            if str(cls).lower() == "fbs":
                out.setdefault(int(season), set()).add(team)
    return out


def filter_to_fbs(preseason: Optional[pd.DataFrame],
                  games: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Restrict the preseason table to FBS teams.

    This matters more than it looks. `/recruiting/teams` and `/player/portal`
    both cover FCS programs, so the raw table runs ~270 teams per season against
    FBS's 134. The composite is a z-score, and standardizing over a mixed
    FBS+FCS population is standardizing over a BIMODAL one: every FBS team lands
    above the combined mean and the SD is inflated by the gap between the two
    groups, so the spread that actually matters — the spread among FBS teams —
    gets compressed toward zero before it ever reaches the Elo blend.

    Teams are resolved per season, since programs move up (James Madison,
    Sam Houston, Jacksonville State). Seasons that predate the games frame (the
    trailing recruiting window) fall back to the union of all known FBS teams.
    """
    if preseason is None or preseason.empty:
        return preseason
    by_season = fbs_teams_by_season(games)
    if not by_season:
        return preseason
    all_fbs = set().union(*by_season.values())

    def keep(row) -> bool:
        season = int(row["season"])
        allowed = by_season.get(season, all_fbs)
        return row["team"] in allowed

    filtered = preseason[preseason.apply(keep, axis=1)]
    return filtered.reset_index(drop=True) if len(filtered) else preseason


def load_bundle(seasons: List[int], version: str = "v1") -> SeasonBundle:
    games = load_games(seasons)
    if not has_ppa_features(version):
        return SeasonBundle(games=games, seasons=seasons)

    game_ppa = load_game_ppa(seasons)
    # +1 season of history so season N's anchor (season N-1) exists.
    anchor_seasons = [min(seasons) - 1] + list(seasons)
    season_ppa = load_season_ppa(anchor_seasons)

    preseason = None
    if has_preseason_features(version):
        # Recruiting is a trailing 4-year window, so reach back far enough that
        # the first modelled season has a full window rather than a stub.
        pre_seasons = list(range(min(seasons) - settings.recruiting_window,
                                 max(seasons) + 1))
        preseason = load_preseason(pre_seasons)
        preseason = filter_to_fbs(preseason, games)

    return SeasonBundle(games=games, game_ppa=game_ppa, season_ppa=season_ppa,
                        preseason=preseason, seasons=seasons)
