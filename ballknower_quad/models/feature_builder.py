"""
ballknower_quad.models.feature_builder
=======================================

Turns a games frame into a **leakage-free** feature matrix, walking games in
chronological order so every feature is computed from information available *at
kickoff only* — the same discipline that made Gridiron's backtest honest.

V1 feature set (a deliberate parallel to NFL V3.1's Elo + team-efficiency +
rest/schedule core, minus the noisy QB-rating term):

    elo_diff_with_hca   Elo gap including home-field (neutral-aware)
    elo_diff            raw Elo gap
    eff_margin_diff     rolling avg scoring margin, home minus away
    home_eff_margin     home rolling avg scoring margin
    away_eff_margin     away rolling avg scoring margin
    rest_diff           home rest days minus away rest days
    either_short_week   1 if either team is on a short week (<6 days)
    is_neutral          neutral-site flag
    is_conference_game  conference matchup flag
    week_in_season      week number (early-season uncertainty proxy)

Rolling efficiency uses an **expanding mean of point margin** with prior-season
carryover toward zero (``eff_season_carry``) — the offline-friendly, zero-extra-
API-call analog of NFL net-points-diff.

V1.1 adds the **opponent-adjusted PPA** block (see ``ppa_ratings.py``), gated by
version so V1 remains bit-identical for a clean ablation:

    off_ppa_diff        home offensive PPA − away offensive PPA
    def_ppa_diff        home defensive strength − away (sign-flipped: + = home better)
    net_ppa_diff        home net PPA − away net PPA  (the headline term)
    mean_net_ppa        average of the two nets — game-quality context

This mirrors NFL V3's ``off_epa_diff`` / ``def_epa_diff`` / ``net_epa_diff``
trio. The offense/defense split is kept deliberately: Elo is symmetric, PPA
isn't, and teams with an elite defense + bad offense behave differently from the
inverse — that was the stated reason ``def_epa_diff`` earned its seat in NFL V3.

Targets: ``y_win`` (home win, ties→0.5 dropped) and ``y_margin`` (home − away).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ballknower_quad.models.cfb_elo import CFBElo, _is_fcs
from ballknower_quad.data.ppa_loader import estimate_fcs_offset
from ballknower_quad.models.preseason_prior import PreseasonPrior
from ballknower_quad.models.ppa_ratings import (PPARatings, index_game_ppa,
                                                lookup_ppa)
from ballknower_quad.settings import settings

FEATURE_COLUMNS_V1: List[str] = [
    "elo_diff_with_hca",
    "elo_diff",
    "eff_margin_diff",
    "home_eff_margin",
    "away_eff_margin",
    "rest_diff",
    "either_short_week",
    "is_neutral",
    "is_conference_game",
    "week_in_season",
]

PPA_FEATURE_COLUMNS: List[str] = [
    "off_ppa_diff",
    "def_ppa_diff",
    "net_ppa_diff",
    "mean_net_ppa",
]

# V1.2: split the neutral-site flag.
# `is_neutral` alone conflates two structurally different game types: bowls
# (62% of neutral games — opt-outs, month-long layoffs, flat motivation) and
# Week 1-2 kickoff classics (12% — full rosters, peak intensity). With only one
# flag the model learns bowl behaviour from the majority and then applies it to
# season openers, which is how Ohio State ended up a 48.8% underdog to Texas on
# a neutral field despite a 39-point Elo edge.
POSTSEASON_FEATURE_COLUMNS: List[str] = [
    "is_postseason",
]

# V2: preseason "cold start" block. See models/preseason_prior.py.
# Weeks 1-4 are the model's weakest stretch — Elo is just last season regressed
# toward the mean, so a team that lost its roster or landed a transfer QB is
# mispriced until it has played enough games to re-rate itself. These features
# describe THIS year's roster and are all known before kickoff of Week 1.
PRESEASON_FEATURE_COLUMNS: List[str] = [
    "preseason_prior_diff",   # composite roster prior, home - away (Elo units)
    "returning_diff",         # returning production share, home - away
    "portal_net_diff",        # net transfer-portal value, home - away
]

FEATURE_COLUMNS_V11: List[str] = FEATURE_COLUMNS_V1 + PPA_FEATURE_COLUMNS
FEATURE_COLUMNS_V12: List[str] = FEATURE_COLUMNS_V11 + POSTSEASON_FEATURE_COLUMNS
FEATURE_COLUMNS_V2: List[str] = FEATURE_COLUMNS_V11 + PRESEASON_FEATURE_COLUMNS

# Explicit registry. Arithmetic gating on the version string (major/minor
# comparisons) silently broke at v2 — "v2" has minor 0, so a `minor >= 1` test
# dropped the PPA block from the very version meant to build on it. A table
# can't develop that class of bug.
VERSION_FEATURES: Dict[str, List[str]] = {
    "v1":   FEATURE_COLUMNS_V1,
    "v1.1": FEATURE_COLUMNS_V11,
    "v1.2": FEATURE_COLUMNS_V12,
    "v2":   FEATURE_COLUMNS_V2,
    "v2.1": FEATURE_COLUMNS_V2 + POSTSEASON_FEATURE_COLUMNS,
}

# Back-compat alias — anything importing FEATURE_COLUMNS gets the V1 set.
FEATURE_COLUMNS = FEATURE_COLUMNS_V1


def _norm_version(version: str) -> str:
    v = str(version).strip().lower()
    return v if v.startswith("v") else f"v{v}"


def columns_for(version: str) -> List[str]:
    v = _norm_version(version)
    if v not in VERSION_FEATURES:
        raise ValueError(
            f"Unknown model version {version!r}. "
            f"Known: {', '.join(sorted(VERSION_FEATURES))}")
    return list(VERSION_FEATURES[v])


def has_ppa_features(version: str) -> bool:
    return PPA_FEATURE_COLUMNS[0] in columns_for(version)


def has_postseason_feature(version: str) -> bool:
    return POSTSEASON_FEATURE_COLUMNS[0] in columns_for(version)


def has_preseason_features(version: str) -> bool:
    return PRESEASON_FEATURE_COLUMNS[0] in columns_for(version)


EFF_SEASON_CARRY = 0.50   # regress rolling efficiency toward 0 between seasons
EFF_RIDGE_GAMES = 2.0     # phantom games toward 0 (shrinks small-sample early wk)
SHORT_WEEK_DAYS = 6
MAX_REST_DAYS = 21.0      # cap byes / openers so the feature isn't dominated by them


class FeatureBuilder:
    """Builds features + targets and leaves the rating engines warm for inference."""

    def __init__(self, elo: CFBElo | None = None, *, version: str = "v1",
                 game_ppa: Optional[pd.DataFrame] = None,
                 season_ppa: Optional[pd.DataFrame] = None,
                 preseason: Optional[pd.DataFrame] = None):
        self.elo = elo or CFBElo()
        self.version = version
        self.use_ppa = has_ppa_features(version)
        self.use_postseason = has_postseason_feature(version)
        self.use_preseason = has_preseason_features(version)
        self.feature_columns = columns_for(version)

        self.prior: Optional[PreseasonPrior] = None
        if self.use_preseason and preseason is not None and len(preseason):
            self.prior = PreseasonPrior(preseason)
        self._cur_season: Optional[int] = None

        # PPA state (only populated for v1.1+)
        self.ppa: Optional[PPARatings] = None
        self._ppa_idx: Dict = {}
        self._season_ppa: Optional[pd.DataFrame] = None
        if self.use_ppa:
            self.ppa = PPARatings(
                opponent_adjust=settings.ppa_opponent_adjust,
                strength=settings.ppa_adjust_strength)
            if game_ppa is not None and len(game_ppa):
                self._ppa_idx = index_game_ppa(game_ppa)
                if settings.ppa_fcs_offset is None:
                    est = estimate_fcs_offset(game_ppa)
                    if est is not None:
                        self.ppa.fcs_offset = est
            self._season_ppa = season_ppa

        # rolling efficiency state: team -> (sum_margin, n_games)
        self._eff_sum: dict = defaultdict(float)
        self._eff_n: dict = defaultdict(float)
        self._last_game_date: dict = {}
        self._last_season: int | None = None
        self.ppa_games_skipped = 0

    # ------------------------------------------------------------------ #
    def _eff_rating(self, team: str) -> float:
        """Ridge-shrunk expanding mean of point margin (toward 0)."""
        n = self._eff_n[team]
        s = self._eff_sum[team]
        return s / (n + EFF_RIDGE_GAMES)

    def _carry_efficiency(self) -> None:
        for team in list(self._eff_sum):
            # keep the *rate* but shrink the sample so a new season re-learns
            rate = self._eff_sum[team] / max(self._eff_n[team], 1.0)
            self._eff_n[team] = min(self._eff_n[team], 4.0)
            self._eff_sum[team] = rate * EFF_SEASON_CARRY * self._eff_n[team]

    def _rest_days(self, team: str, game_date) -> float:
        prev = self._last_game_date.get(team)
        if prev is None or pd.isna(prev) or pd.isna(game_date):
            return 7.0
        d = (game_date - prev).total_seconds() / 86400.0
        return float(np.clip(d, 0.0, MAX_REST_DAYS))

    # ------------------------------------------------------------------ #
    def build(self, games: pd.DataFrame, fit_elo: bool = True
              ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Walk games chronologically, emit one feature row per game.

        Returns (X, y_win, y_margin). Rows are emitted for every game (so the
        same routine serves training and live inference); callers filter to
        completed games for fitting. Games are processed (Elo + efficiency
        updated) *after* their feature row is recorded — leakage free.
        """
        g = games.sort_values(["season", "start_date", "week"]).reset_index(
            drop=True).copy()

        rows, y_win, y_margin = [], [], []
        for r in g.itertuples(index=False):
            new_season = (self._last_season is not None
                          and r.season != self._last_season)
            if new_season:
                # Only regress Elo across seasons when we're fitting it. A pre-
                # loaded warm Elo already has carryover baked in; re-walking games
                # to rebuild *efficiency* must not touch it.
                if fit_elo:
                    priors = (self.prior.priors_for_season(int(r.season))
                              if self.prior is not None else None)
                    self.elo._carry_to_new_season(
                        priors=priors,
                        blend=settings.preseason_blend if priors else 0.0)
                self._carry_efficiency()
                if self.use_ppa:
                    # New season: current-season PPA accumulators reset, and a
                    # prior-season anchor takes over for Week 1. This is the fix
                    # for the exact hole that made Gridiron's V2 QB rating so
                    # noisy in September.
                    if settings.ppa_anchor_source == "self":
                        self.ppa.snapshot_as_anchors()   # before clearing state
                        self.ppa.start_new_season()
                    else:
                        self.ppa.start_new_season()
                        if self._season_ppa is not None:
                            self.ppa.set_anchors_from_season(
                                self._season_ppa, int(r.season))
            self._last_season = r.season
            self._cur_season = int(r.season) if pd.notna(r.season) else None
            self.elo.last_season = r.season

            neutral = bool(r.neutral_site)
            elo_diff = self.elo.rating_diff(
                r.home_team, r.away_team, neutral=neutral,
                home_conf=r.home_conference, away_conf=r.away_conference)
            elo_diff_raw = self.elo.rating_diff(
                r.home_team, r.away_team, neutral=True,
                home_conf=r.home_conference, away_conf=r.away_conference)

            home_eff = self._eff_rating(r.home_team)
            away_eff = self._eff_rating(r.away_team)

            home_rest = self._rest_days(r.home_team, r.start_date)
            away_rest = self._rest_days(r.away_team, r.start_date)
            short = float(min(home_rest, away_rest) < SHORT_WEEK_DAYS)

            row = {
                "elo_diff_with_hca": elo_diff,
                "elo_diff": elo_diff_raw,
                "eff_margin_diff": home_eff - away_eff,
                "home_eff_margin": home_eff,
                "away_eff_margin": away_eff,
                "rest_diff": home_rest - away_rest,
                "either_short_week": short,
                "is_neutral": float(neutral),
                "is_conference_game": float(bool(r.conference_game)),
                "week_in_season": float(r.week) if pd.notna(r.week) else 0.0,
            }
            if self.use_ppa:
                row.update(self._ppa_features(r.home_team, r.away_team))
            if self.use_postseason:
                st = str(getattr(r, "season_type", "regular")).lower()
                row["is_postseason"] = float(st != "regular")
            if self.use_preseason:
                row.update(self._preseason_features(
                    r.home_team, r.away_team,
                    int(r.season) if pd.notna(r.season) else None))
            rows.append(row)

            completed = (bool(getattr(r, "completed", False))
                         and pd.notna(r.home_points) and pd.notna(r.away_points))
            if completed:
                margin = float(r.home_points - r.away_points)
                y_win.append(1 if margin > 0 else (0 if margin < 0 else np.nan))
                y_margin.append(margin)
                # update state AFTER recording features
                if fit_elo:
                    self.elo.update_game(
                        r.home_team, r.away_team, r.home_points, r.away_points,
                        neutral=neutral,
                        home_conf=r.home_conference, away_conf=r.away_conference)
                self._eff_sum[r.home_team] += margin
                self._eff_n[r.home_team] += 1
                self._eff_sum[r.away_team] += -margin
                self._eff_n[r.away_team] += 1
                self._last_game_date[r.home_team] = r.start_date
                self._last_game_date[r.away_team] = r.start_date
                if self.use_ppa:
                    self._update_ppa_from_game(r)
            else:
                y_win.append(np.nan)
                y_margin.append(np.nan)

        X = pd.DataFrame(rows, columns=self.feature_columns)
        return X, pd.Series(y_win, name="y_win"), pd.Series(y_margin, name="y_margin")

    # ------------------------------------------------------------------ #
    # PPA helpers (v1.1+)
    # ------------------------------------------------------------------ #
    def _ppa_features(self, home: str, away: str) -> dict:
        """PPA feature block, read BEFORE this game is folded in — leakage-free."""
        h_off, h_def = self.ppa.get(home)
        a_off, a_def = self.ppa.get(away)
        h_net, a_net = h_off - h_def, a_off - a_def
        return {
            "off_ppa_diff": h_off - a_off,
            # def_ppa is PPA *allowed* (lower = better), so flip the sign to make
            # "positive = home defense is better" and keep every diff same-signed.
            "def_ppa_diff": a_def - h_def,
            "net_ppa_diff": h_net - a_net,
            "mean_net_ppa": (h_net + a_net) / 2.0,
        }

    # ------------------------------------------------------------------ #
    # Preseason helpers (v2+)
    # ------------------------------------------------------------------ #
    def _preseason_features(self, home: str, away: str,
                            season: Optional[int]) -> dict:
        """
        Roster-prior features for one game.

        Note these are *season-level constants* — identical for every game a team
        plays that year. They earn their keep in Weeks 1-4, when Elo hasn't
        re-rated anyone yet; the model gets `week_in_season` alongside them and
        can learn to stop leaning on them once real results exist.
        """
        if self.prior is None or season is None:
            return {"preseason_prior_diff": 0.0, "returning_diff": 0.0,
                    "portal_net_diff": 0.0}
        base = settings.elo_base
        hp = self.prior.prior_elo(season, home)
        ap = self.prior.prior_elo(season, away)
        return {
            "preseason_prior_diff": float((hp if hp is not None else base)
                                          - (ap if ap is not None else base)),
            "returning_diff": (self.prior.component(season, home, "z_returning")
                               - self.prior.component(season, away, "z_returning")),
            "portal_net_diff": (self.prior.component(season, home, "z_portal")
                                - self.prior.component(season, away, "z_portal")),
        }

    def _update_ppa_from_game(self, r) -> None:
        """Fold a completed game's PPA into the ratings.

        FBS-vs-FCS games are SKIPPED, for two reasons that showed up in the data:

        1. CFBD's /ppa/games returns only the FBS side of those matchups (one row
           instead of two). Accumulating half a game keeps the FBS team's
           inflated offense and suppressed defense-allowed while discarding the
           offsetting half — which is why the raw league means came out
           asymmetric (off 0.175 vs def 0.150) when they are arithmetically
           required to be equal.
        2. The FCS opponent has no rating, so `get()` falls back to the league
           average — meaning the opponent adjustment silently treats a cupcake as
           an AVERAGE FBS team, handing out a free efficiency bonus for a game
           that carries almost no information.

        Elo still processes these games (that's what the FCS floor is for); it's
        only the efficiency ratings that ignore them.
        """
        home_fcs = _is_fcs(r.home_team, r.home_conference)
        away_fcs = _is_fcs(r.away_team, r.away_conference)
        is_fcs_game = home_fcs or away_fcs
        mode = settings.ppa_fcs_mode

        if is_fcs_game and mode == "skip":
            self.ppa_games_skipped += 1
            return

        gid = getattr(r, "game_id", np.nan)
        h = lookup_ppa(self._ppa_idx, gid, r.season, r.week, r.home_team)
        a = lookup_ppa(self._ppa_idx, gid, r.season, r.week, r.away_team)

        if is_fcs_game and mode == "pooled":
            # Keep the game, but rate the FCS side at a pooled floor so the
            # adjustment discounts it. Only the FBS team gets a rating update —
            # we never build individual FCS ratings, same as Elo. The update is
            # flagged so the league means stay FBS-vs-FBS only.
            if h is not None and not home_fcs:
                self.ppa.update(r.home_team, r.away_team, h[0], h[1],
                                opponent_is_fcs=True)
            if a is not None and not away_fcs:
                self.ppa.update(r.away_team, r.home_team, a[0], a[1],
                                opponent_is_fcs=True)
            return

        # FBS vs FBS (every mode): require BOTH sides. A one-sided row would
        # feed half a game into the league means and re-open the very asymmetry
        # the FCS handling exists to close.
        if h is None or a is None:
            self.ppa_games_skipped += 1
            return
        self.ppa.update(r.home_team, r.away_team, h[0], h[1])
        self.ppa.update(r.away_team, r.home_team, a[0], a[1])

    # ------------------------------------------------------------------ #
    def features_for_matchup(self, home: str, away: str, *, neutral: bool = False,
                             home_conf=None, away_conf=None,
                             conference_game: bool = False, week: int = 8,
                             home_rest: float = 7.0, away_rest: float = 7.0,
                             postseason: bool = False,
                             season: Optional[int] = None
                             ) -> pd.DataFrame:
        """One feature row for a hypothetical matchup, using current warm state."""
        elo_diff = self.elo.rating_diff(home, away, neutral=neutral,
                                        home_conf=home_conf, away_conf=away_conf)
        elo_diff_raw = self.elo.rating_diff(home, away, neutral=True,
                                            home_conf=home_conf, away_conf=away_conf)
        home_eff = self._eff_rating(home)
        away_eff = self._eff_rating(away)
        row = {
            "elo_diff_with_hca": elo_diff,
            "elo_diff": elo_diff_raw,
            "eff_margin_diff": home_eff - away_eff,
            "home_eff_margin": home_eff,
            "away_eff_margin": away_eff,
            "rest_diff": float(np.clip(home_rest, 0, MAX_REST_DAYS)
                               - np.clip(away_rest, 0, MAX_REST_DAYS)),
            "either_short_week": float(min(home_rest, away_rest) < SHORT_WEEK_DAYS),
            "is_neutral": float(neutral),
            "is_conference_game": float(conference_game),
            "week_in_season": float(week),
        }
        if self.use_ppa:
            row.update(self._ppa_features(home, away))
        if self.use_postseason:
            row["is_postseason"] = float(postseason)
        if self.use_preseason:
            row.update(self._preseason_features(home, away,
                                                season or self._cur_season))
        return pd.DataFrame([row], columns=self.feature_columns)
