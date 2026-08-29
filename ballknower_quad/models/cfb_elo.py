"""
ballknower_quad.models.cfb_elo
==============================

A transparent Elo engine for college football. Same spirit as Gridiron's NFL Elo
and Pitch's ``SoccerElo``, with four CFB-specific design choices:

1. **Margin-of-victory multiplier (538-style).** Blowouts are common in CFB (a
   top-10 team vs. a cupcake), so a raw win/loss Elo update would massively
   over-credit beating bad teams. The log-margin multiplier with an autocorrelation
   correction damps this: ``ln(|margin|+1) * 2.2 / (0.001*elo_diff_winner + 2.2)``,
   capped by ``elo_mov_cap``.

2. **Neutral-site aware.** Bowls and neutral-site openers carry no home-field
   bump (mirrors Pitch).

3. **FCS / non-FBS floor.** Teams not in our classification (FCS opponents) are
   pooled at a single floored rating so an FBS team beating an FCS side barely
   moves, and *losing* to one is correctly punished.

4. **Prior-season carryover.** With only ~13 games/season, ratings are regressed
   toward the mean between seasons (``elo_season_carry``), the CFB analog of the
   prior-season anchoring we added to Gridiron's rolling features. New/unseen
   teams enter at ``elo_base``.

The engine also supports a **leakage-free walk**: ``process_games`` records each
team's *pre-game* rating, so the feature builder can use "rating as of kickoff"
without peeking at the result.

DISCLAIMER: entertainment/education only — not betting advice.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from ballknower_quad.settings import settings

FCS_FLOOR = 1200.0   # pooled rating for non-FBS opponents
NEW_TEAM = settings.elo_base


def _is_fcs(team: str, conference: Optional[str]) -> bool:
    """
    True for non-FBS opponents, which get pooled at FCS_FLOOR.

    ``conference`` here is the *normalized* conference from the loader: any team
    whose CFBD classification isn't "fbs" arrives with the sentinel "FCS". We
    also accept a raw classification string ("fcs", "ii", "iii") in case this is
    called with unnormalized data, and keep the name-suffix check for the
    synthetic fixtures.

    Do NOT try to infer this from real conference names — FCS teams come back
    with genuine conference labels (Big Sky, MVFC, ASUN) or null.
    """
    if conference is not None:
        c = str(conference).strip().upper()
        if c in {"FCS", "II", "III", "NONE", "NAN"}:
            return True
    return isinstance(team, str) and team.upper().endswith("FCS")


class CFBElo:
    def __init__(
        self,
        base: float = settings.elo_base,
        k: float = settings.elo_k,
        hca: float = settings.elo_hca,
        mov_enabled: bool = settings.elo_mov_enabled,
        mov_cap: float = settings.elo_mov_cap,
    ):
        self.base = base
        self.k = k
        self.hca = hca
        self.mov_enabled = mov_enabled
        self.mov_cap = mov_cap
        self.ratings: Dict[str, float] = {}
        self.last_season: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Rating access
    # ------------------------------------------------------------------ #
    def get(self, team: str, conference: Optional[str] = None) -> float:
        if _is_fcs(team, conference):
            return FCS_FLOOR
        return self.ratings.get(team, self.base)

    def _set(self, team: str, value: float, conference: Optional[str]) -> None:
        if _is_fcs(team, conference):
            return  # don't persist individual FCS ratings; they stay pooled
        self.ratings[team] = value

    # ------------------------------------------------------------------ #
    # Win expectancy & MOV
    # ------------------------------------------------------------------ #
    @staticmethod
    def expected(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def rating_diff(self, home: str, away: str, neutral: bool = False,
                    home_conf=None, away_conf=None) -> float:
        """Home Elo (+HCA unless neutral) minus away Elo."""
        h = self.get(home, home_conf) + (0.0 if neutral else self.hca)
        a = self.get(away, away_conf)
        return h - a

    def win_prob(self, home: str, away: str, neutral: bool = False,
                 home_conf=None, away_conf=None) -> float:
        diff = self.rating_diff(home, away, neutral, home_conf, away_conf)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def _mov_multiplier(self, margin: float, elo_diff_winner: float) -> float:
        if not self.mov_enabled:
            return 1.0
        m = math.log(abs(margin) + 1.0) * (2.2 / (0.001 * elo_diff_winner + 2.2))
        return min(m, self.mov_cap)

    # ------------------------------------------------------------------ #
    # Season carryover
    # ------------------------------------------------------------------ #
    def _carry_to_new_season(self, priors: Optional[Dict[str, float]] = None,
                             blend: float = 0.0) -> None:
        """
        Roll ratings into a new season.

        Baseline behaviour regresses every team toward the mean by
        ``elo_season_carry``. When a preseason ``priors`` map is supplied (V2),
        the regressed carryover is then BLENDED toward each team's roster-based
        prior:

            start = (1 - blend) * carryover + blend * prior

        Blended rather than added, because recruiting and last season's Elo both
        measure program strength — summing them would double-count it. Teams with
        no prior (new FBS members, data gaps) keep the plain carryover.
        """
        carry = settings.elo_season_carry
        for team in list(self.ratings):
            rolled = self.base + carry * (self.ratings[team] - self.base)
            if priors and blend > 0:
                prior = priors.get(team)
                if prior is not None:
                    rolled = (1.0 - blend) * rolled + blend * float(prior)
            self.ratings[team] = rolled

        # A team with a prior but no rating yet (first FBS season) starts there
        # instead of at a flat 1500.
        if priors and blend > 0:
            for team, prior in priors.items():
                if team not in self.ratings:
                    self.ratings[team] = (self.base
                                          + blend * (float(prior) - self.base))

    # ------------------------------------------------------------------ #
    # Single-game update
    # ------------------------------------------------------------------ #
    def update_game(
        self,
        home: str, away: str,
        home_points: float, away_points: float,
        neutral: bool = False,
        home_conf=None, away_conf=None,
    ) -> None:
        rh = self.get(home, home_conf)
        ra = self.get(away, away_conf)
        eh = rh + (0.0 if neutral else self.hca)
        exp_home = self.expected(eh, ra)

        margin = home_points - away_points
        if margin > 0:
            s_home, winner_diff = 1.0, eh - ra
        elif margin < 0:
            s_home, winner_diff = 0.0, ra - eh
        else:
            s_home, winner_diff = 0.5, 0.0

        mult = self._mov_multiplier(margin, winner_diff)
        delta = self.k * mult * (s_home - exp_home)
        self._set(home, rh + delta, home_conf)
        self._set(away, ra - delta, away_conf)

    # ------------------------------------------------------------------ #
    # Leakage-free walk over a games frame
    # ------------------------------------------------------------------ #
    def process_games(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Walk completed games in chronological order, applying season carryover at
        each new season. Returns the frame with pre-game ratings attached:
        ``home_elo_pre``, ``away_elo_pre`` (as of kickoff — leakage free).
        Unplayed rows (null points) are scored with current ratings but do NOT
        update the engine.
        """
        g = games.sort_values(["season", "start_date", "week"]).reset_index(
            drop=True).copy()
        home_pre, away_pre = [], []
        for row in g.itertuples(index=False):
            if self.last_season is not None and row.season != self.last_season:
                self._carry_to_new_season()
            self.last_season = row.season

            home_pre.append(self.get(row.home_team, row.home_conference))
            away_pre.append(self.get(row.away_team, row.away_conference))

            if bool(getattr(row, "completed", False)) and pd.notna(
                    row.home_points) and pd.notna(row.away_points):
                self.update_game(
                    row.home_team, row.away_team,
                    row.home_points, row.away_points,
                    neutral=bool(row.neutral_site),
                    home_conf=row.home_conference, away_conf=row.away_conference,
                )
        g["home_elo_pre"] = home_pre
        g["away_elo_pre"] = away_pre
        return g

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "base": self.base, "k": self.k, "hca": self.hca,
            "mov_enabled": self.mov_enabled, "mov_cap": self.mov_cap,
            "last_season": self.last_season, "ratings": self.ratings,
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "CFBElo":
        d = json.loads(Path(path).read_text())
        e = cls(base=d["base"], k=d["k"], hca=d["hca"],
                mov_enabled=d["mov_enabled"], mov_cap=d["mov_cap"])
        e.ratings = {k: float(v) for k, v in d["ratings"].items()}
        e.last_season = d.get("last_season")
        return e

    def top(self, n: int = 25) -> pd.DataFrame:
        s = pd.Series(self.ratings).sort_values(ascending=False).head(n)
        return s.rename("elo").reset_index().rename(columns={"index": "team"})
