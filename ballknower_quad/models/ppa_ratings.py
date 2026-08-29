"""
ballknower_quad.models.ppa_ratings
==================================

Rolling, **opponent-adjusted**, leakage-free PPA ratings.

Three problems this solves, each one a lesson carried over from a sibling project:

1. **Opponent adjustment.** CFBD's per-game PPA is raw. In college football that
   matters enormously: a 0.45 offensive PPA against Mercer is not a 0.45 against
   Georgia. Unadjusted, the metric mostly measures schedule. We adjust each game
   against the opponent's *rating as of that game* —
       ``adj_off = raw_off − (opp_def_allowed − league_avg_def)``
   using only prior information, so it stays leakage-free. This is the same
   move as the golf model's field adjustment (measure performance against the
   quality actually faced, not in a vacuum).

2. **Small-sample early season.** Week 2 ratings built on one game are wild.
   Ridge shrinkage toward the league mean (``ppa_ridge_games`` phantom games)
   damps that — same device as Links' ``--ridge`` phantom rounds.

3. **Week 1 has no current-season data at all.** This is exactly the hole that
   made Gridiron's V2 QB rating so noisy (Mahomes opening the season at 0.0).
   The fix there — and here — is a **prior-season anchor** with a games-played
   ramp: start from last year's opponent-adjusted season PPA, then blend toward
   current-season form as games accumulate:
       ``w = n_games / (n_games + ppa_anchor_k)``
       ``rating = w * current_rolling + (1 − w) * prior_season``

Defensive orientation note: CFBD defensive PPA is PPA **allowed**, so *lower is
better*. Internally we store it that way (allowed) and flip the sign only when
building the model feature, so ``def_ppa_diff`` reads "home defense better" when
positive.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ballknower_quad.settings import settings

LEAGUE_AVG_OFF = 0.30   # fallback league-average offensive PPA per play
LEAGUE_AVG_DEF = 0.30   # fallback league-average PPA allowed per play


class PPARatings:
    """Maintains per-team rolling opponent-adjusted offensive / defensive PPA."""

    def __init__(self,
                 ridge_games: float = None,
                 anchor_k: float = None,
                 opponent_adjust: bool = True,
                 strength: float = None,
                 clamp: float = None):
        self.ridge_games = (settings.ppa_ridge_games if ridge_games is None
                            else ridge_games)
        self.anchor_k = (settings.ppa_anchor_k if anchor_k is None else anchor_k)
        self.opponent_adjust = opponent_adjust
        self.strength = (settings.ppa_adjust_strength if strength is None
                         else strength)
        self.clamp = settings.ppa_adjust_clamp if clamp is None else clamp
        # Resolved by the caller (FeatureBuilder) from the data when possible;
        # settings value wins if explicitly set.
        self.fcs_offset = (settings.ppa_fcs_offset
                           if settings.ppa_fcs_offset is not None
                           else settings.ppa_fcs_offset_fallback)

        # current-season accumulators
        self._off_sum: Dict[str, float] = defaultdict(float)
        self._def_sum: Dict[str, float] = defaultdict(float)
        self._n: Dict[str, float] = defaultdict(float)
        # prior-season anchors: team -> (off, def_allowed)
        self._anchor: Dict[str, Tuple[float, float]] = {}
        # running league means (updated online, so they're leakage-free too)
        self._lg_off_sum = 0.0
        self._lg_def_sum = 0.0
        self._lg_n = 0.0
        # self-centering reference: the mean of the opponent RATINGS we actually
        # subtract, so value and reference always share a population.
        self._ref_off_sum = 0.0
        self._ref_def_sum = 0.0
        self._ref_n = 0.0

    # ------------------------------------------------------------------ #
    # League means
    # ------------------------------------------------------------------ #
    @property
    def league_off(self) -> float:
        return (self._lg_off_sum / self._lg_n) if self._lg_n >= 20 else LEAGUE_AVG_OFF

    @property
    def league_def(self) -> float:
        return (self._lg_def_sum / self._lg_n) if self._lg_n >= 20 else LEAGUE_AVG_DEF

    @property
    def ref_off(self) -> float:
        """Mean opponent OFFENSIVE rating, seeded with the raw league mean."""
        prior = settings.ppa_ref_prior_n
        return ((self._ref_off_sum + prior * self.league_off)
                / (self._ref_n + prior))

    @property
    def ref_def(self) -> float:
        """Mean opponent DEFENSIVE rating, seeded with the raw league mean."""
        prior = settings.ppa_ref_prior_n
        return ((self._ref_def_sum + prior * self.league_def)
                / (self._ref_n + prior))

    def anchor_weight(self, team: str) -> float:
        """Share of a team's current rating coming from the prior-season anchor.

        1.0 means the rating IS last season's number — which is exactly the
        situation in the preseason and the first couple of weeks. Read any
        leaderboard printed then as a view of CFBD's prior-season aggregate,
        not of this engine's in-season adjustment.
        """
        n = self._n[team]
        return 1.0 - (n / (n + self.anchor_k)) if n > 0 else 1.0

    # ------------------------------------------------------------------ #
    # Anchors
    # ------------------------------------------------------------------ #
    def set_anchors_from_season(self, season_ppa: pd.DataFrame,
                                season: int) -> None:
        """Load prior-season opponent-adjusted PPA as the anchor for `season`."""
        prior = season_ppa[season_ppa["season"] == season - 1]
        self._anchor = {
            r.team: (float(r.off_ppa_season), float(r.def_ppa_season))
            for r in prior.itertuples(index=False)
            if pd.notna(r.off_ppa_season) and pd.notna(r.def_ppa_season)
        }

    def snapshot_as_anchors(self) -> None:
        """Carry OUR OWN end-of-season ratings forward as next year's anchor.

        Regressed toward the league mean by ``ppa_anchor_carry`` for the same
        reason Elo is: rosters turn over, and a rating built on last year's
        players shouldn't arrive at full strength.
        """
        carry = settings.ppa_anchor_carry
        lo, ld = self.league_off, self.league_def
        new = {}
        for team in (set(self._n) | set(self._anchor)):
            off, dfn = self.get(team)
            new[team] = (lo + carry * (off - lo), ld + carry * (dfn - ld))
        self._anchor = new

    def start_new_season(self) -> None:
        """Clear current-season accumulators; anchors are set separately."""
        self._off_sum.clear()
        self._def_sum.clear()
        self._n.clear()
        self._ref_off_sum = 0.0
        self._ref_def_sum = 0.0
        self._ref_n = 0.0

    # ------------------------------------------------------------------ #
    # Read: rating as of now (leakage-free if called before update)
    # ------------------------------------------------------------------ #
    def get(self, team: str) -> Tuple[float, float]:
        """Return (off_ppa, def_ppa_allowed) — ridge-shrunk, anchor-blended."""
        anchor_off, anchor_def = self._anchor.get(
            team, (self.league_off, self.league_def))
        n = self._n[team]

        if n <= 0:
            return anchor_off, anchor_def

        # ridge toward league mean for small samples
        cur_off = ((self._off_sum[team] + self.ridge_games * self.league_off)
                   / (n + self.ridge_games))
        cur_def = ((self._def_sum[team] + self.ridge_games * self.league_def)
                   / (n + self.ridge_games))

        # games-played ramp toward current-season form
        w = n / (n + self.anchor_k)
        return (w * cur_off + (1 - w) * anchor_off,
                w * cur_def + (1 - w) * anchor_def)

    def net(self, team: str) -> float:
        """Net PPA: offense minus defense-allowed. Higher = better team."""
        off, dfn = self.get(team)
        return off - dfn

    # ------------------------------------------------------------------ #
    # Update: fold one team-game in (call AFTER recording features)
    # ------------------------------------------------------------------ #
    def fcs_rating(self) -> Tuple[float, float]:
        """Pooled rating for a non-FBS opponent — the PPA analog of the Elo floor.

        An FCS team is weak on both sides: its offense generates less than a
        league-average offense, and its defense allows more than a league-average
        defense. Rating them explicitly (instead of letting `get()` fall back to
        the league mean) is what stops a 56-0 cupcake win from reading as an
        average-opponent performance.
        """
        d = self.fcs_offset
        return self.league_off - d, self.league_def + d

    def update(self, team: str, opponent: str,
               off_ppa: float, def_ppa: float,
               opponent_is_fcs: bool = False) -> None:
        if pd.isna(off_ppa) or pd.isna(def_ppa):
            return

        adj_off, adj_def = float(off_ppa), float(def_ppa)
        if self.opponent_adjust and self.strength > 0:
            # How good was the opponent, relative to the population of opponent
            # ratings we're actually comparing against?
            #
            # The reference MUST come from the same population as the value
            # being subtracted. Comparing an *adjusted* opponent rating against a
            # *raw* league mean lets the two scales drift apart, and because each
            # adjustment feeds the next, that drift compounds over a season. The
            # tell is adjusted offense and defense centering on different means —
            # impossible in a closed system, since every offensive play is some
            # defense's play.
            opp_off, opp_def = (self.fcs_rating() if opponent_is_fcs
                                else self.get(opponent))
            self._ref_off_sum += opp_off
            self._ref_def_sum += opp_def
            self._ref_n += 1.0

            off_dev = float(np.clip(opp_def - self.ref_def,
                                    -self.clamp, self.clamp))
            def_dev = float(np.clip(opp_off - self.ref_off,
                                    -self.clamp, self.clamp))
            adj_off = off_ppa - self.strength * off_dev
            adj_def = def_ppa - self.strength * def_dev

        self._off_sum[team] += adj_off
        self._def_sum[team] += adj_def
        self._n[team] += 1.0

        # League means must describe FBS-vs-FBS play only. CFBD returns just one
        # side of an FBS-vs-FCS game, so folding those rows in keeps the FBS
        # team's inflated offense and suppressed defense-allowed while dropping
        # the offsetting half — which is exactly what pulled the raw means apart
        # (off 0.175 vs def 0.150) when they're arithmetically required to match.
        if not opponent_is_fcs:
            self._lg_off_sum += float(off_ppa)
            self._lg_def_sum += float(def_ppa)
            self._lg_n += 1.0

    # ------------------------------------------------------------------ #
    # Scale diagnostics
    # ------------------------------------------------------------------ #
    def centering_report(self) -> dict:
        """
        Mean of the adjusted offensive and defensive ratings.

        In a closed system these MUST be equal — team A's offensive PPA in a game
        is exactly team B's defensive PPA allowed, so summed over all team-games
        the two totals are identical. A gap means the adjustment has pulled the
        two scales apart and `net_ppa` carries a schedule-dependent bias.
        """
        teams = set(self._n) | set(self._anchor)
        if not teams:
            return {}
        offs, defs = [], []
        for t in teams:
            o, d = self.get(t)
            offs.append(o)
            defs.append(d)
        return {
            "n_teams": len(teams),
            "raw_league_off": self.league_off,
            "raw_league_def": self.league_def,
            "adj_mean_off": float(np.mean(offs)),
            "adj_mean_def": float(np.mean(defs)),
            "centering_gap": float(np.mean(offs) - np.mean(defs)),
            "fcs_offset": self.fcs_offset,
            "ref_off": self.ref_off,
            "ref_def": self.ref_def,
            "anchor_teams": len(self._anchor),
            "teams_with_current_games": int(sum(1 for v in self._n.values()
                                                if v > 0)),
        }

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "ridge_games": self.ridge_games,
            "anchor_k": self.anchor_k,
            "opponent_adjust": self.opponent_adjust,
            "strength": self.strength, "clamp": self.clamp,
            "fcs_offset": self.fcs_offset,
            "ref": [self._ref_off_sum, self._ref_def_sum, self._ref_n],
            "off_sum": dict(self._off_sum), "def_sum": dict(self._def_sum),
            "n": dict(self._n), "anchor": self._anchor,
            "lg": [self._lg_off_sum, self._lg_def_sum, self._lg_n],
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "PPARatings":
        d = json.loads(Path(path).read_text())
        r = cls(ridge_games=d["ridge_games"], anchor_k=d["anchor_k"],
                opponent_adjust=d.get("opponent_adjust", True),
                strength=d.get("strength"), clamp=d.get("clamp"))
        r._ref_off_sum, r._ref_def_sum, r._ref_n = d.get("ref", [0.0, 0.0, 0.0])
        if d.get("fcs_offset") is not None:
            r.fcs_offset = d["fcs_offset"]
        r._off_sum = defaultdict(float, d["off_sum"])
        r._def_sum = defaultdict(float, d["def_sum"])
        r._n = defaultdict(float, d["n"])
        r._anchor = {k: tuple(v) for k, v in d["anchor"].items()}
        r._lg_off_sum, r._lg_def_sum, r._lg_n = d["lg"]
        return r

    def top(self, n: int = 25) -> pd.DataFrame:
        teams = set(self._n) | set(self._anchor)
        rows = []
        for t in teams:
            off, dfn = self.get(t)
            rows.append({"team": t, "off_ppa": round(off, 4),
                         "def_ppa_allowed": round(dfn, 4),
                         "net_ppa": round(off - dfn, 4),
                         "games": int(self._n[t]),
                         "anchor_w": round(self.anchor_weight(t), 2)})
        return (pd.DataFrame(rows).sort_values("net_ppa", ascending=False)
                .head(n).reset_index(drop=True))


def index_game_ppa(game_ppa: pd.DataFrame) -> Dict[Tuple, Tuple[float, float]]:
    """
    Build a fast lookup: (game_id, team) -> (off_ppa, def_ppa).
    Falls back to (season, week, team) when game_id is missing/misaligned.
    """
    idx: Dict[Tuple, Tuple[float, float]] = {}
    for r in game_ppa.itertuples(index=False):
        vals = (r.off_ppa, r.def_ppa)
        if pd.notna(r.game_id):
            idx[("g", r.game_id, r.team)] = vals
        idx[("swt", r.season, r.week, r.team)] = vals
    return idx


def lookup_ppa(idx: Dict, game_id, season, week, team
               ) -> Optional[Tuple[float, float]]:
    if pd.notna(game_id) and ("g", game_id, team) in idx:
        return idx[("g", game_id, team)]
    return idx.get(("swt", season, week, team))
