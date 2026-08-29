"""
ballknower_quad.data.cfbd_loader
================================

Thin client for the **CollegeFootballData v2 API**, plus an
on-disk cache and a deterministic **synthetic** fallback so the package boots and
backtests offline (same convention as Links/Insights).

Why the care here: the free tier is **1,000 calls/month** and every request needs
a Bearer token. Note the quota is a **shared CFB + CBB pool** — if you're also
hitting collegebasketballdata.com on the same account (Hardwood), those calls come
out of the same 1,000. So we (a) cache every response to ``data_cache/cfbd/``
with a TTL, (b) retry transient failures with exponential backoff (lifted from
Gridiron's ``team_efficiency_loader``), and (c) pull the *whole-season* endpoints
(one call per season) rather than per-game ones.

Endpoints used in V1
--------------------
* ``/games``        — scores + schedule (the spine: training history AND the
                      upcoming slate, since unplayed games come back with null
                      points, exactly like Pitch's martj42 file).
* ``/calendar``     — week → date mapping, for rest-day features and the slate.

Endpoints wired but reserved for V2 (kept here so the budget math is visible)
* ``/ppa/teams``           — season opponent-adjusted efficiency (EPA analog)
* ``/ratings/sp``          — Bill Connelly SP+ (consume, don't rebuild)
* ``/recruiting/teams``    — recruiting class rankings
* ``/player/returning``    — returning production
* ``/player/portal``       — transfer portal
* ``/talent``              — composite roster talent

Set ``CFBD_API_KEY`` in a ``.env`` (see settings.py). No key → synthetic mode.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ballknower_quad.settings import settings

import logging

log = logging.getLogger("ballknower_quad.cfbd")


# --------------------------------------------------------------------------- #
# Cache helpers
# --------------------------------------------------------------------------- #
def _cache_dir() -> Path:
    d = settings.data_cache_root / "cfbd"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(endpoint: str, params: Dict) -> Path:
    raw = endpoint + "?" + urllib.parse.urlencode(sorted(params.items()))
    h = hashlib.sha1(raw.encode()).hexdigest()[:16]
    safe = endpoint.strip("/").replace("/", "_")
    return _cache_dir() / f"{safe}__{h}.json"


def _is_fresh(path: Path, ttl_hours: int) -> bool:
    if not path.exists():
        return False
    age_h = (time.time() - path.stat().st_mtime) / 3600.0
    return age_h < ttl_hours


# --------------------------------------------------------------------------- #
# Raw GET with retry/backoff
# --------------------------------------------------------------------------- #
def _cfbd_get(endpoint: str, params: Dict, ttl_hours: int) -> list:
    """GET a CFBD endpoint as parsed JSON (list). Cached; [] on hard failure."""
    cache_path = _cache_key(endpoint, params)
    if _is_fresh(cache_path, ttl_hours):
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass

    if not settings.has_cfbd_key:
        # No key: never hit the network. Caller handles synthetic fallback.
        return []

    url = f"{settings.cfbd_base_url}{endpoint}?{urllib.parse.urlencode(params)}"
    headers = {
        "Authorization": f"Bearer {settings.cfbd_api_key}",
        "Accept": "application/json",
    }
    last_err: Optional[Exception] = None
    for attempt in range(settings.http_max_retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            cache_path.write_text(json.dumps(data))
            return data
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (401, 403):
                log.error("CFBD auth failed (%s). Check CFBD_API_KEY.", e.code)
                return []
            if e.code == 429:
                log.warning("CFBD rate-limited (429) — backing off.")
            # 5xx / transient → retry
        except Exception as e:  # noqa: BLE001
            last_err = e
        sleep_s = settings.http_backoff_base_s * (2 ** attempt)
        time.sleep(sleep_s)
    log.warning("CFBD GET failed after retries: %s (%s)", endpoint, last_err)
    # Serve a stale cache if we have one rather than nothing.
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except Exception:
            pass
    return []


def _cfbd_get_uncached(endpoint: str, params: Dict | None = None):
    """
    Single uncached GET returning whatever JSON the endpoint gives (dict or list).
    Used for /info and /info/usage, where a cached answer would be useless — the
    whole point is the *current* quota. Returns None on failure/no key.
    """
    if not settings.has_cfbd_key:
        return None
    qs = urllib.parse.urlencode(params or {})
    url = f"{settings.cfbd_base_url}{endpoint}" + (f"?{qs}" if qs else "")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {settings.cfbd_api_key}",
        "Accept": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return {"_error": f"HTTP {e.code}", "_reason": str(e.reason)}
    except Exception as e:  # noqa: BLE001
        return {"_error": type(e).__name__, "_reason": str(e)}


def fetch_account_info():
    """GET /info — Patreon level, tier, monthly limit, remaining/used calls.

    Returns ``None`` when the request isn't authenticated, which makes this the
    cheapest possible check that a key actually works.
    """
    return _cfbd_get_uncached("/info")


def fetch_usage(days: int = 7, limit: int = 10):
    """GET /info/usage — recent usage across the shared CFB + CBB call pool."""
    return _cfbd_get_uncached("/info/usage", {"days": days, "limit": limit})


# --------------------------------------------------------------------------- #
# Public: games (the spine)
# --------------------------------------------------------------------------- #
_GAME_COLS = [
    "game_id", "season", "week", "season_type", "start_date",
    "neutral_site", "conference_game",
    "home_team", "home_conference", "home_classification", "home_points",
    "away_team", "away_conference", "away_classification", "away_points",
    "completed",
]

# CFBD returns classification as "fbs" / "fcs" / "ii" / "iii". Anything that is
# not FBS gets its conference normalized to the sentinel "FCS" so the Elo engine's
# pooled non-FBS floor fires. This matters: real FCS opponents come back with
# their ACTUAL conference (ASUN, Big Sky, MVFC) or null — never the string "FCS" —
# so a conference-name heuristic alone silently treats North Dakota State as a
# full-fledged FBS program.
NON_FBS_SENTINEL = "FCS"


def _normalize_games(rows: list) -> pd.DataFrame:
    """Map raw CFBD /games JSON into our canonical frame (v2 camelCase keys)."""
    recs = []
    for g in rows:
        # v2 uses camelCase; be defensive about either casing.
        def gv(*keys, default=None):
            for k in keys:
                if k in g and g[k] is not None:
                    return g[k]
            return default

        hp = gv("homePoints", "home_points")
        ap = gv("awayPoints", "away_points")
        h_class = gv("homeClassification", "home_classification")
        a_class = gv("awayClassification", "away_classification")
        h_conf = gv("homeConference", "home_conference")
        a_conf = gv("awayConference", "away_conference")
        if h_class is not None and str(h_class).lower() != "fbs":
            h_conf = NON_FBS_SENTINEL
        if a_class is not None and str(a_class).lower() != "fbs":
            a_conf = NON_FBS_SENTINEL
        recs.append({
            "game_id": gv("id", "game_id"),
            "season": gv("season"),
            "week": gv("week"),
            "season_type": gv("seasonType", "season_type", default="regular"),
            "start_date": gv("startDate", "start_date"),
            "neutral_site": bool(gv("neutralSite", "neutral_site", default=False)),
            "conference_game": bool(gv("conferenceGame", "conference_game",
                                       default=False)),
            "home_team": gv("homeTeam", "home_team"),
            "home_conference": h_conf,
            "home_classification": h_class,
            "home_points": hp,
            "away_team": gv("awayTeam", "away_team"),
            "away_conference": a_conf,
            "away_classification": a_class,
            "away_points": ap,
            "completed": bool(gv("completed", default=(hp is not None
                                                       and ap is not None))),
        })
    df = pd.DataFrame.from_records(recs, columns=_GAME_COLS)
    if not df.empty:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce",
                                          utc=True)
        for c in ("season", "week", "home_points", "away_points"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_games(seasons: List[int], season_type: str = "both") -> pd.DataFrame:
    """
    Load games for the given seasons. Completed seasons cache for ~30 days;
    the current season refreshes every few hours. Falls back to synthetic data
    when no API key is present.
    """
    if not settings.has_cfbd_key:
        log.info("No CFBD key — generating synthetic games for %s", seasons)
        return _synthetic_games(seasons)

    current_year = datetime.now(timezone.utc).year
    frames = []
    for season in seasons:
        ttl = (settings.cache_ttl_current_hours if season >= current_year
               else settings.cache_ttl_completed_hours)
        params = {"year": season, "classification": settings.classification}
        if season_type in ("regular", "postseason"):
            params["seasonType"] = season_type
            rows = _cfbd_get("/games", params, ttl)
        else:  # both
            rows = _cfbd_get("/games", {**params, "seasonType": "regular"}, ttl)
            rows += _cfbd_get("/games", {**params, "seasonType": "postseason"}, ttl)
        frames.append(_normalize_games(rows))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=_GAME_COLS)
    if df.empty:
        log.warning("CFBD returned no games — falling back to synthetic.")
        return _synthetic_games(seasons)
    df = df.dropna(subset=["home_team", "away_team", "season"])
    return df.sort_values("start_date").reset_index(drop=True)


def load_calendar(season: int) -> pd.DataFrame:
    """Week → (firstGameStart, lastGameStart) map for rest-day features."""
    if not settings.has_cfbd_key:
        return pd.DataFrame(columns=["season", "week", "first_game", "last_game"])
    rows = _cfbd_get("/calendar", {"year": season},
                     settings.cache_ttl_completed_hours)
    recs = []
    for r in rows:
        recs.append({
            "season": season,
            "week": r.get("week"),
            "first_game": r.get("firstGameStart") or r.get("startDate"),
            "last_game": r.get("lastGameStart") or r.get("endDate"),
        })
    df = pd.DataFrame(recs)
    if not df.empty:
        for c in ("first_game", "last_game"):
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)
    return df


# --------------------------------------------------------------------------- #
# Synthetic fallback — deterministic, so offline runs are reproducible
# --------------------------------------------------------------------------- #
_SYNTH_TEAMS = [
    # A spread of "blue bloods", solid programs, and have-nots, plus a couple of
    # FCS-flavored cupcakes, to exercise the talent-gap / blowout behavior.
    ("Georgia", 1, "SEC"), ("Alabama", 1, "SEC"), ("Ohio State", 1, "Big Ten"),
    ("Michigan", 2, "Big Ten"), ("Texas", 2, "SEC"), ("Oregon", 2, "Big Ten"),
    ("Penn State", 3, "Big Ten"), ("LSU", 3, "SEC"), ("Notre Dame", 3, "ACC"),
    ("Clemson", 3, "ACC"), ("Tennessee", 4, "SEC"), ("Ole Miss", 4, "SEC"),
    ("Utah", 4, "Big 12"), ("Kansas State", 5, "Big 12"), ("Iowa", 5, "Big Ten"),
    ("Miami", 4, "ACC"), ("Louisville", 5, "ACC"), ("SMU", 5, "ACC"),
    ("Boise State", 5, "MWC"), ("Memphis", 6, "AAC"), ("Tulane", 6, "AAC"),
    ("Toledo", 7, "MAC"), ("Appalachian State", 7, "Sun Belt"),
    ("UMass", 9, "Independent"), ("Kent State", 10, "MAC"),
    ("New Mexico State", 9, "CUSA"), ("Akron", 10, "MAC"),
    ("Sam Houston FCS", 11, "FCS"), ("Mercer FCS", 12, "FCS"),
]


def _synthetic_games(seasons: List[int], with_strengths: bool = False
                     ) -> pd.DataFrame:
    """
    Deterministic fake season history.

    ``with_strengths=True`` additionally returns the hidden ``synth_home_strength`` /
    ``synth_away_strength`` columns — the latent "true" team quality at the time of
    each game. The synthetic PPA generator uses these so that offline PPA is a
    *realistically* informative signal (a lower-noise read on true strength than
    the scoreboard) rather than either pure noise or a giveaway.
    """
    rng = np.random.default_rng(settings.random_state)
    teams = _SYNTH_TEAMS
    # latent "true" strength: tier 1 strongest. Map tier→rating.
    true_strength = {name: (6.0 - tier) * 6.0 + rng.normal(0, 1.5)
                     for name, tier, _ in teams}
    conf = {name: c for name, _, c in teams}
    recs = []
    gid = 1
    for season in seasons:
        # mild season-to-season drift in true strength
        for name, _, _ in teams:
            true_strength[name] += rng.normal(0, 2.0)
        # 13 weeks, round-robin-ish random schedule
        for week in range(1, 14):
            shuffled = list(teams)
            rng.shuffle(shuffled)
            for i in range(0, len(shuffled) - 1, 2):
                home_name = shuffled[i][0]
                away_name = shuffled[i + 1][0]
                neutral = bool(week >= 13 and rng.random() < 0.3)  # bowl-ish
                hca = 0.0 if neutral else 2.6
                exp_margin = (true_strength[home_name]
                              - true_strength[away_name] + hca)
                margin = exp_margin + rng.normal(0, 13.5)  # game-day noise
                base = 24 + rng.normal(0, 7)
                home_pts = max(0, round(base + margin / 2))
                away_pts = max(0, round(base - margin / 2))
                start = pd.Timestamp(f"{season}-08-30", tz="UTC") + pd.Timedelta(
                    weeks=week - 1, days=int(rng.integers(0, 3)))
                recs.append({
                    "game_id": gid, "season": season, "week": week,
                    "season_type": "regular", "start_date": start,
                    "neutral_site": neutral,
                    "conference_game": conf[home_name] == conf[away_name],
                    "home_team": home_name, "home_conference": conf[home_name],
                    "home_classification": ("fcs" if conf[home_name] == "FCS"
                                            else "fbs"),
                    "home_points": home_pts,
                    "away_team": away_name, "away_conference": conf[away_name],
                    "away_classification": ("fcs" if conf[away_name] == "FCS"
                                            else "fbs"),
                    "away_points": away_pts, "completed": True,
                    "synth_home_strength": true_strength[home_name],
                    "synth_away_strength": true_strength[away_name],
                })
                gid += 1
    cols = _GAME_COLS + (["synth_home_strength", "synth_away_strength"]
                         if with_strengths else [])
    df = pd.DataFrame.from_records(recs, columns=cols)
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True)
    return df.sort_values("start_date").reset_index(drop=True)


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    g = load_games([2022, 2023])
    print(g.shape)
    print(g.head().to_string())
