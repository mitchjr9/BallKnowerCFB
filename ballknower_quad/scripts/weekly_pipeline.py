"""
ballknower_quad.scripts.weekly_pipeline
=======================================

Pull the upcoming slate, predict every game with V1 (+ margin if trained), and
write a newsletter to ``content/cfb/<date>/`` as markdown, HTML and JSON — the
CFB analog of Gridiron's ``weekly_football_pipeline``.

    python -m ballknower_quad.scripts.weekly_pipeline
    python -m ballknower_quad.scripts.weekly_pipeline --horizon-days 9

Offline (no CFBD key) it predicts the next synthetic "week" so the pipeline is
fully demonstrable.

DISCLAIMER: entertainment/education only — not betting advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from dataclasses import asdict
from typing import List, Optional

import pandas as pd

from ballknower_quad.data.bundle import load_bundle, season_range
from ballknower_quad.models.cfb_elo import CFBElo
from ballknower_quad.models.cfb_margin_v1 import CFBMarginV1
from ballknower_quad.models.cfb_model_v1 import CFBModelV1
from ballknower_quad.models.feature_builder import FeatureBuilder
from ballknower_quad.content_utils import (GamePrediction, build_newsletter,
                                           markdown_to_html, render_blog)
from ballknower_quad.settings import settings


def _upcoming(games: pd.DataFrame, horizon_days: int,
              include_started: bool = False,
              week: Optional[int] = None) -> pd.DataFrame:
    """
    The next single week of games — not a rolling date window.

    A date window is the wrong unit here. `schedule_horizon_days` is measured
    from *now*, so running on a Saturday sweeps in the following weekend too:
    104 games across two weeks, with 72 teams appearing twice and Ohio State
    down to play both Kent State and Texas. CFBD stamps a `week` on every game,
    so anchor on that instead — find the week containing the next kickoff and
    take exactly that week. The horizon then only acts as a safety bound.
    """
    now = pd.Timestamp.now(tz="UTC")
    future = games[(~games["completed"]) | (games["home_points"].isna())]
    if future.empty:
        # synthetic/offline: treat the last available week as the slate
        last_season = games["season"].max()
        last_week = games[games["season"] == last_season]["week"].max()
        return games[(games["season"] == last_season)
                     & (games["week"] == last_week)].copy()

    lower = now - pd.Timedelta(days=1) if include_started else now
    fut = future[future["start_date"] >= lower]
    if fut.empty:
        fut = future

    if week is not None:
        target_season = int(fut["season"].min())
        target_week = int(week)
    else:
        nxt = fut.sort_values("start_date").iloc[0]
        target_season, target_week = int(nxt["season"]), int(nxt["week"])

    sl = fut[(fut["season"] == target_season) & (fut["week"] == target_week)]
    sl = sl[sl["start_date"] <= now + pd.Timedelta(days=horizon_days)]
    if sl.empty:
        sl = fut.head(20)
    return sl.copy()


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate the weekly CFB newsletter.")
    p.add_argument("--horizon-days", type=int, default=settings.schedule_horizon_days)
    p.add_argument("--seasons-back", type=int, default=11)
    p.add_argument("--version", default=None,
                   help="model version to use (default: active_model_version)")
    p.add_argument("--week", type=int, default=None,
                   help="force a specific week number instead of the week "
                        "containing the next kickoff")
    p.add_argument("--include-started", action="store_true",
                   help="also include games that have already kicked off "
                        "(they cannot be pre-registered in the ledger)")
    p.add_argument("--include-fcs", action="store_true",
                   help="keep FBS-vs-FCS games in the newsletter (off by "
                        "default — they're 95%%+ locks nobody reads)")
    args = p.parse_args(argv)

    version = (args.version or settings.active_model_version).lower()
    try:
        model = CFBModelV1.load(settings.models_dir_for(version))
    except Exception:
        print(f"No trained {version} model. Run: "
              f"python -m ballknower_quad.scripts.train --version {version}")
        return 1
    try:
        margin_model = CFBMarginV1.load(settings.models_dir_for(version))
    except Exception:
        margin_model = None

    seasons = season_range(args.seasons_back)
    bundle = load_bundle(seasons, version=version)
    games = bundle.games

    # warm a builder on completed games only (leakage free), then score the slate
    fb = FeatureBuilder(version=version, game_ppa=bundle.game_ppa,
                        season_ppa=bundle.season_ppa,
                        preseason=bundle.preseason)
    elo_path = settings.models_store_root / "elo_state.json"
    if elo_path.exists():
        fb.elo = CFBElo.load(elo_path)
    completed = games[games["completed"] & games["home_points"].notna()]
    fb.build(completed, fit_elo=not elo_path.exists())

    # Warming on completed games alone stops at the last finished game — in
    # August that's last season's title game. Without this the engines still
    # hold last season's in-season state, which both skews the PPA features away
    # from what the model was trained on and makes the ledger read Week 1 data
    # depth as "rich". Roll forward exactly the way the training walk does.
    slate_season = None
    upcoming_seasons = games.loc[~games["completed"], "season"].dropna()
    if len(upcoming_seasons):
        slate_season = int(upcoming_seasons.min())
    if slate_season is not None and fb._last_season != slate_season:
        fb.roll_to_season(slate_season, roll_elo=not elo_path.exists())
        print(f"  rolled rating state into {slate_season} "
              f"(no completed {slate_season} games yet)")

    slate = _upcoming(games, args.horizon_days,
                      include_started=args.include_started, week=args.week)
    if len(slate):
        wks = sorted(set(int(w) for w in slate["week"].dropna()))
        print(f"  target: {int(slate['season'].min())} week "
              f"{wks[0] if len(wks) == 1 else wks}")
    n_all = len(slate)
    if not args.include_fcs and "away_classification" in slate.columns:
        is_fcs_game = ((slate["home_classification"].astype(str).str.lower()
                        != "fbs")
                       | (slate["away_classification"].astype(str).str.lower()
                          != "fbs"))
        dropped = int(is_fcs_game.sum())
        if dropped:
            slate = slate[~is_fcs_game]
            print(f"  filtered out {dropped} FBS-vs-FCS game(s) "
                  f"(--include-fcs to keep them)")
    print(f"Slate: {len(slate)} of {n_all} candidate games "
          f"({'CFBD' if settings.has_cfbd_key else 'SYNTHETIC'})")

    preds: List[GamePrediction] = []
    for r in slate.itertuples(index=False):
        X = fb.features_for_matchup(
            r.home_team, r.away_team, neutral=bool(r.neutral_site),
            home_conf=r.home_conference, away_conf=r.away_conference,
            conference_game=bool(r.conference_game),
            week=int(r.week) if pd.notna(r.week) else 8,
            postseason=str(getattr(r, "season_type", "regular")).lower()
            != "regular")
        probs = model.predict_proba(X)
        gp = GamePrediction.from_probs(
            game_date=(r.start_date.date().isoformat()
                       if pd.notna(r.start_date) else dt.date.today().isoformat()),
            home_team=r.home_team, away_team=r.away_team,
            p_blended=float(probs["p_blended"].iloc[0]),
            p_model=float(probs["p_model"].iloc[0]),
            p_elo=float(probs["p_elo"].iloc[0]),
            elo_home=fb.elo.get(r.home_team, r.home_conference),
            elo_away=fb.elo.get(r.away_team, r.away_conference),
            game_id=(str(int(r.game_id)) if pd.notna(getattr(r, "game_id", None))
                     else None),
            season=int(r.season) if pd.notna(r.season) else None,
            week=int(r.week) if pd.notna(r.week) else None,
            neutral_site=bool(r.neutral_site),
            conference_game=bool(r.conference_game),
            games_played_min=min(fb.games_played(r.home_team),
                                 fb.games_played(r.away_team)))
        if margin_model is not None:
            gp.attach_margin(float(margin_model.predict_margin(X)[0]), None)

        # Carry a few feature values so the blog can explain WHY, and record how
        # much played football backs each side — that governs the ledger's
        # data_depth and caps how confident a tier is allowed to be.
        row = X.iloc[0]
        gp.drivers = {k: float(row[k]) for k in
                      ("net_ppa_diff", "preseason_prior_diff", "elo_diff_with_hca")
                      if k in X.columns}
        preds.append(gp)

    date_label = dt.date.today().strftime("%B %d, %Y")
    md = build_newsletter(preds, date_label)
    html = markdown_to_html(md)

    week_no = None
    wk = [p.week for p in preds if p.week is not None]
    if wk:
        week_no = int(min(wk))
    blog = render_blog(preds, date_label, week=week_no)

    out_dir = settings.content_root / dt.date.today().isoformat()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "newsletter.md").write_text(md)
    (out_dir / "newsletter.html").write_text(html)
    (out_dir / "blog.md").write_text(blog)
    (out_dir / "blog.html").write_text(markdown_to_html(blog))

    # asof_ts = the latest data the model was permitted to see. Stamping it on
    # every row is what lets the ledger enforce the leakage invariant instead of
    # trusting that we ran the pipeline at a sensible time.
    asof = (completed["start_date"].max().isoformat()
            if len(completed) else dt.datetime.now(dt.timezone.utc).isoformat())
    payload = []
    for p in preds:
        d = asdict(p)
        d["asof_ts"] = asof
        payload.append(d)
    (out_dir / "predictions.json").write_text(
        json.dumps(payload, indent=2, default=str))

    print(f"  wrote → {out_dir}/")
    print("           newsletter.md/.html   (scannable briefing)")
    print("           blog.md/.html         (narrative post)")
    print("           predictions.json      (ledger input)")
    locks = sum(1 for p in preds if "Lock" in p.tier)
    depths = {}
    for p_ in preds:
        from ballknower_quad.ledger.emit_records import data_depth_for
        d = data_depth_for(p_.games_played_min)
        depths[d] = depths.get(d, 0) + 1
    print(f"  {len(preds)} games · {locks} Locks · asof {asof[:16]}")
    print(f"  data depth: {depths}"
          + ("   (thin/none caps the tier — early-season ratings lean on "
             "preseason priors)" if depths.get("rich", 0) < len(preds) else ""))
    print("\n" + settings.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
