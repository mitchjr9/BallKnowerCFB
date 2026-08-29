"""
ballknower_quad.scripts.predict
===============================

Predict college-football games with the trained V1 models.

    # single matchup (home first), optional neutral site / spread
    python -m ballknower_quad.scripts.predict "Ohio State" "Michigan"
    python -m ballknower_quad.scripts.predict Georgia Texas --neutral --spread -3.5
    python -m ballknower_quad.scripts.predict "Boise State" "Toledo" --no-margin

The model must be trained first (python -m ballknower_quad.scripts.train).
Team names must match CFBD spellings (e.g. "Ohio State", "Texas A&M"). Spread
uses CFBD convention: positive = home favored.

DISCLAIMER: entertainment/education only — not betting advice.
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import List, Optional

from ballknower_quad.data.bundle import load_bundle, season_range
from ballknower_quad.models.cfb_elo import CFBElo
from ballknower_quad.models.cfb_margin_v1 import CFBMarginV1
from ballknower_quad.models.cfb_model_v1 import CFBModelV1
from ballknower_quad.models.feature_builder import FeatureBuilder
from ballknower_quad.content_utils import GamePrediction, confidence_tier
from ballknower_quad.settings import settings


def _warm_builder(version: str, seasons_back: int = 11) -> FeatureBuilder:
    """Rebuild a FeatureBuilder warmed to the latest games (Elo + efficiency + PPA)."""
    seasons = season_range(seasons_back)
    bundle = load_bundle(seasons, version=version)
    fb = FeatureBuilder(version=version, game_ppa=bundle.game_ppa,
                        season_ppa=bundle.season_ppa,
                        preseason=bundle.preseason)
    elo_path = settings.models_store_root / "elo_state.json"
    if elo_path.exists():
        fb.elo = CFBElo.load(elo_path)
    # Walk completed games to rebuild rolling efficiency + PPA state. Elo is only
    # re-fit when we don't already have a saved warm state.
    fb.build(bundle.games, fit_elo=not elo_path.exists())
    return fb


def _load_margin(version: str) -> Optional[CFBMarginV1]:
    try:
        return CFBMarginV1.load(settings.models_dir_for(version))
    except Exception:
        return None


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Predict a CFB matchup.")
    p.add_argument("home")
    p.add_argument("away")
    p.add_argument("--neutral", action="store_true")
    p.add_argument("--conference", action="store_true",
                   help="flag as a conference game")
    p.add_argument("--postseason", action="store_true",
                   help="bowl / playoff game (v1.2+ treats these differently "
                        "from neutral-site regular-season openers)")
    p.add_argument("--week", type=int, default=8)
    p.add_argument("--home-rest", type=float, default=7.0)
    p.add_argument("--away-rest", type=float, default=7.0)
    p.add_argument("--spread", type=float, default=None,
                   help="CFBD convention: positive = home favored")
    p.add_argument("--no-margin", action="store_true")
    p.add_argument("--blend-elo", type=float, default=None)
    p.add_argument("--version", default=None,
                   help="model version to load (default: active_model_version)")
    args = p.parse_args(argv)

    version = (args.version or settings.active_model_version).lower()
    try:
        model = CFBModelV1.load(settings.models_dir_for(version))
    except Exception:
        print(f"No trained {version} model found. Run: "
              f"python -m ballknower_quad.scripts.train --version {version}")
        return 1

    fb = _warm_builder(version)
    X = fb.features_for_matchup(
        args.home, args.away, neutral=args.neutral,
        conference_game=args.conference, week=args.week,
        home_rest=args.home_rest, away_rest=args.away_rest,
        postseason=args.postseason)
    probs = model.predict_proba(X, blend_elo=args.blend_elo)
    p_blended = float(probs["p_blended"].iloc[0])

    pred = GamePrediction.from_probs(
        game_date=dt.date.today().isoformat(),
        home_team=args.home, away_team=args.away,
        p_blended=p_blended, p_model=float(probs["p_model"].iloc[0]),
        p_elo=float(probs["p_elo"].iloc[0]),
        elo_home=fb.elo.get(args.home), elo_away=fb.elo.get(args.away),
        neutral_site=args.neutral, conference_game=args.conference, week=args.week)

    margin_model = None if args.no_margin else _load_margin(version)
    if margin_model is not None:
        pm = float(margin_model.predict_margin(X)[0])
        pred.attach_margin(pm, args.spread)

    sep = "vs (neutral)" if args.neutral else "@"
    print("\n" + "=" * 56)
    print(f"  {args.away} {sep} {args.home}")
    print("=" * 56)
    print(f"  Elo:        {args.home} {pred.elo_home:.0f}  |  "
          f"{args.away} {pred.elo_away:.0f}")
    print(f"  Model win%: {args.home} {p_blended * 100:.1f}%  "
          f"(raw {probs['p_model'].iloc[0] * 100:.1f}%, "
          f"Elo {probs['p_elo'].iloc[0] * 100:.1f}%)")
    print(f"  Pick:       {pred.favorite}  ({pred.fav_prob * 100:.1f}%)  "
          f"{pred.tier}")
    if pred.predicted_margin is not None:
        print(f"  Margin:     {args.home} {pred.predicted_margin:+.1f}")
        if pred.ats_gap is not None:
            print(f"  ATS:        line {args.spread:+.1f} → leans "
                  f"{pred.ats_pick} by {abs(pred.ats_gap):.1f}")
    print("\n" + settings.disclaimer + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
