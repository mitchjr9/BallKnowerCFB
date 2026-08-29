"""
ballknower_quad.scripts.train
=============================

Fit and persist the V1 models. Run this first.

    python -m ballknower_quad.scripts.train                 # win-prob + margin
    python -m ballknower_quad.scripts.train --no-margin     # win-prob only
    python -m ballknower_quad.scripts.train --seasons-back 12

With no CFBD_API_KEY set, trains on a deterministic synthetic history so you can
verify the whole pipeline offline.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from typing import List, Optional

from ballknower_quad.data.bundle import load_bundle, season_range
from ballknower_quad.models.cfb_margin_v1 import train_margin_v1
from ballknower_quad.models.cfb_model_v1 import train_v1
from ballknower_quad.models.feature_builder import (has_ppa_features,
                                                    has_preseason_features)
from ballknower_quad.settings import settings


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train BallKnower Quad V1.")
    p.add_argument("--seasons-back", type=int, default=11)
    p.add_argument("--version", default=None,
                   help="feature-set version: v1 (Elo+margin) or v1.1 (adds PPA). "
                        "Default: settings.active_model_version")
    p.add_argument("--no-margin", action="store_true")
    p.add_argument("--blend-elo", type=float, default=None)
    args = p.parse_args(argv)

    version = (args.version or settings.active_model_version).lower()
    seasons = season_range(args.seasons_back)
    src = "CFBD" if settings.has_cfbd_key else "SYNTHETIC (no CFBD_API_KEY)"
    print(f"Loading seasons {seasons[0]}–{seasons[-1]} [{src}] for {version} …")
    bundle = load_bundle(seasons, version=version)
    games = bundle.games
    completed = games[games["completed"]]
    print(f"  {len(games):,} games ({len(completed):,} completed), "
          f"{games['home_team'].nunique()} teams")
    if has_ppa_features(version):
        npg = 0 if bundle.game_ppa is None else len(bundle.game_ppa)
        nsp = 0 if bundle.season_ppa is None else len(bundle.season_ppa)
        print(f"  PPA: {npg:,} team-game rows | {nsp:,} season-anchor rows")
    if has_preseason_features(version):
        npre = 0 if bundle.preseason is None else len(bundle.preseason)
        print(f"  Preseason: {npre:,} team-season rows "
              f"(blend={settings.preseason_blend})")
    print()

    print(f"Training {version} win-probability model …")
    model, fb = train_v1(games, blend_elo=args.blend_elo, calibrate=True,
                         version=version, game_ppa=bundle.game_ppa,
                         season_ppa=bundle.season_ppa,
                         preseason=bundle.preseason)
    mdir = settings.models_dir_for(version)
    model.save(mdir)
    fb.elo.save(settings.models_store_root / "elo_state.json")
    if fb.ppa is not None:
        fb.ppa.save(settings.models_store_root / "ppa_state.json")
    print(f"  saved → {mdir}")
    print(f"  features ({len(model.feature_columns)}): "
          f"{', '.join(model.feature_columns)}")
    print("  holdout metrics: " + json.dumps(model.metrics, indent=2))

    print("\n  Top 15 Elo:")
    print(fb.elo.top(15).to_string(index=False))
    if fb.ppa is not None:
        print("\n  Top 15 net PPA (opponent-adjusted):")
        print(fb.ppa.top(15).to_string(index=False))
    if getattr(fb, "prior", None) is not None:
        cur = max(seasons)
        tbl = fb.prior.top(cur, 15)
        if len(tbl):
            print(f"\n  Top 15 preseason prior ({cur}):")
            print(tbl.to_string(index=False))

    if not args.no_margin:
        print(f"\nTraining {version} margin model …")
        mm = train_margin_v1(games, version=version, game_ppa=bundle.game_ppa,
                             season_ppa=bundle.season_ppa,
                             preseason=bundle.preseason)
        mm.save(settings.models_dir_for(version))
        print("  margin metrics: " + json.dumps(mm.metrics, indent=2))

    print("\n" + settings.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
