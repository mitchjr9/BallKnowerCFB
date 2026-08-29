"""
ballknower_quad.scripts.check_api
=================================

Run this FIRST, before training on real data. It answers three questions in one
shot, for ~2-3 API calls:

    1. Is my CFBD_API_KEY being picked up, and does it actually authenticate?
    2. How many of my 1,000 monthly calls are left? (shared CFB + CBB pool)
    3. Can I pull real games right now?

    python -m ballknower_quad.scripts.check_api
    python -m ballknower_quad.scripts.check_api --no-live   # skip the games test

Note: ``/info`` returns null when a request isn't authenticated, so a null/None
response here means the key is missing or wrong — not that the API is down.
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from ballknower_quad.data.cfbd_loader import (fetch_account_info, fetch_usage,
                                              load_games)
from ballknower_quad.settings import settings


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Verify CFBD API access and quota.")
    p.add_argument("--no-live", action="store_true",
                   help="skip the live /games pull")
    p.add_argument("--year", type=int, default=2024,
                   help="season to use for the live test (default 2024)")
    args = p.parse_args(argv)

    print("=" * 60)
    print("  BallKnower Quad — CFBD API check")
    print("=" * 60)
    print(f"  base URL : {settings.cfbd_base_url}")

    if not settings.has_cfbd_key:
        print("  key      : ✗ NOT FOUND")
        print("\n  CFBD_API_KEY isn't set. Either:")
        print("    • create a .env next to ballknower_quad/ with:")
        print("        CFBD_API_KEY=your_key_here")
        print("    • or export it:  export CFBD_API_KEY='your_key_here'")
        print("\n  Get a free key: https://collegefootballdata.com/key")
        print("  (The package still runs in SYNTHETIC mode without one.)")
        return 1

    masked = settings.cfbd_api_key[:4] + "…" + settings.cfbd_api_key[-4:]
    print(f"  key      : found ({masked})")

    info = fetch_account_info()
    if info is None:
        print("  auth     : ✗ /info returned null — key not authenticating.")
        return 1
    if isinstance(info, dict) and "_error" in info:
        print(f"  auth     : ✗ {info['_error']} — {info.get('_reason')}")
        if "401" in str(info.get("_error")):
            print("             401 = bad/expired key. Request a fresh one.")
        return 1

    print("  auth     : ✓ authenticated\n")
    print(f"  tier          : {info.get('tierName')} "
          f"(patron level {info.get('patronLevel')})")
    print(f"  monthly limit : {info.get('monthlyLimit')}")
    print(f"  used / left   : {info.get('usedCalls')} / {info.get('remainingCalls')}")
    print(f"  resets at     : {info.get('resetAt')}")
    if info.get("sharedPool"):
        print("  pool          : SHARED with college basketball (CBB) calls")
    feats = info.get("features") or {}
    if feats:
        on = [k for k, v in feats.items() if v]
        off = [k for k, v in feats.items() if not v]
        print(f"  features on   : {', '.join(on) if on else '(none)'}")
        if off:
            print(f"  features off  : {', '.join(off)}   <- tier-gated")

    usage = fetch_usage(days=7)
    if isinstance(usage, dict) and "totals" in usage:
        t = usage["totals"]
        print(f"\n  last 7 days   : {t.get('requests')} requests "
              f"({t.get('cfbRequests')} CFB / {t.get('cbbRequests')} CBB)")

    if not args.no_live:
        print(f"\n  Live test — pulling {args.year} games …")
        games = load_games([args.year])
        completed = int(games["completed"].sum()) if len(games) else 0
        print(f"    {len(games):,} games returned ({completed:,} completed), "
              f"{games['home_team'].nunique()} teams")
        if len(games):
            print("    sample:")
            cols = ["season", "week", "home_team", "home_points",
                    "away_team", "away_points"]
            print(games[cols].head(3).to_string(index=False))
            print("\n  ✓ Real data is flowing. You're clear to train.")
        else:
            print("  ✗ No games returned — check the year/classification filter.")
            return 1

    print("\n" + settings.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
