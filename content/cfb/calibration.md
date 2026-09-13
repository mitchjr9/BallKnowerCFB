# BallKnower Quad — Calibration

_Pre-registered college football forecasts, live forecasts only. Every row was committed before kickoff and graded afterward whether it worked or not._

## Win probability

| metric | value |
|---|---:|
| forecasts graded | 90 |
| hit rate | 0.778 |
| average confidence | 0.740 |
| Brier | 0.1455 |
| log loss | 0.4509 |
| calibration error (ECE) | 0.0700 |

The number that matters is the gap between the last two: we said **74.0%** on average and hit **77.8%** (+3.8%). Being right more often than we claimed is as much a calibration miss as the reverse — it just flatters us.

### By confidence tier

| data depth | tier | picks | advertised | hit rate | gap | ±1 SE |
|---|---|---:|---:|---:|---:|---:|
| none | Lean | 21 | 0.936 | 0.952 | +0.016 | 0.046 |
| none | Pass | 69 | 0.680 | 0.725 | +0.045 | 0.054 |

**Data depth** is how much football the ratings are built on. In the opening weeks a team's rating comes mostly from preseason roster signals rather than results, so the model caps how confident a label it will publish — which is why an early-season tier can advertise a probability well above its name. Depths are reported separately because pooling them would average two different things.

## Margin

| metric | value |
|---|---:|
| forecasts graded | 90 |
| CRPS | 9.904 |
| mean absolute error | 14.10 pts |
| bias (predicted − actual) | -3.50 pts |
| inside ±1σ | 0.700 (target 0.683) |
| inside ±2σ | 0.922 (target 0.954) |

Coverage well under target means the margin model is overconfident — the stated uncertainty is too narrow. Over target means it's hedging.

## Last slate (2026-09-12) — 35-12

| | pick | said | tier | result |
|---|---|---:|---|---|
| ✅ | Ole Miss defeats Charlotte | 99% | Lean | Ole Miss won 41-9 |
| ✅ | Georgia defeats Western Kentucky | 98% | Lean | Georgia won 70-20 |
| ✅ | Notre Dame defeats Rice | 97% | Lean | Notre Dame won 52-0 |
| ✅ | Nebraska defeats Bowling Green | 96% | Lean | Nebraska won 56-7 |
| ✅ | Washington defeats Utah State | 96% | Lean | Washington won 16-14 |
| ✅ | Utah defeats Arkansas | 89% | Lean | Utah won 43-10 |
| ✅ | USC defeats Louisiana | 88% | Lean | USC won 49-30 |
| ✅ | LSU defeats Louisiana Tech | 87% | Lean | LSU won 45-14 |
| ✅ | UAB defeats UL Monroe | 86% | Lean | UAB won 26-20 |
| ✅ | Auburn defeats Southern Miss | 85% | Pass | Auburn won 43-8 |
| ❌ | Oregon defeats Oklahoma State | 83% | Pass | Oklahoma State won 39-31 |
| ✅ | Vanderbilt defeats Delaware | 82% | Pass | Vanderbilt won 35-26 |
| ✅ | Clemson defeats Georgia Southern | 82% | Pass | Clemson won 22-7 |
| ✅ | Texas A&M defeats Arizona State | 81% | Pass | Texas A&M won 48-20 |
| ✅ | Texas Tech defeats Oregon State | 81% | Pass | Texas Tech won 35-24 |
| ✅ | Penn State defeats Temple | 77% | Pass | Penn State won 27-9 |
| ✅ | Michigan State defeats Eastern Michigan | 76% | Pass | Michigan State won 35-7 |
| ✅ | Kansas State defeats Washington State | 75% | Pass | Kansas State won 34-7 |
| ✅ | Hawai'i defeats New Mexico State | 75% | Pass | Hawai'i won 29-19 |
| ✅ | Florida International defeats Buffalo | 74% | Pass | Florida International won 33-20 |
| ✅ | Alabama defeats Kentucky | 74% | Pass | Alabama won 45-17 |
| ✅ | Tulane defeats South Alabama | 69% | Pass | Tulane won 28-24 |
| ✅ | Pittsburgh defeats UCF | 67% | Pass | Pittsburgh won 12-7 |
| ✅ | Iowa defeats Iowa State | 67% | Pass | Iowa won 16-13 |
| ✅ | North Dakota State defeats Air Force | 67% | Pass | North Dakota State won 38-32 |
| ❌ | East Carolina defeats App State | 67% | Pass | App State won 27-24 |
| ❌ | Minnesota defeats Mississippi State | 66% | Pass | Mississippi State won 38-13 |
| ✅ | Fresno State defeats Sacramento State | 65% | Pass | Fresno State won 49-3 |
| ✅ | Ohio defeats Jacksonville State | 64% | Pass | Ohio won 29-27 |
| ✅ | Texas defeats Ohio State | 63% | Pass | Texas won 24-23 |
| ✅ | UCLA defeats San Diego State | 63% | Pass | UCLA won 28-10 |
| ❌ | Syracuse defeats California | 63% | Pass | California won 21-18 |
| ❌ | Kennesaw State defeats Georgia State | 61% | Pass | Georgia State won 31-17 |
| ❌ | Illinois defeats Duke | 61% | Pass | Duke won 31-27 |
| ✅ | Virginia Tech defeats Old Dominion | 60% | Pass | Virginia Tech won 44-21 |
| ✅ | Tulsa defeats Sam Houston | 60% | Pass | Tulsa won 23-17 |
| ✅ | Tennessee defeats Georgia Tech | 60% | Pass | Tennessee won 45-24 |
| ❌ | UConn defeats Maryland | 59% | Pass | Maryland won 38-14 |
| ✅ | South Florida defeats Army | 58% | Pass | South Florida won 28-24 |
| ✅ | Marshall defeats Middle Tennessee | 58% | Pass | Marshall won 28-26 |
| ✅ | UTSA defeats Texas State | 58% | Pass | UTSA won 31-26 |
| ❌ | Memphis defeats Boise State | 58% | Pass | Boise State won 38-20 |
| ❌ | Purdue defeats Wake Forest | 57% | Pass | Wake Forest won 38-36 |
| ❌ | Oklahoma defeats Michigan | 57% | Pass | Michigan won 17-10 |
| ✅ | North Texas defeats UNLV | 56% | Pass | North Texas won 44-6 |
| ❌ | Arizona defeats BYU | 54% | Pass | BYU won 28-17 |
| ❌ | Navy defeats Florida Atlantic | 51% | Pass | Florida Atlantic won 38-30 |


---
_For entertainment and educational purposes only — not financial or betting advice._
