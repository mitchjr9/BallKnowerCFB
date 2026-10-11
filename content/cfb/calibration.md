# BallKnower Quad — Calibration

_Pre-registered college football forecasts, live forecasts only. Every row was committed before kickoff and graded afterward whether it worked or not._

## Win probability

| metric | value |
|---|---:|
| forecasts graded | 254 |
| hit rate | 0.732 |
| average confidence | 0.718 |
| Brier | 0.1717 |
| log loss | 0.5135 |
| calibration error (ECE) | 0.0412 |

The number that matters is the gap between the last two: we said **71.8%** on average and hit **73.2%** (+1.5%). Being right more often than we claimed is as much a calibration miss as the reverse — it just flatters us.

### By confidence tier

| data depth | tier | picks | advertised | hit rate | gap | ±1 SE |
|---|---|---:|---:|---:|---:|---:|
| rich | Lean | 1 | 0.877 | 1.000 | +0.123 | 0.000 |
| rich | Lock | 2 | 0.982 | 1.000 | +0.018 | 0.000 |
| rich | Pass | 21 | 0.644 | 0.619 | -0.025 | 0.106 |
| rich | Strong | 2 | 0.926 | 0.500 | -0.426 | 0.354 |
| thin | Pass | 15 | 0.660 | 0.333 | -0.327 | 0.122 |
| thin | Strong | 1 | 0.981 | 1.000 | +0.019 | 0.000 |
| none | Lean | 45 | 0.928 | 0.956 | +0.027 | 0.031 |
| none | Pass | 167 | 0.667 | 0.719 | +0.051 | 0.035 |

**Data depth** is how much football the ratings are built on. In the opening weeks a team's rating comes mostly from preseason roster signals rather than results, so the model caps how confident a label it will publish — which is why an early-season tier can advertise a probability well above its name. Depths are reported separately because pooling them would average two different things.

## Margin

| metric | value |
|---|---:|
| forecasts graded | 254 |
| CRPS | 9.390 |
| mean absolute error | 13.20 pts |
| bias (predicted − actual) | -1.63 pts |
| inside ±1σ | 0.736 (target 0.683) |
| inside ±2σ | 0.945 (target 0.954) |

Coverage well under target means the margin model is overconfident — the stated uncertainty is too narrow. Over target means it's hedging.

## Last slate (2026-10-10) — 25-21

| | pick | said | tier | result |
|---|---|---:|---|---|
| ✅ | Ohio State defeats Maryland | 98% | Lock | Ohio State won 45-16 |
| ✅ | Northwestern defeats Ball State | 98% | Strong | Northwestern won 45-14 |
| ✅ | Notre Dame defeats Stanford | 98% | Lock | Notre Dame won 58-7 |
| ✅ | North Texas defeats Charlotte | 94% | Strong | North Texas won 50-10 |
| ❌ | Toledo defeats Buffalo | 92% | Strong | Buffalo won 23-8 |
| ✅ | Western Michigan defeats Kent State | 88% | Lean | Western Michigan won 38-7 |
| ✅ | Utah defeats Kansas | 87% | Lean | Utah won 41-24 |
| ❌ | Marshall defeats Coastal Carolina | 85% | Pass | Coastal Carolina won 27-21 |
| ✅ | East Carolina defeats Rice | 85% | Pass | East Carolina won 28-12 |
| ❌ | App State defeats Old Dominion | 82% | Pass | Old Dominion won 10-3 |
| ❌ | Oregon defeats UCLA | 81% | Pass | UCLA won 17-15 |
| ✅ | Tennessee defeats Arkansas | 81% | Pass | Tennessee won 24-17 |
| ✅ | Arizona defeats West Virginia | 80% | Pass | Arizona won 52-31 |
| ✅ | James Madison defeats Georgia Southern | 75% | Pass | James Madison won 27-24 |
| ❌ | Texas wins at a neutral site | 74% | Pass | Oklahoma won 27-24 |
| ❌ | Pittsburgh defeats North Carolina | 72% | Pass | North Carolina won 21-17 |
| ✅ | Air Force defeats Northern Illinois | 72% | Pass | Air Force won 28-7 |
| ✅ | Arizona State defeats Hawai'i | 71% | Pass | Arizona State won 30-12 |
| ✅ | Virginia defeats Syracuse | 71% | Pass | Virginia won 28-25 |
| ✅ | Eastern Michigan defeats Akron | 71% | Pass | Eastern Michigan won 25-23 |
| ❌ | Florida defeats South Carolina | 71% | Pass | South Carolina won 38-19 |
| ✅ | Memphis defeats UAB | 70% | Pass | Memphis won 38-13 |
| ✅ | Missouri defeats Texas A&M | 69% | Pass | Missouri won 27-6 |
| ✅ | Oregon State defeats San Diego State | 68% | Pass | Oregon State won 66-14 |
| ✅ | Ohio defeats Central Michigan | 67% | Pass | Ohio won 31-14 |
| ❌ | LSU defeats Kentucky | 67% | Pass | Kentucky won 35-34 |
| ❌ | Duke defeats Georgia Tech | 65% | Pass | Georgia Tech won 28-25 |
| ❌ | Kansas State defeats Houston | 63% | Pass | Houston won 35-16 |
| ❌ | NC State defeats Wake Forest | 63% | Pass | Wake Forest won 38-14 |
| ❌ | Virginia Tech defeats California | 63% | Pass | California won 25-20 |
| ✅ | Ole Miss defeats Vanderbilt | 58% | Pass | Ole Miss won 52-17 |
| ❌ | Minnesota defeats Purdue | 57% | Pass | Purdue won 41-13 |
| ✅ | Nevada defeats UTEP | 57% | Pass | Nevada won 30-13 |
| ✅ | Illinois defeats Michigan State | 56% | Pass | Illinois won 38-37 |
| ❌ | Fresno State defeats Boise State | 55% | Pass | Boise State won 38-31 |
| ✅ | UConn defeats Temple | 55% | Pass | UConn won 34-20 |
| ✅ | UNLV defeats North Dakota State | 54% | Pass | UNLV won 13-7 |
| ❌ | Penn State defeats USC | 54% | Pass | USC won 31-23 |
| ❌ | Tulsa defeats Navy | 54% | Pass | Navy won 34-26 |
| ❌ | UCF defeats Oklahoma State | 53% | Pass | Oklahoma State won 42-28 |
| ❌ | Sacramento State defeats Bowling Green | 53% | Pass | Bowling Green won 38-10 |
| ❌ | Louisiana defeats Louisiana Tech | 53% | Pass | Louisiana Tech won 32-31 |
| ❌ | Tulane defeats Army | 52% | Pass | Army won 20-17 |
| ✅ | Miami (OH) defeats Massachusetts | 51% | Pass | Miami (OH) won 20-19 |
| ❌ | Alabama defeats Georgia | 51% | Pass | Georgia won 20-17 |
| ✅ | Indiana defeats Nebraska | 50% | Pass | Indiana won 20-17 |


---
_For entertainment and educational purposes only — not financial or betting advice._
