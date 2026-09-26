# BallKnower Quad — Calibration

_Pre-registered college football forecasts, live forecasts only. Every row was committed before kickoff and graded afterward whether it worked or not._

## Win probability

| metric | value |
|---|---:|
| forecasts graded | 147 |
| hit rate | 0.803 |
| average confidence | 0.739 |
| Brier | 0.1385 |
| log loss | 0.4333 |
| calibration error (ECE) | 0.0733 |

The number that matters is the gap between the last two: we said **73.9%** on average and hit **80.3%** (+6.4%). Being right more often than we claimed is as much a calibration miss as the reverse — it just flatters us.

### By confidence tier

| data depth | tier | picks | advertised | hit rate | gap | ±1 SE |
|---|---|---:|---:|---:|---:|---:|
| none | Lean | 35 | 0.938 | 0.971 | +0.033 | 0.028 |
| none | Pass | 112 | 0.677 | 0.750 | +0.073 | 0.041 |

**Data depth** is how much football the ratings are built on. In the opening weeks a team's rating comes mostly from preseason roster signals rather than results, so the model caps how confident a label it will publish — which is why an early-season tier can advertise a probability well above its name. Depths are reported separately because pooling them would average two different things.

## Margin

| metric | value |
|---|---:|
| forecasts graded | 147 |
| CRPS | 8.794 |
| mean absolute error | 12.37 pts |
| bias (predicted − actual) | -2.20 pts |
| inside ±1σ | 0.755 (target 0.683) |
| inside ±2σ | 0.952 (target 0.954) |

Coverage well under target means the margin model is overconfident — the stated uncertainty is too narrow. Over target means it's hedging.

## Last slate (2026-09-13) — 48-9

| | pick | said | tier | result |
|---|---|---:|---|---|
| ✅ | Indiana defeats Western Kentucky | 98% | Lean | Indiana won 38-0 |
| ✅ | Penn State defeats Buffalo | 98% | Lean | Penn State won 55-13 |
| ✅ | Ohio State defeats Kent State | 97% | Lean | Ohio State won 59-3 |
| ✅ | Utah defeats Utah State | 97% | Lean | Utah won 33-0 |
| ✅ | Michigan defeats UTEP | 97% | Lean | Michigan won 52-17 |
| ✅ | Arizona defeats Northern Illinois | 96% | Lean | Arizona won 42-17 |
| ✅ | Tennessee defeats Kennesaw State | 96% | Lean | Tennessee won 42-9 |
| ✅ | Iowa State defeats Bowling Green | 96% | Lean | Iowa State won 55-7 |
| ✅ | Alabama defeats Florida State | 96% | Lean | Alabama won 50-36 |
| ✅ | Notre Dame defeats Michigan State | 94% | Lean | Notre Dame won 27-10 |
| ✅ | Missouri defeats Troy | 91% | Lean | Missouri won 27-17 |
| ✅ | Georgia defeats Arkansas | 88% | Lean | Georgia won 45-17 |
| ✅ | Texas defeats UTSA | 88% | Lean | Texas won 30-6 |
| ✅ | Wisconsin defeats Eastern Michigan | 87% | Lean | Wisconsin won 54-10 |
| ❌ | Texas A&M defeats Kentucky | 83% | Pass | Kentucky won 31-21 |
| ✅ | Pittsburgh defeats Syracuse | 82% | Pass | Pittsburgh won 27-13 |
| ✅ | App State defeats Charlotte | 82% | Pass | App State won 26-21 |
| ✅ | USC defeats Rutgers | 82% | Pass | USC won 42-35 |
| ✅ | UCLA defeats Purdue | 82% | Pass | UCLA won 52-38 |
| ✅ | Minnesota defeats Akron | 81% | Pass | Minnesota won 41-7 |
| ✅ | Oklahoma defeats New Mexico | 80% | Pass | Oklahoma won 14-6 |
| ✅ | Miami defeats Wake Forest | 77% | Pass | Miami won 33-20 |
| ✅ | Liberty defeats Ball State | 77% | Pass | Liberty won 51-15 |
| ✅ | TCU defeats Arkansas State | 74% | Pass | TCU won 31-7 |
| ✅ | Toledo defeats Temple | 73% | Pass | Toledo won 49-48 |
| ✅ | Cincinnati defeats Miami (OH) | 73% | Pass | Cincinnati won 35-31 |
| ✅ | Kansas State defeats Tulane | 72% | Pass | Kansas State won 31-20 |
| ✅ | Baylor defeats Louisiana Tech | 71% | Pass | Baylor won 36-19 |
| ✅ | UCF defeats Georgia State | 70% | Pass | UCF won 44-30 |
| ✅ | Duke defeats Stanford | 69% | Pass | Duke won 35-7 |
| ✅ | Vanderbilt defeats NC State | 68% | Pass | Vanderbilt won 35-31 |
| ✅ | Texas Tech defeats Houston | 68% | Pass | Texas Tech won 28-26 |
| ✅ | South Alabama defeats Ohio | 67% | Pass | South Alabama won 41-36 |
| ✅ | Ole Miss defeats LSU | 66% | Pass | Ole Miss won 32-24 |
| ✅ | Central Michigan defeats Wyoming | 66% | Pass | Central Michigan won 24-10 |
| ✅ | Northwestern defeats Colorado | 66% | Pass | Northwestern won 41-7 |
| ✅ | Clemson defeats North Carolina | 65% | Pass | Clemson won 28-20 |
| ❌ | Virginia wins at a neutral site | 65% | Pass | West Virginia won 38-27 |
| ✅ | BYU defeats Colorado State | 64% | Pass | BYU won 41-23 |
| ✅ | Delaware defeats Coastal Carolina | 64% | Pass | Delaware won 22-14 |
| ✅ | Louisiana defeats UAB | 64% | Pass | Louisiana won 21-14 |
| ❌ | Old Dominion defeats East Carolina | 63% | Pass | East Carolina won 20-17 |
| ❌ | Maryland defeats Virginia Tech | 63% | Pass | Virginia Tech won 35-26 |
| ✅ | Western Michigan defeats Rice | 62% | Pass | Western Michigan won 28-21 |
| ❌ | South Carolina defeats Mississippi State | 62% | Pass | Mississippi State won 41-34 |
| ❌ | Missouri State defeats Marshall | 61% | Pass | Marshall won 30-24 |
| ✅ | Florida Atlantic defeats Florida International | 61% | Pass | Florida Atlantic won 16-10 |
| ✅ | North Dakota State defeats Sacramento State | 61% | Pass | North Dakota State won 31-10 |
| ✅ | Arizona State wins at a neutral site | 60% | Pass | Arizona State won 24-17 |
| ✅ | Fresno State defeats San José State | 59% | Pass | Fresno State won 26-10 |
| ✅ | James Madison defeats San Diego State | 59% | Pass | James Madison won 26-13 |
| ❌ | SMU defeats Louisville | 57% | Pass | Louisville won 41-31 |
| ✅ | Middle Tennessee defeats Nevada | 56% | Pass | Middle Tennessee won 27-20 |
| ❌ | Southern Miss defeats UConn | 55% | Pass | UConn won 48-20 |
| ✅ | Florida defeats Auburn | 54% | Pass | Florida won 44-39 |
| ❌ | Georgia Southern defeats Jacksonville State | 53% | Pass | Jacksonville State won 31-27 |
| ✅ | Texas State defeats North Texas | 50% | Pass | Texas State won 49-35 |


---
_For entertainment and educational purposes only — not financial or betting advice._
