# BallKnower Quad — Calibration

_Pre-registered college football forecasts, live forecasts only. Every row was committed before kickoff and graded afterward whether it worked or not._

## Win probability

| metric | value |
|---|---:|
| forecasts graded | 208 |
| hit rate | 0.774 |
| average confidence | 0.723 |
| Brier | 0.1544 |
| log loss | 0.4730 |
| calibration error (ECE) | 0.0635 |

The number that matters is the gap between the last two: we said **72.3%** on average and hit **77.4%** (+5.1%). Being right more often than we claimed is as much a calibration miss as the reverse — it just flatters us.

### By confidence tier

| data depth | tier | picks | advertised | hit rate | gap | ±1 SE |
|---|---|---:|---:|---:|---:|---:|
| none | Lean | 44 | 0.929 | 0.955 | +0.025 | 0.031 |
| none | Pass | 164 | 0.668 | 0.726 | +0.058 | 0.035 |

**Data depth** is how much football the ratings are built on. In the opening weeks a team's rating comes mostly from preseason roster signals rather than results, so the model caps how confident a label it will publish — which is why an early-season tier can advertise a probability well above its name. Depths are reported separately because pooling them would average two different things.

## Margin

| metric | value |
|---|---:|
| forecasts graded | 208 |
| CRPS | 9.421 |
| mean absolute error | 13.20 pts |
| bias (predicted − actual) | -1.77 pts |
| inside ±1σ | 0.726 (target 0.683) |
| inside ±2σ | 0.942 (target 0.954) |

Coverage well under target means the margin model is overconfident — the stated uncertainty is too narrow. Over target means it's hedging.

## Last slate (2026-10-02) — 37-15

| | pick | said | tier | result |
|---|---|---:|---|---|
| ✅ | Georgia defeats Vanderbilt | 94% | Lean | Georgia won 38-14 |
| ✅ | Boise State defeats Utah State | 89% | Lean | Boise State won 37-18 |
| ❌ | South Florida defeats Temple | 89% | Lean | Temple won 17-13 |
| ✅ | South Alabama defeats UL Monroe | 89% | Lean | South Alabama won 52-35 |
| ✅ | Nebraska defeats Maryland | 88% | Lean | Nebraska won 48-23 |
| ✅ | North Dakota State defeats Wyoming | 88% | Lean | North Dakota State won 28-0 |
| ✅ | New Mexico defeats UTEP | 86% | Lean | New Mexico won 61-7 |
| ✅ | SMU defeats Boston College | 84% | Pass | SMU won 25-16 |
| ❌ | Miami (OH) defeats Bowling Green | 84% | Pass | Bowling Green won 24-20 |
| ✅ | Notre Dame defeats North Carolina | 83% | Pass | Notre Dame won 37-26 |
| ✅ | Memphis defeats Charlotte | 82% | Pass | Memphis won 59-8 |
| ✅ | Toledo defeats Ball State | 82% | Pass | Toledo won 39-24 |
| ✅ | Texas A&M defeats Arkansas | 80% | Pass | Texas A&M won 34-7 |
| ✅ | Kansas defeats Middle Tennessee | 78% | Pass | Kansas won 55-0 |
| ✅ | James Madison defeats Marshall | 73% | Pass | James Madison won 45-17 |
| ❌ | Illinois defeats Purdue | 72% | Pass | Purdue won 24-17 |
| ✅ | Wisconsin defeats Michigan State | 71% | Pass | Wisconsin won 31-3 |
| ✅ | UNLV defeats California | 70% | Pass | UNLV won 39-31 |
| ❌ | UConn defeats Syracuse | 70% | Pass | Syracuse won 42-41 |
| ✅ | Western Michigan defeats Buffalo | 70% | Pass | Western Michigan won 20-17 |
| ✅ | Indiana defeats Rutgers | 69% | Pass | Indiana won 47-15 |
| ✅ | Louisiana defeats Arkansas State | 69% | Pass | Louisiana won 23-20 |
| ✅ | Central Michigan defeats Akron | 68% | Pass | Central Michigan won 41-17 |
| ✅ | Wake Forest defeats Stanford | 68% | Pass | Wake Forest won 57-3 |
| ❌ | Massachusetts defeats Eastern Michigan | 67% | Pass | Eastern Michigan won 38-14 |
| ✅ | Alabama defeats Mississippi State | 67% | Pass | Alabama won 56-23 |
| ✅ | Tennessee defeats Auburn | 66% | Pass | Tennessee won 24-14 |
| ✅ | Texas Tech defeats Colorado | 65% | Pass | Texas Tech won 29-7 |
| ✅ | Houston defeats UCF | 65% | Pass | Houston won 27-17 |
| ✅ | USC defeats Washington | 64% | Pass | USC won 25-21 |
| ❌ | Hawai'i defeats San José State | 64% | Pass | San José State won 20-16 |
| ✅ | Iowa State defeats West Virginia | 63% | Pass | Iowa State won 45-42 |
| ✅ | Louisiana Tech defeats Army | 63% | Pass | Louisiana Tech won 31-29 |
| ✅ | UTSA defeats Rice | 62% | Pass | UTSA won 16-14 |
| ❌ | Colorado State defeats Oregon State | 61% | Pass | Oregon State won 56-26 |
| ✅ | Arizona defeats Cincinnati | 61% | Pass | Arizona won 34-7 |
| ✅ | Fresno State defeats Washington State | 59% | Pass | Fresno State won 26-6 |
| ✅ | Georgia State defeats Old Dominion | 59% | Pass | Georgia State won 42-10 |
| ❌ | Texas State defeats San Diego State | 58% | Pass | San Diego State won 31-29 |
| ❌ | Michigan defeats Minnesota | 58% | Pass | Minnesota won 20-14 |
| ✅ | Missouri defeats Florida | 58% | Pass | Missouri won 45-17 |
| ❌ | Arizona State defeats Baylor | 58% | Pass | Baylor won 55-19 |
| ✅ | Air Force defeats Navy | 57% | Pass | Air Force won 14-9 |
| ✅ | Ohio State defeats Iowa | 57% | Pass | Ohio State won 31-14 |
| ✅ | NC State defeats Louisville | 57% | Pass | NC State won 31-28 |
| ❌ | Penn State defeats Northwestern | 56% | Pass | Northwestern won 34-13 |
| ❌ | Coastal Carolina defeats Georgia Southern | 56% | Pass | Georgia Southern won 31-24 |
| ✅ | Ohio defeats Kent State | 55% | Pass | Ohio won 13-10 |
| ❌ | Clemson defeats Miami | 55% | Pass | Miami won 41-13 |
| ❌ | South Carolina defeats Kentucky | 54% | Pass | Kentucky won 35-34 |
| ❌ | TCU defeats BYU | 54% | Pass | BYU won 17-10 |
| ✅ | Florida State defeats Virginia | 53% | Pass | Florida State won 38-7 |


---
_For entertainment and educational purposes only — not financial or betting advice._
