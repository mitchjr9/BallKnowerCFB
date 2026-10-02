# BallKnower Quad — Calibration

_Pre-registered college football forecasts, live forecasts only. Every row was committed before kickoff and graded afterward whether it worked or not._

## Win probability

| metric | value |
|---|---:|
| forecasts graded | 156 |
| hit rate | 0.795 |
| average confidence | 0.736 |
| Brier | 0.1422 |
| log loss | 0.4413 |
| calibration error (ECE) | 0.0629 |

The number that matters is the gap between the last two: we said **73.6%** on average and hit **79.5%** (+5.9%). Being right more often than we claimed is as much a calibration miss as the reverse — it just flatters us.

### By confidence tier

| data depth | tier | picks | advertised | hit rate | gap | ±1 SE |
|---|---|---:|---:|---:|---:|---:|
| none | Lean | 37 | 0.937 | 0.973 | +0.036 | 0.027 |
| none | Pass | 119 | 0.673 | 0.739 | +0.066 | 0.040 |

**Data depth** is how much football the ratings are built on. In the opening weeks a team's rating comes mostly from preseason roster signals rather than results, so the model caps how confident a label it will publish — which is why an early-season tier can advertise a probability well above its name. Depths are reported separately because pooling them would average two different things.

## Margin

| metric | value |
|---|---:|
| forecasts graded | 156 |
| CRPS | 8.909 |
| mean absolute error | 12.59 pts |
| bias (predicted − actual) | -1.58 pts |
| inside ±1σ | 0.750 (target 0.683) |
| inside ±2σ | 0.949 (target 0.954) |

Coverage well under target means the margin model is overconfident — the stated uncertainty is too narrow. Over target means it's hedging.

## Last slate (2026-09-26) — 6-3

| | pick | said | tier | result |
|---|---|---:|---|---|
| ✅ | SMU defeats Missouri State | 93% | Lean | SMU won 34-24 |
| ✅ | Fresno State defeats Rice | 89% | Lean | Fresno State won 38-24 |
| ✅ | Florida Atlantic defeats UL Monroe | 77% | Pass | Florida Atlantic won 45-17 |
| ❌ | Washington defeats Minnesota | 69% | Pass | Minnesota won 27-24 |
| ❌ | Sacramento State defeats Massachusetts | 64% | Pass | Massachusetts won 35-6 |
| ✅ | Arkansas defeats Tulsa | 63% | Pass | Arkansas won 34-6 |
| ✅ | Oregon State defeats UTEP | 54% | Pass | Oregon State won 33-7 |
| ❌ | Georgia Tech defeats Stanford | 54% | Pass | Stanford won 34-27 |
| ✅ | Air Force defeats Nevada | 51% | Pass | Air Force won 36-33 |


---
_For entertainment and educational purposes only — not financial or betting advice._
