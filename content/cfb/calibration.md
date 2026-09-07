# BallKnower Quad — Calibration

_Pre-registered college football forecasts, live forecasts only. Every row was committed before kickoff and graded afterward whether it worked or not._

## Win probability

| metric | value |
|---|---:|
| forecasts graded | 43 |
| hit rate | 0.814 |
| average confidence | 0.764 |
| Brier | 0.1343 |
| log loss | 0.4278 |
| calibration error (ECE) | 0.0827 |

The number that matters is the gap between the last two: we said **76.4%** on average and hit **81.4%** (+5.0%). Being right more often than we claimed is as much a calibration miss as the reverse — it just flatters us.

### By confidence tier

| data depth | tier | picks | advertised | hit rate | gap | ±1 SE |
|---|---|---:|---:|---:|---:|---:|
| none | Lean | 12 | 0.941 | 0.917 | -0.025 | 0.080 |
| none | Pass | 31 | 0.696 | 0.774 | +0.078 | 0.075 |

**Data depth** is how much football the ratings are built on. In the opening weeks a team's rating comes mostly from preseason roster signals rather than results, so the model caps how confident a label it will publish — which is why an early-season tier can advertise a probability well above its name. Depths are reported separately because pooling them would average two different things.

## Margin

| metric | value |
|---|---:|
| forecasts graded | 43 |
| CRPS | 10.974 |
| mean absolute error | 16.08 pts |
| bias (predicted − actual) | -5.46 pts |
| inside ±1σ | 0.605 (target 0.683) |
| inside ±2σ | 0.907 (target 0.954) |

Coverage well under target means the margin model is overconfident — the stated uncertainty is too narrow. Over target means it's hedging.


---
_For entertainment and educational purposes only — not financial or betting advice._
