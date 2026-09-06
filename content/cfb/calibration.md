# BallKnower Quad — Calibration

_Pre-registered college football forecasts, live forecasts only. Every row was committed before kickoff and graded afterward whether it worked or not._

## Win probability

| metric | value |
|---|---:|
| forecasts graded | 40 |
| hit rate | 0.800 |
| average confidence | 0.766 |
| Brier | 0.1391 |
| log loss | 0.4371 |
| calibration error (ECE) | 0.0695 |

The number that matters is the gap between the last two: we said **76.6%** on average and hit **80.0%** (+3.4%). Being right more often than we claimed is as much a calibration miss as the reverse — it just flatters us.

### By confidence tier

| tier | picks | advertised | hit rate | gap | ±1 SE |
|---|---:|---:|---:|---:|---:|
| Lean | 12 | 0.941 | 0.917 | -0.025 | 0.080 |
| Pass | 28 | 0.691 | 0.750 | +0.059 | 0.082 |

## Margin

| metric | value |
|---|---:|
| forecasts graded | 40 |
| CRPS | 11.318 |
| mean absolute error | 16.58 pts |
| bias (predicted − actual) | -5.98 pts |
| inside ±1σ | 0.600 (target 0.683) |
| inside ±2σ | 0.900 (target 0.954) |

Coverage well under target means the margin model is overconfident — the stated uncertainty is too narrow. Over target means it's hedging.


_6 forecast(s) still open, 0 void._


---
_For entertainment and educational purposes only — not financial or betting advice._
