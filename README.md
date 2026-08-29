# BallKnower Quad 🏟️ — V2

The **college-football** sibling of BallKnower **Gridiron** (NFL), **Hoops**
(NBA), **Insights** (tennis), **Pitch** (soccer) and **Links** (golf). Same
playbook: a transparent rating engine (Elo) + a calibrated probability layer
(XGBoost) + an honest walk-forward backtest harness + a newsletter-ready CLI.

V1 is built **strictly on the patterns we proved in Gridiron and the fantasy
package** — Elo with prior-season carryover, rolling efficiency, rest/schedule
context, a calibrated win-probability classifier blended with Elo, and an optional
margin model — adapted for the realities of the college game.

> **Disclaimer:** For entertainment and educational purposes only — not financial
> or betting advice.

## Folder layout (same split as Gridiron/Pitch)

```
BallKnowerCFB/                    <- project root (run python -m from HERE)
├── ballknower_quad/              <- the package
│   ├── settings.py               <- all config / hyperparameters / tiers / paths
│   ├── content_utils.py          <- newsletter engine (GamePrediction + renderers)
│   ├── data/
│   │   ├── cfbd_loader.py         <- CFBD v2 API client + cache + synthetic fallback
│   │   ├── ppa_loader.py          <- V1.1: /ppa/games + /ppa/teams
│   │   ├── preseason_loader.py    <- V2: talent/recruiting/returning/portal
│   │   └── bundle.py              <- one loader all scripts share
│   ├── models/
│   │   ├── cfb_elo.py             <- Elo: HCA, MOV damping, FCS floor, season carry
│   │   ├── ppa_ratings.py         <- V1.1: rolling opponent-adjusted PPA + anchors
│   │   ├── preseason_prior.py     <- V2: roster prior that seeds each season
│   │   ├── feature_builder.py     <- leakage-free feature matrix (version-gated)
│   │   ├── cfb_model_v1.py        <- win prob: calibrated XGB classifier + Elo blend
│   │   └── cfb_margin_v1.py       <- margin: XGB regressor (+ ATS read)
│   ├── ledger/
│   │   ├── emit_records.py    <- forecast record model + invariants + tier fn
│   │   ├── emit.py            <- predictions.json -> forecasts.jsonl + manifest
│   │   └── selftest.py        <- 24 invariant tests (incl. ones that must fail)
│   ├── backtest/
│   │   ├── evaluate.py            <- walk-forward; acc / Brier / log-loss / ECE / ATS
│   │   └── leakage_audit.py       <- truncation-invariance proof (no future peeking)
│   └── scripts/
│       ├── check_api.py          <- verify CFBD key + remaining monthly quota
│       ├── train.py              <- fit & persist (run this first)
│       ├── predict.py            <- single matchup / ATS
│       ├── calibrate_tiers.py    <- derive tier thresholds from real data
│       ├── compare_versions.py   <- paired ablation + SHIP/WATCH/KILL verdict
│       ├── repro_check.py        <- proves the pipeline is deterministic
│       ├── diagnose.py           <- FCS contamination, PPA-vs-Elo, redundancy
│       └── weekly_pipeline.py    <- pull slate → md/html/json newsletter
├── data_cache/                   <- cfbd/ cache, parquet
├── models_store/                 <- v1/ (model files), elo_state.json
├── content/cfb/<date>/           <- generated newsletters
├── requirements.txt
└── README.md
```

## Quickstart

```bash
pip install -r requirements.txt

# Optional: real data. Get a free key (1,000 calls/month) at
# https://collegefootballdata.com/key , then:
cp .env.example .env && echo "CFBD_API_KEY=your_key" >> .env
# Without a key it runs in deterministic SYNTHETIC mode — everything works offline.

python -m ballknower_quad.scripts.check_api                   # verify key + quota
python -m ballknower_quad.scripts.train --version v1.1        # fit V1.1 + margin
python -m ballknower_quad.scripts.predict "Ohio State" "Michigan"
python -m ballknower_quad.scripts.predict Georgia Texas --neutral --spread -3.5
python -m ballknower_quad.scripts.weekly_pipeline             # write newsletter
python -m ballknower_quad.backtest.evaluate --with-margin --version v1.1
python -m ballknower_quad.backtest.leakage_audit --version v1.1   # prove no peeking
python -m ballknower_quad.scripts.compare_versions                # V1 vs V1.1 verdict
```

## The data

Backbone: the **CollegeFootballData v2 API** (`api.collegefootballdata.com`).
The `/games` endpoint is the spine — it carries both training history *and* the
upcoming slate (unplayed games return with null points, exactly how Pitch used the
martj42 fixtures). `/calendar` gives week→date mapping for rest features.

Two things changed since the fantasy package used CFBD, and the loader handles
both: **v1 was retired** before the 2025 season (since May 2025 both `api.*` and
`apinext.*` point to v2, with `api.*` canonical), and **every call needs a Bearer
token** — free tier is **1,000 calls/month**, shared across CFB *and* college
basketball on the same account. So the loader caches every response to disk and
pulls whole-season endpoints (one call per season). No key → synthetic mode.

Your existing CFBD key still works — keys are account-level and weren't
invalidated by the v1→v2 move. Verify in one command:

```bash
python -m ballknower_quad.scripts.check_api
```

## How V1 works

1. **Elo** (`cfb_elo.py`) — base 1500, home-field ≈ 65 (neutral-aware), a 538-style
   margin-of-victory multiplier (capped) so beating a cupcake 56–0 doesn't
   over-credit, an FCS floor at 1200, and prior-season regression toward the mean
   (carry 0.66) since seasons are short.
2. **Features** (`feature_builder.py`) — leakage-free, recorded *as of kickoff*:
   Elo gap (±HCA), rolling scoring-margin efficiency (ridge-shrunk, season-carried),
   rest-day diff + short-week flag, neutral and conference flags, week number.
3. **Win probability** (`cfb_model_v1.py`) — XGBoost classifier, isotonic-calibrated
   on a time-ordered tail, then **blended with the Elo baseline** (weight 0.30) the
   way Gridiron blended Elo and the NFL build blended Vegas.
4. **Margin** (`cfb_margin_v1.py`) — XGB regressor → home−away points; with a line
   it gives an ATS read. CFBD `/lines` uses the **same sign convention as nflverse**
   (positive = home favored), so the cover signal is `predicted_margin − line`
   (the sign fix from the Gridiron leakage audit, carried forward).
5. **Backtest** (`backtest/evaluate.py`) — walk-forward, fit on `< S`, predict `S`.

## A note on metrics (read before you trust a number)

CFB **accuracy looks high** — talent gaps and cupcake games mean favorites win a
lot — so accuracy alone is a vanity metric. The honest read is **Brier + ECE +
the reliability table**. On real data expect roughly **72–76% accuracy** with a
**well-calibrated** probability; the synthetic-mode numbers run higher only because
the toy data has cleaner, larger talent gaps than reality. Against real Vegas lines,
expect ATS near **50%** (no edge) — same honest result we landed on for Gridiron V5.
Treat margin/ATS output as *interesting reads*, never bets.

## A note on tiers

CFB favorites are more dominant than NFL ones, so the win-prob distribution is
right-shifted and the Lock/Strong/Lean/Pass thresholds in `settings.py` are set a
notch higher than Gridiron's. After your first real backtest, recalibrate them off
the reliability table — same way we tuned Gridiron's tiers.

## V1.1 — opponent-adjusted PPA

PPA is CFBD's expected-points metric, the direct analog of the nflverse EPA that
drove most of the gain in Gridiron's NFL V3. Four features are added, gated by
version so V1 stays bit-identical for a clean ablation:
`off_ppa_diff`, `def_ppa_diff`, `net_ppa_diff`, `mean_net_ppa`.

Three design points that matter more in college than in the NFL:

- **Opponent adjustment is done here, not by CFBD.** `/ppa/games` returns raw
  per-game PPA. With CFB's wildly unbalanced schedules, raw PPA largely measures
  *who you played*. Each game is adjusted against the opponent's rating as of
  that game — the same idea as the golf model's field adjustment.
- **Per-game, never season-aggregate, within a season.** A season aggregate
  contains the game you're predicting. `/ppa/teams` is used *only* for prior
  seasons, as the Week 1 anchor.
- **Prior-season anchor with a games-played ramp** (`w = n/(n+k)`), which fixes
  the exact hole that made Gridiron's V2 QB rating so noisy in September.

Verify it yourself: `leakage_audit` rebuilds features with future games removed
and asserts every row is identical.

## Known gotcha: non-FBS opponents

CFBD returns FCS teams with their **real** conference (Big Sky, MVFC, ASUN) or
`null` — never the literal string `"FCS"`. Detect them via
`homeClassification` / `awayClassification` (`"fbs"` vs `"fcs"`), which the
loader now captures and normalizes. If you skip this, North Dakota State gets a
full 1500-base Elo rating and your Week 1 slate will show an FBS host as a
coin flip against an FCS visitor.

If you pulled games before this fix, **delete `data_cache/cfbd/` and re-pull** —
the cached JSON predates the classification capture.

```bash
rm -rf data_cache/cfbd && python -m ballknower_quad.scripts.diagnose --version v1.1
```

## V1.2 — splitting neutral-site from postseason

`is_neutral` alone conflated two structurally different game types. In 11 seasons
of real data, **62% of neutral-site games are bowls** (opt-outs, month-long
layoffs, flat motivation) and only **12% are Week 1–2 kickoff classics** (full
rosters, peak intensity). With a single flag the model learns bowl behaviour from
the majority and applies it to season openers — which is how Ohio State came out
a 48.8% underdog to Texas on a neutral field despite a 39-point Elo edge.

V1.2 adds one feature, `is_postseason`, letting the tree form the interaction
itself. One feature, not five — the V4 weather lesson.

## Reading the PPA leaderboard correctly

`get()` returns the prior-season anchor when a team has no current-season games,
so **a leaderboard printed in August is CFBD's own `/ppa/teams` output**, not this
engine's in-season adjustment. `top()` now shows `games` and `anchor_w` so you can
see how much of a rating is last year's number. To judge the adjustment, look
mid-season:

```bash
python -m ballknower_quad.scripts.diagnose --version v1.2 --through-week 10
```

## Ablating the opponent adjustment

The adjustment is the least-proven part of the PPA block. Test it in one command
(no settings edit, no extra API calls):

```bash
python -m ballknower_quad.scripts.compare_versions --a v1.1 --b v1.1 --b-ppa-strength 0.0
```

If raw PPA (strength 0) beats adjusted PPA, the adjustment is hurting.

## Version status

| version | features | verdict |
|---|---|---|
| v1 | Elo + rolling margin + schedule | baseline |
| v1.1 | + opponent-adjusted PPA | superseded by v2 |
| v1.2 | + `is_postseason` | **KILLED** — worse on the full slate twice |
| **v2** | **v1.1 + preseason roster prior** | **PRODUCTION** — full slate t=2.58 |

Production config: `v2`, `fcs=pooled`, `ppa_fcs_offset=None` (auto),
`anchor=cfbd`, `portal=in` — accuracy 0.7419, Brier 0.1704, ECE 0.0155 over
2019–2025 (n=6,002), against a 0.6275 always-pick-home baseline. v2 beats v1.1
on the full slate at t=2.58 (ΔBrier +0.00233).

The v2 gain is NOT confined to the cold-start window it was designed for
(+0.00273 in weeks 1-4, +0.00210 implied for weeks 5+). The prior seeds Elo at
the season boundary and Elo carries that starting point forward, so a better
Week 1 rating improves the whole season.

## V2 — the preseason cold start

Weeks 1–4 are the model's weakest stretch and the part the newsletter covers
while readers are most engaged. Elo enters a season as nothing but last year
regressed toward 1500, so a team that returned two starters and a team that
returned nineteen look identical apart from last year's record.

V2 builds a composite roster prior from four CFBD feeds — `/talent`,
`/recruiting/teams` (trailing 4 years), `/player/returning`, `/player/portal` —
standardizes each **within its own season**, maps the composite onto the Elo
scale, and **blends** it into the season carryover:

    start = (1 - blend) * regressed_carryover + blend * roster_prior

Blended, not added: recruiting and last year's Elo both measure program
strength, and summing them double-counts it. The prior's influence then decays
on its own as real results accumulate — no hand-tuned schedule.

Three features ride along for the model to use directly:
`preseason_prior_diff`, `returning_diff`, `portal_net_diff`.

**Judge V2 on the window it targets**, not the full slate:

```bash
python -m ballknower_quad.scripts.compare_versions --a v1.1 --b v2 --subset early
```

Ablate the mechanism itself (features stay, Elo seeding switches off):

```bash
python -m ballknower_quad.scripts.compare_versions --a v2 --b v2 --b-preseason-blend 0.0
```

## Add tests vs. knockout tests

`compare_versions` answers two questions that read in **opposite directions**:

* **Add test** (`--a v1.1 --b v2`) — challenger is a new version. Positive ΔBrier
  means ship it.
* **Knockout test** (`--a v2 --b v2 --b-preseason-blend 0.0`) — challenger is the
  *same* version with a mechanism switched **off**. A negative ΔBrier means
  removing it hurt, i.e. **keep** the mechanism.

The verdict line adapts to which one you ran, so a knockout says KEEP/REMOVE
rather than SHIP/KILL. A "KILL" on a knockout test would mean the opposite of
what you want.

## Evaluating subpopulation features

`is_postseason` can only move ~7% of the slate. A whole-slate paired test
dilutes that effect roughly **14x** and will read as noise no matter how well the
feature works. Test the games the feature can actually touch:

```bash
python -m ballknower_quad.scripts.compare_versions --a v1.1 --b v1.2 --subset neutral
```

Subsets: `neutral`, `postseason`, `regular`, `early` (wk≤3), `late` (wk≥10),
`close` (win prob within 15pts of a coin flip).

## Reading the Brier decomposition

`compare_versions` prints Murphy's decomposition:

    Brier = Reliability - Resolution + Uncertainty

* **Reliability** — calibration error (what ECE tracks). Lower is better.
* **Resolution** — how far predictions move from the base rate, i.e. sharpness.
  **Higher** is better.
* **Uncertainty** — base-rate variance, fixed by the data.

This is how a version can improve ECE while *regressing* on Brier: it got more
honest but less decisive, and Brier charges for both. A newsletter arguably cares
more about reliability than Brier does — but a model that hedges everything
toward 50% has perfect calibration and zero usefulness, so don't optimize
reliability alone.

## FBS-vs-FCS games and the PPA ratings

CFBD's `/ppa/games` returns **only the FBS side** of an FBS-vs-FCS matchup — one
row instead of two (about 1,242 missing rows across 11 seasons). Three ways to
handle it, all selectable via `settings.ppa_fcs_mode`:

| mode | what it does | trade-off |
|---|---|---|
| `naive` | treat the FCS side as an average FBS team | free efficiency bonus; skews league means |
| `skip` | drop those games entirely | unbiased, but discards ~13% of the sample |
| `pooled` | keep the game, rate the FCS side at a floor | **default** — keeps the data, removes the bias |

`pooled` is the PPA analog of the Elo FCS floor. `ppa_fcs_offset = None`
(the default) estimates the floor from the data: the gap between one-sided
(cupcake) rows and two-sided rows. The offensive and defensive views give
independent reads on the same quantity and agree to ~0.001, which is a good sign
the estimate is real. Pin a number only if you want to ablate it. Ablate any of these:

```bash
python -m ballknower_quad.scripts.compare_versions --a v1.1 --b v1.1 --b-fcs-mode skip
```

## Recalibrating tiers after promoting a model

Thresholds do not transfer between model versions — promoting one reshapes the
probability distribution underneath the labels. That's how "Lock" ended up firing
on 45% of the slate without anyone touching a threshold.

Don't hand-tune them. Derive them:

```bash
python -m ballknower_quad.scripts.calibrate_tiers                  # quantile
python -m ballknower_quad.scripts.calibrate_tiers --auto           # constrained
python -m ballknower_quad.scripts.calibrate_tiers --keep-current   # just validate
python -m ballknower_quad.scripts.calibrate_tiers --auto --write   # apply
```

**Quantile mode** sets thresholds so each tier captures a target share of the
slate. It assumes calibration is uniform across the probability range — which it
isn't. This model is mildly overconfident in the tail, so pushing the Lock cut
higher selects a *more* overconfident subpopulation and makes the gap worse.

**`--auto`** searches for the most scarce thresholds that still pass the honesty
check, rather than picking by quantile and hoping. Use it whenever quantile mode
fails.

**`--keep-current`** validates the config you already ship without proposing a
change — the right call when the shipped thresholds already pass.

Either way it then checks the thing that actually matters: **does each tier hit
at or above the confidence it advertises?** A tier fails only when it is both
*materially* overconfident (more than `--tolerance`, default 2pp) **and**
*significantly* so (worse than 2 standard errors). Both conditions matter: a
285-pick tier at 94% carries an SE near 0.014, so a −0.027 gap is under 2 SE and
could be noise — failing on that alone would mean retuning thresholds every week
in response to variance. `--write` refuses to apply a config that fails.

The report shows `share_pre_cap` (by probability alone) next to `share` (after
the ledger's data-depth cap), because early-season games get their tier capped —
so raw thresholds overstate how many Locks you actually publish in September.

`tier_config_version` is a hash of the thresholds, so it changes automatically
when they do. Rows written before and after a recalibration stay distinguishable
in the ledger — which is what keeps a tier-level reliability curve that spans the
change from being quietly meaningless.

## Publishing to GitHub

`.gitignore` keeps `.env` (your API key), `venv/`, `data_cache/` and
`models_store/` out. It deliberately does **not** ignore `content/cfb/` — that's
the ledger, and GitHub's commit timestamp is what makes a forecast verifiable as
having existed before kickoff.

```bash
cd ~/Downloads/BallKnowerCFB
git init
git add .gitignore && git commit -m "Add gitignore"   # secrets excluded FIRST
git status --short | grep -E "\.env$|venv/" && echo "STOP — secrets staged"
git add . && git commit -m "BallKnower Quad v2"
git branch -M main
git remote add origin https://github.com/mitchjr9/BallKnowerCFB.git
git push -u origin main
```

Commit `.gitignore` before anything else. Once a key is in git history, removing
it needs a history rewrite and the key should be rotated regardless.

## Weekly run

```bash
cd ~/Downloads/BallKnowerCFB && source venv/bin/activate

python -m ballknower_quad.scripts.train              # refresh ratings (v2)
python -m ballknower_quad.scripts.weekly_pipeline    # slate -> content
python -m ballknower_quad.ledger.emit                # pre-register forecasts
git add content/cfb && git commit -m "ledger: cfb $(date +%F)" && git push
```

Outputs land in `content/cfb/<date>/`:

| file | what it is |
|---|---|
| `newsletter.md` / `.html` | scannable briefing — tiers, top picks, full slate table |
| `blog.md` / `.html` | narrative post with the reasoning behind each call |
| `predictions.json` | structured slate, stamped with `asof_ts` — ledger input |
| `ledger/forecasts.jsonl` | one row per resolvable proposition |
| `ledger/manifest.json` | chain head + counts — commit this for attestation |

Run it **early in the week**. The emitter refuses any live row whose
`committed_ts` is after kickoff, which is the point — a forecast published after
the game starts isn't a forecast.

## The forecast ledger

Each game emits **two** records, because they resolve independently:

* `moneyline` — probabilistic, scored by Brier / log-loss
* `spread` — distributional (`mu` = predicted margin, `sigma` = margin-model
  RMSE), scored by CRPS and interval coverage

Two invariants, deliberately separated:

* **Leakage** (`asof_ts <= event_start_ts`) applies to every row, always.
* **Pre-registration** (`asof_ts <= committed_ts <= event_start_ts`) applies only
  to `provenance='live'`. A backfilled backtest row is legitimately committed
  today about a 2019 game — that's what `provenance` is for, and why public
  calibration surfaces filter to live only.

Tier is a pure function of probability **and data depth**, versioned as
`quad-tier-v1`. Depth is Quad-specific and load-bearing: in Week 1 a rating is
mostly preseason prior, so thin data caps the tier. A Week 1 "Lock" would be
advertising confidence the model hasn't earned.

```bash
python -m ballknower_quad.ledger.selftest    # 24 invariant tests
```

## Reproducibility

Effects at this stage are ~0.0005 Brier. Before trusting any of them, confirm the
pipeline itself doesn't wobble by that much:

```bash
python -m ballknower_quad.scripts.repro_check --version v1.2
```

It runs the same walk-forward twice and asserts identical output. If it ever
reports drift, pin `n_jobs=1` in `settings.xgb_clf_params`; until then, treat any
movement in a baseline between sessions as a **real config change worth
explaining**, not as noise.

## Promoting a version

Never promote on a single number. Run the paired ablation:

```bash
python -m ballknower_quad.scripts.compare_versions --a v1 --b v1.1
```

Kill criteria, declared before looking: **SHIP** if ΔBrier improvement > 2 SE,
**WATCH** if positive but inside 2 SE, **KILL** if ≤ 0. A tie is a kill — spare
features are just overfitting room. If it ships, set `active_model_version` in
`settings.py` to `v1.1`.

## V2 roadmap (see chat notes)

Recruiting class rankings (`/recruiting/teams`), returning production
(`/player/returning`), transfer portal (`/player/portal`), SP+/FPI consumed as
priors (`/ratings/sp`, `/ratings/fpi`), and weather (`/games/weather`, tier-gated).
NIL is deliberately *not* on this list as a model feature — it's estimated rather
than disclosed and collinear with recruiting; treat it as newsletter content. All are CFBD endpoints or free
public datasets; the design keeps them as **lightly-blended orthogonal priors**,
not core features, for the same reason Klement and Vegas were blended lightly.
