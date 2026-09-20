# Predicting overtakes in Formula One's 2026 regulation era

**Applied Machine Learning (Advanced) — project report**

---

## 1. Goal

Given two Formula One cars fighting on track, estimate the probability that the
car behind completes a pass, and keep that estimate honest as a brand-new
regulation era accumulates data one race at a time.

Three things make this harder than it sounds, and they shaped every decision
in the project:

1. **The mechanism of overtaking changed in 2026.** DRS was removed and
   replaced by an energy-based Overtake Mode, active aerodynamics were
   introduced, and the power unit moved to a 50/50 combustion-electric split.
   A model trained on 2022-2025 does not transfer as-is.
2. **There is almost no in-era data.** Fourteen races existed when this was
   built, yielding **470 passes** across 6,669 usable observations.
3. **The era is ongoing.** Every fortnight adds a race, so the interesting
   question is not "what is the best model on a fixed dataset" but "how should a
   model be built, evaluated and updated when the data keeps arriving".

The third point is why the project is structured around a walk-forward
evaluation and a retraining loop rather than a single train/test split.

## 2. Materials

| What | Where |
|---|---|
| Report | this file |
| Code | [github.com/AminN77/formula1-overtake-prediction](https://github.com/AminN77/formula1-overtake-prediction) — package `era2026/`, 88 tests |
| Design record | `era2026/DESIGN.md`, `era2026/PLAN.md` |
| Data reconnaissance | `era2026/recon_report.md` |
| Interactive results | `era2026/ui/index.html` (open directly, no server) |
| Prior project, frozen | `legacy/` at tag `legacy-v6` |

Everything is reproducible from `era2026/` with `uv sync` and the commands in
section 15.

## 3. Data

### 3.1 Source

All data comes from **FastF1**, which wraps Formula One's live timing service.
Only three things are read: lap records, weather, and race control messages.
Car telemetry is never needed, because the speed-trap measurements the features
use (`SpeedI1`, `SpeedI2`, `SpeedFL`, `SpeedST`) are lap-level columns.

A reconnaissance pass (`era2026/recon.py`) established what 2026 actually
exposes before any modelling decision was taken. Findings:

- All 14 completed rounds load cleanly: **16,255 lap records**, 22 drivers, 11
  teams.
- The lap schema is **identical to 2024**: 31 columns, nothing added or removed.
- **DRS is dead, not merely deprecated.** The telemetry channel still exists but
  all 35,553 samples read zero, against a live distribution in a 2024 reference
  race. This is measured, not assumed.
- **There is no telemetry channel for Overtake Mode, active aero or energy
  deployment.** The 2026 car data carries the same ten channels as 2024.
- Overtake Mode does surface in race control as lap-stamped
  `OVERTAKE ENABLED` / `OVERTAKE DISABLED` messages.

### 3.2 Defining an overtake

An overtake is detected as a **pairwise order flip**: driver A is behind driver
B at the end of lap L and ahead of them at the end of lap L+1. This is the only
definition available from timing data that says nothing about DRS, aero or
energy, which is precisely why it survives the regulation change.

Raw flips massively overcount, because pit stops, retirements and slow punctures
all reorder the field without anyone passing on track. Each flip therefore
carries metadata and pit-involved flips are excluded. Lap one, safety-car periods
and inaccurate timing are **kept**, because they contain real passes and
filtering them costs more recall than it buys precision.

**Validation against published figures.** This is the checkpoint that decides
whether anything downstream is trustworthy:

| Circuit | Published 2026 | This extraction |
|---|---|---|
| Australian | 39 | 37 |
| Chinese | 71 | 73 |
| Japanese | 43 | 42 |
| Miami | 50 | 58 |
| **Total** | **203** | **210** (+3.4%) |

The same detector applied to the legacy 2025 data gives 77 against a published
84. So the method lands within roughly 10% of published totals **in both eras**,
which is what the transfer strategy in section 7 assumes.

### 3.3 Reconstructing Overtake Mode

Worth describing because the obvious approach is wrong. Race control announces
an explicit `OVERTAKE DISABLED` only at the standing start. Later
neutralisations disable the mode *silently*, and the re-enable is announced
**inconsistently**: Canada takes VSCs on laps 31, 46 and 53 and logs
`OVERTAKE ENABLED` after each; Australia takes VSCs on laps 12, 18 and 34 and
logs none at all. Replaying messages alone leaves Australia switched off for its
final 46 laps.

The rule that survives all 14 rounds uses track status as the primary signal
with messages as explicit overrides. Result: the mode is available on **727 of
855 race laps (85.0%)**.

One trap: `LAPPED CARS MAY NOW OVERTAKE THE SAFETY CAR` contains the word
OVERTAKE and is not a toggle. Messages are matched exactly.

## 4. Exploratory data analysis

### 4.1 Overtaking increased sharply, and unevenly

On the 13 circuits raced in both seasons, passes rose **+72% against 2025** and
+42% against the 2022-2025 mean. But the change is not a uniform multiplier:

| Circuit | 2022-25 mean | 2026 | ratio |
|---|---|---|---|
| Italian (Monza) | 31.2 | 97 | **3.10x** |
| Chinese | 32.0 | 73 | 2.28x |
| Japanese | 22.0 | 42 | 1.91x |
| Miami | 36.2 | 58 | 1.60x |
| Monaco | 8.5 | 12 | 1.41x |
| Canadian | 36.0 | 35 | 0.97x |
| Austrian | 37.5 | 31 | 0.83x |
| Barcelona | 42.5 | 24 | **0.56x** |

Monza more than tripled while Barcelona *fell by nearly half*. This single table
drove a major design decision (section 7.2).

### 4.2 Passes happen from very close range

Across 475 passes with a measurable prior-lap gap:

| | gap at the lap before the pass |
|---|---|
| median | **0.43 s** |
| 90th percentile | 0.99 s |
| 99th percentile | 3.01 s |

This fixed the candidate threshold empirically rather than by inheritance:

| threshold | passes captured | candidate rows |
|---|---|---|
| 1.0 s | 90.1% | 3,293 |
| 2.0 s | 96.4% | 5,909 |
| **3.0 s** | **98.9%** | **7,491** |
| 4.0 s | 99.2% | 8,652 |

3.0 s captures 98.9% and the marginal return collapses past it.

### 4.3 The target is rare and varies tenfold by circuit

The base rate is **7.05%**, but per race it swings from **1.1% at Monaco** (7
passes from 109 battle episodes) to **11.8% at Monza** (69 from 217). Any
evaluation that reports a single race's score is measuring the circuit as much
as the model.

### 4.4 Univariate signal

Pass rate between the lowest and highest third of each feature:

| feature | low → high |
|---|---|
| `gap_ahead` | 18.30% → 0.81% |
| `attacker_on_newer_stint` | 5.19% → 21.50% |
| `gap_min_3` | 16.87% → 1.26% |
| `speed_fl_delta` | 3.55% → 15.07% |
| `closing_rate` | 1.44% → 10.84% |
| `team_pace_rank_delta` | 12.14% → 3.37% |

Every direction is physically sensible, which is the first evidence the dataset
is learnable rather than noise.

## 5. Problem formulation: discrete-time hazard

The obvious formulation — "given this battle, will a pass happen within three
laps?" — has three defects, all of which the previous project suffered from:

1. **Overlapping labels.** A pair battling on laps 12, 13 and 14 with a pass on
   15 produces three rows, all labelled 1, with nearly identical features. The
   model counts three examples and receives about one example of information.
2. **Deleted information.** If the defender pits, the pass *cannot* happen. The
   previous pipeline deleted those rows, discarding the genuine evidence that
   the pair fought for two laps without a pass.
3. **A fixed horizon.** Three separate label columns for three horizons.

This project instead models the **per-lap hazard**:

```
h(t) = P(pass on lap t+1 | the pair is still battling at lap t)
```

Any horizon follows by composition, `P(pass within k) = 1 - prod(1 - h)`, so the
horizon becomes a query parameter rather than a training decision.

Each row's outcome is one of three things:

- **event** — a filtered on-track pass for this pair between L and L+1;
- **survived** — no pass, *including* the attacker dropping out of range, which
  is a genuine failure to pass rather than a missing observation;
- **censored** — the opportunity was removed rather than declined (either car
  pitted, either car retired, or the race ended). These rows carry no label and
  are excluded from training, but every earlier lap of the episode is kept.

**Resulting dataset:** 860 episodes, 7,491 rows, **6,669 trainable**, 470
events, 7.05% positive. Mean episode length 8.71 laps, longest 75. Censoring
retains **706 pit-affected rows** that the naive formulation would have deleted.

## 6. Evaluation protocol

Fixed **before** any model was trained, because the previous project's headline
number came from a holdout that had been tuned against six times.

**Expanding-origin backtest.** For each round k from 5 to 14: train on rounds
1..k-1, predict round k, advance. Every round is an out-of-sample test exactly
once, so 14 rounds yield **10 test results** rather than one. A spy model in the
test suite verifies that round k never appears in its own training set.

**Hyperparameters are frozen.** A 40-trial random search ran on rounds 1-4 only
and the winner was written to `frozen_params.json`. A configuration chosen by
looking at round k would make round k no longer out of sample.

**Metrics.** PR-AUC as primary (the target is rare, so ROC-AUC flatters),
plus ROC-AUC, Brier and expected calibration error. Calibration matters
independently because the output is displayed as a probability.

**The noise floor.** Per-round PR-AUC has a standard deviation of **0.22**. This
number governs how every later result should be read: a difference of 0.02
between two models is not a difference.

## 7. Features and the transfer strategy

### 7.1 The governing idea

> **Levels come from 2026. Shapes come from the old era.**

The overtake rate is directly measurable from 2026 alone and belongs in the
intercept. What the old era can supply is the *conditional* structure: given a
battle, what makes a pass likely. Pooling the two eras naively violates this,
because one shared intercept across two base rates makes the model explain the
difference using old-era car performance.

Features are therefore tagged by transfer tier:

- **Tier 1 (regulation-invariant, 21 features)** — race state, weather,
  positions, Overtake Mode state, and cross-race form computed from prior rounds
  only (team pace ranks, teammate-relative pace, overtake and hold rates).
- **Tier 2 (same meaning, shifted relationship, 31 features)** — gap and gap
  dynamics, pace and speed-trap *deltas*, tyre and stint state, surrounding
  traffic.
- **Tier 3 (dropped)** — all DRS features, all *absolute* car speeds and lap
  times, and team identity. A test enforces their absence.

Team identity is dropped for a concrete reason: 2026 has 11 teams, of which
Audi and Cadillac have no history at all, and the other nine carry a 2022-2025
performance level that is no longer true.

### 7.2 A design idea that failed its own validation

The original plan collapsed `track` into a **historical passability prior** —
a per-circuit overtake rate fitted on 2022-2025 — on the argument that the level
shifts but the ranking is geometry and geometry does not change. Measured
against the 13 comparable circuits:

| | |
|---|---|
| Spearman rank correlation | **rho = +0.09** (p = 0.76, n = 13) |
| per-circuit ratio spread | 0.56x to 3.10x |

No detectable rank signal, and the spread is not scatter around a common
multiplier. The idea was dropped.

The mechanism explains the failure. Under DRS, passability tracked zone length
and slipstream. Overtake Mode is an *energy* aid, so what matters is
full-throttle time. Monza is the extreme case of full-throttle time and shows
the largest gain; Barcelona is aero-limited and fell. The geometry story is not
wrong — the *relevant* geometry changed.

### 7.2b Circuit physics, validated before use

The replacement transfers **physics** rather than **outcome**. A historical
overtake rate encodes what happened under a ruleset that no longer exists; a
speed trap encodes the shape of a track the cars still drive. That distinction
is testable, and it was tested before being used. Across the 13 circuits raced
in both eras:

| quantity | Spearman rho | p |
|---|---|---|
| mid-sector speed trap (i2) | **+0.974** | < 0.0001 |
| straight dominance (top / i2) | **+0.967** | < 0.0001 |
| sector-1 speed trap (i1) | **+0.966** | < 0.0001 |
| race distance | +0.999 | < 0.0001 |
| **top speed (SpeedST)** | **+0.461** | 0.11 |

Circuit physics transfers almost perfectly, with one exception: raw top speed,
which is precisely the property the new power unit and active aero changed. It
is excluded; the *ratio* is kept because the ratio transfers.

Three features result (`circuit_speed_i1`, `circuit_speed_i2`,
`circuit_straight_dominance`), plus a `circuit_known` flag, since Madrid is new
and receives the cross-circuit median — what a production system must do for any
new venue.

**Effect: +1.20% PR-AUC (0.4836 → 0.4894), and Monaco's PR-AUC rises from 0.096
to 0.148.** Helpful, and honestly short of a fix: the bias at Monaco (+4.9 pp)
and Monza (-5.6 pp) barely moves. Speed-trap physics captures how *fast* a
circuit is, not how *passable* it is. Monaco's problem is that it is too narrow
to pass on at any speed, and track width, braking-zone count and corner-exit
geometry are not derivable from timing data at all. Closing that gap needs an
external geometry source, which is the clearest single piece of future work.

### 7.3 The transfer mechanism that worked

The pre-2026 era enters the model as **one frozen scalar per row**:
`old_era_score`, the output of a LightGBM fitted once on 48,932 legacy rows
across 2022-2025, reading the 43 features both eras share and targeting
`overtake_next_lap` to match the in-era hazard question. It never sees a 2026
row, so it cannot leak into any fold and needs no refitting.

Eight seasons of data therefore reach the model as a single column. Switch it
off and the comparison is exact.

## 8. Models compared

| Model | Rationale |
|---|---|
| Base rate | The non-ML baseline. Predicts the training positive rate for everything. |
| Gap alone | Logistic regression on one feature. Tests how much the other 59 are worth. |
| Logistic, 5 features | A simple, interpretable linear model. |
| Logistic, all features | Regularised linear on the full set. |
| Random forest | Bagged ensemble: variance reduction by averaging independent fits. |
| LightGBM | Boosted ensemble: bias reduction by fitting residuals sequentially. |
| SVM (RBF) | A max-margin method with a genuinely different inductive bias. |
| MLP | Feed-forward network on the same tabular features. |
| GRU | Recurrent model over the episode sequence (section 11). |

Gradient boosting uses monotone constraints encoding known physics (closer gap
raises the hazard, closing raises it, fresher tyres raise it).

## 9. Results

Pooled over the 10 backtest rounds, 4,680 rows, 308 events, 6.58% base rate:

| Model | PR-AUC | lift | ROC-AUC | Brier |
|---|---|---|---|---|
| Base rate | 0.057 | 0.87x | 0.440 | 0.0617 |
| Gap alone | 0.323 | 4.90x | 0.851 | 0.0514 |
| Logistic, 5 | 0.370 | 5.62x | 0.874 | 0.0495 |
| MLP | 0.389 | 5.92x | 0.819 | 0.0500 |
| SVM (RBF) | 0.400 | 6.07x | 0.851 | 0.0554 |
| Logistic, all | 0.450 | 6.83x | 0.858 | 0.0468 |
| GRU (pretrained) | 0.453 | 6.89x | 0.876 | 0.1018 |
| Random forest | 0.493 | 7.49x | 0.906 | 0.0457 |
| **LightGBM** | **0.515** | **7.82x** | 0.909 | 0.0448 |
| **LightGBM + isotonic (shipped)** | 0.502 | 7.62x | **0.911** | **0.0438** |

**Reading this honestly.** Against a per-round standard deviation of 0.22, the
tree ensembles are clearly ahead of everything else, but random forest versus
LightGBM is **not resolvable**. On the full feature set RF led 0.506 to 0.484
while winning only 6 of 10 rounds; on the final feature set the order reverses.
That instability is itself the finding: the family matters, the member does not.

The shipped model is LightGBM plus isotonic calibration. It gives up a little
PR-AUC and takes the best ROC-AUC, the best Brier and a threefold lower
calibration error, which is the right trade for a model whose output is
displayed as a probability.

### 9.1 The transfer layer

| | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| In-era features only | 0.470 | 0.901 | 0.0469 |
| **plus `old_era_score`** | **0.490** | **0.907** | **0.0458** |

**+4.22%, better in 9 of 10 rounds.** With this much per-round noise the
consistency matters more than the size: 9 of 10 would occur about 1% of the time
by chance.

The result that matters is not the magnitude. The old-era model is *weaker on
its own than a single in-era feature* (ROC 0.768 against 0.862 for raw gap), yet
it still improves the full model. It is not re-supplying gap. It carries
structure the in-era model cannot recover from fourteen races.

### 9.2 Calibration

| | Brier | ECE |
|---|---|---|
| Raw | 0.0458 | 0.0142 |
| **Isotonic** | **0.0440** | **0.0049** |
| Platt (sigmoid) | 0.0458 | 0.0291 |

Isotonic cuts calibration error threefold. Platt is *worse than no calibration*
and is not used. The raw model is well behaved everywhere except the top decile,
where it predicts 0.478 against an observed 0.402 — it overstates exactly the
battles a user would look at.

### 9.3 Uncertainty

A race-level bootstrap (resampling whole races, since rows within a race are not
independent) gives an interval around each hazard. Raw width is confounded by
the test round's own hazard level, so width is normalised by it:

```
relative width vs training rounds:  Spearman rho = -0.624  (p = 0.054)
first half 1.390 → second half 1.121   (-19.3%)
```

Uncertainty narrows as the era accumulates data. This is the only support the
project's premise currently has, and it is carefully **not** an accuracy claim —
see section 12.

## 10. Explainability

Permutation importance measured on held-out rounds (PR-AUC drop when a feature
is shuffled), cross-checked with TreeSHAP:

| feature | perm. importance | mean abs SHAP | direction |
|---|---|---|---|
| `gap_ahead` | 0.070 | 0.648 | -0.79 |
| **`old_era_score`** | **0.033** | **0.388** | +0.61 |
| `gap_pressure_ratio` | 0.020 | 0.123 | +0.50 |
| `team_pace_rank_delta` | 0.019 | 0.288 | -0.94 |
| `pace_delta_mean_3` | 0.014 | 0.256 | -0.56 |
| `speed_fl_delta` | 0.012 | 0.200 | +0.78 |

Every direction is physically correct: a bigger gap lowers the hazard, closing
raises it, a faster attacking team raises it, a higher finish-line speed raises
it.

The transfer scalar is the **second most important feature by both methods**,
which is independent confirmation of the +4.22% the backtest measured.

### 10.1 The feature set is over-engineered

**40 of 60 features have zero or negative permutation importance.** Cutting the
set *improves* the model:

| feature set | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| All 64 | 0.4894 | 0.9036 | 0.0462 |
| Top 20 by importance | 0.4895 | 0.9056 | 0.0460 |
| **Dropping race-constant features (57)** | **0.5146** | **0.9091** | **0.0448** |

Race-constant features (weather, lap count, career round counts) were acting as
**circuit proxies** — memorisation of which race the model is looking at, which
cannot generalise to an unseen circuit. Dropping them is simpler *and* better,
and it is the single largest improvement in the project after the transfer
layer.

The validated `circuit_*` features are deliberately kept. They are also
constant within a race, but they encode physical character that was shown to
transfer rather than an incidental fingerprint. The distinction is the point.

## 11. The sequence model

Every other model treats a battle-lap as an independent row with hand-computed
history. A GRU over the episode learns its own summary instead. The comparison
is deliberately fair: identical features, target and protocol.

| | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| Random forest | 0.506 | 0.906 | 0.0454 |
| LightGBM | 0.484 | 0.905 | 0.0462 |
| GRU, pretrained on 2022-25 | 0.453 | 0.876 | 0.1018 |
| GRU | 0.452 | 0.886 | 0.1172 |
| GRU, frozen encoder | 0.294 | 0.870 | 0.1677 |

It loses, as expected at ~300 positives in an early training fold. Two details
matter more than the headline:

1. On **ranking** the gap to LightGBM is 0.03 against a noise floor of 0.22, so
   the GRU is not really distinguishable there. Where it is clearly worse is
   **calibration**, with Brier more than twice as bad. For a model whose output
   is a displayed probability, that is disqualifying on its own.
2. **Pretraining the encoder on four prior seasons bought +0.3%**, against
   +4.22% for the same era supplied as a frozen scalar feature. At this scale
   the old era transfers **as a feature, not as a representation** — the
   opposite of what the design predicted.

## 12. Overfitting and convergence

Training on a growing prefix of rounds while always scoring the same held-out
race isolates training size from which race is being predicted:

| train rounds | train rows | train PR-AUC | test PR-AUC | gap |
|---|---|---|---|---|
| 2 | 829 | 0.930 | 0.250 | 0.680 |
| 4 | 1,989 | 0.968 | 0.163 | 0.805 |
| 8 | 3,817 | 0.970 | 0.301 | 0.669 |
| 13 | 6,240 | 0.963 | 0.264 | 0.699 |

Two conclusions:

- **The model overfits heavily.** A generalisation gap above 0.70 at every
  training size. It nearly memorises the training data.
- **More data has not improved accuracy.** Test performance is flat from 2
  training rounds to 13, which matches the per-round trajectory showing no
  trend. The extra data bought narrower *uncertainty* (section 9.3) and nothing
  else.

The feature-reduction result in section 10.1 is the natural response, and
regularising harder is the obvious next step.

## 13. Error analysis

Sliced by circuit, the failures are concentrated and point the same way:

| circuit | rows | actual rate | predicted | bias | PR-AUC |
|---|---|---|---|---|---|
| **Monaco** | 558 | 2.0% | 7.2% | **+5.2 pp** | **0.09** |
| Australian | 75 | 5.3% | 5.9% | +0.6 pp | 0.18 |
| Spanish (Madrid) | 289 | 2.4% | 5.8% | +3.4 pp | 0.35 |
| **Italian (Monza)** | 448 | 12.7% | 6.6% | **-6.1 pp** | 0.67 |
| Hungarian | 456 | 6.4% | 7.3% | +0.9 pp | 0.85 |

**The two extremes of circuit passability are exactly where the model breaks,
in opposite directions.** It is far too optimistic at Monaco and too pessimistic
at Monza. This is the direct, measurable cost of excluding circuit identity, and
it is the strongest argument for finishing the geometric circuit representation.

By gap band, calibration is otherwise excellent (bias under 0.007 in every
band), and the worst individual misses are all long-range passes at Monza —
passes from 1.9 to 2.9 seconds back that the model rated near zero.

## 14. What did not work

Recorded deliberately. Of eight design predictions this project made and then
tested, **six were overturned by measurement and two held**. The misses shaped
the work more than the hits did, and several of them produced the largest
improvements in the final model.

| Prediction | Outcome |
|---|---|
| Historical circuit prior transfers | **Failed.** rho = +0.09 (p = 0.76). |
| Monotone constraints help at small n | **No effect.** 0.4662 vs 0.4641. |
| Cross-race form is the biggest gap | **+0.8%.** In-race pace already subsumes season form. |
| Hyperparameter tuning helps | **-1.2%.** The search selected noise; top 5 trials spanned 0.008 against a 0.065 fold SD. |
| Old-model-as-feature is the best transfer | **Held.** +4.22%, 9 of 10 rounds. |
| Pretraining is where a sequence model wins | **Failed.** +0.3%. |
| Circuit *physics* transfers where circuit *outcome* did not | **Held.** rho +0.97 for speed traps, and +1.2% PR-AUC. |
| More features are better | **Failed.** Dropping 7 race-constants gained 5.1%. |

The frozen tuned parameters remain the default despite being 1.2% worse,
because switching to the hand-picked prior *after seeing it win on the test
rounds* is precisely the post-hoc selection the protocol exists to prevent.

## 15. Engineering and MLOps

The retraining loop is **implemented and runnable but deliberately dormant** —
the cron in `.github/workflows/race-weekend.yml` is commented out. This is a
demonstration that the design is understood, not a service anyone depends on.

```
ingest round k → features → score round k with the INCUMBENT
              → retrain challenger → promotion gate → promote or archive
```

Three design points worth stating:

- **The incumbent scores each new race before anything is retrained**, so one
  computation yields both the out-of-sample result and the drift signal, and it
  is the same computation the backtest performs.
- **The promotion gate compares configurations, not fitted objects.** After
  round k the challenger has trained on rounds 1..k while the incumbent trained
  on 1..k-1, so scoring both on recent races would compare a model that has seen
  them against one that has not, and the gate becomes a rubber stamp. The gate
  instead re-runs the walk-forward backtest over the trailing five rounds for
  both configurations. Five rounds, not one, because of the 0.22 noise floor.
- **Drift reports only what its sample size supports.** Race-level features
  (weather, lap count) have one value per race and need eight races a side, so
  they stay silent until 2027. Career counters drift by construction and are
  never reported. Row-level features are testable from round one.

Tracking is Weights & Biases (one run per model, per-round metrics as a step
series). Results are also exported to a static dashboard at `ui/index.html`.

### Reproducing

```bash
cd era2026
uv sync
uv run era2026-warm             # cache the season (needs network)
uv run era2026-data             # build the modelling table
uv run era2026-tune             # frozen hyperparameter search
uv run era2026-backtest         # the ladder, logged to W&B
uv run era2026-export           # dashboard data
uv run era2026-ui               # build ui/index.html
uv run pytest                   # 88 tests
```

## 16. Limitations

1. **Fourteen races.** 470 events is a small sample and every conclusion should
   be read against a per-round standard deviation of 0.22.
2. **Circuit passability is still not represented.** Circuit *physics* was
   validated and added (section 7.2b), which helped Monaco's ranking but barely
   moved the bias. What is missing is track width, braking-zone count and
   corner-exit geometry, none of which are derivable from timing data. This is
   the clearest single piece of future work and it needs an external source.
3. **Overtake Mode is observable but not identifiable.** Its availability
   correlates with `neutralised` at **-0.954** — the aid is withdrawn essentially
   only when passing is forbidden anyway. Only 49 of 6,669 rows are
   green-track-with-mode-off. The feature is kept but credited with nothing.
4. **Per-car energy deployment is invisible.** Whether a driver had energy banked
   and chose to spend it is not exposed anywhere in the data.
5. **The model overfits** and the feature set is over-engineered (sections 10.1,
   12). Stronger regularisation and the reduced feature set are the obvious next
   steps.
6. **One season means no cross-era validation of the transfer claim.** The
   +4.22% is measured within 2026 only.

## 17. Declaration of LLM use

An LLM assistant was used throughout for implementation, debugging and drafting,
in line with the course's fair-use guidance. Specifically it was used to write
and refactor code, to draft this report from the project's own measured results,
and to propose analyses.

Every methodological choice in this report is supported by a measurement
recorded in the repository's commit history rather than by assertion, and the
project's failed predictions (section 14) are reported alongside its successes.
The author is responsible for, and able to justify, every choice, parameter,
result and interpretation herein.
