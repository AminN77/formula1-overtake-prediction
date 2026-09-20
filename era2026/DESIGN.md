# Design: transfer strategy, evaluation, architecture

Companion to `PLAN.md`. Three questions answered here, in order:

1. Which pre-2026 data transfers, which does not.
2. How the model is evaluated.
3. What the architecture is.

No code until these are agreed.

---

## 1. What transfers from the old era

### The reframe

The worry was: "we know the 2026 rules favour more overtaking, but we have no data to prove it."

We do not need to prove it, and we must not try to learn it from old data. The overtake *rate* is directly measurable from 2026 data alone, and it belongs in the model's intercept. What old data is good for is the *conditional structure*: given a battle, which conditions make a pass more likely. That structure is mostly geometry, weather, tyre state and relative pace, and most of it did not change.

So the rule for the whole project:

> **Levels come from 2026. Shapes come from the old era.**

Pooling old and new data naively violates this, because the model fits a single intercept across two regimes with different base rates and then leans on old-era absolute car performance to explain the difference. It will look fine in cross-validation and fail on the next race.

### Tier 0: reuse that is not a feature at all

The highest-value thing in `legacy/` is not data, it is method.

| Asset | Why it transfers |
|---|---|
| Pairwise order-flip overtake extraction | Defined on positions and laps, not on car behaviour. Regulation-independent. |
| Pit-stop and retirement exclusion logic | Same. |
| Validation against published season overtake totals | The audit methodology transfers even though the totals do not. |
| Track metadata (sector types, track types) | Circuit geometry. |

Reuse the logic, rewrite the code, re-validate the counts against published 2026 figures.

### Tier 1: use old data directly, as fitted values or priors

These are properties of the world or of people, not of the 2026 car. Old data gives us 8 seasons to estimate them, against 14 races in-era.

| Quantity | Source | Notes |
|---|---|---|
| **Circuit passability** | 2018-2025 | The single most valuable transfer. See below. |
| `sector_type`, `track_type` | 2018-2025 | Geometry. |
| Weather effects (`air_temp`, `track_temp`, `humidity`, `rainfall`, `wind_speed`) | 2018-2025 | Rain does the same thing to overtaking in any regulation era. |
| Qualification rank and rank difference | 2018-2025 | Ordinal within a session, so era-invariant by construction. A car qualifying 6 places back is out of position in any era. |
| Teammate-relative driver skill (`quali_vs_teammate`, `race_pace_vs_teammate`) | 2018-2025 | Teammate-relative measures cancel the car, which is exactly the thing that changed. Strongest single class of transferable driver feature. |
| Race-state definitions (`race_progress`, `laps_remaining`, `race_phase`, `safety_car`, `yellow_flag`) | definitional | Free. |

### Tier 2: use old data for shape, not for level

The feature exists in 2026 and means the same thing, but its relationship to the target shifted. Usable through the transfer mechanisms in the next section, not through pooling.

| Feature group | What changed |
|---|---|
| `gap_ahead`, `gap_to_leader`, `gap_to_car_ahead`, `gap_to_car_behind` | Measurement is identical. P(pass given gap) moved, because dirty air and following ability changed. This is the core of the shift. |
| Gap dynamics: `gap_delta_1/2/3`, `is_closing`, `closing_laps`, `gap_mean_3`, `gap_min_3`, `closing_rate`, `battle_duration` | Best transfer candidates in this tier. Relative and dynamic, so less exposed to absolute car performance. "Has been closing for three laps" means the same thing in 2026. |
| `pace_delta`, `pace_delta_avg_3`, `pace_delta_std_3` | Deltas transfer, absolute lap times do not. |
| Tyre state: compounds, `tyre_age`, `stint`, `compound_advantage` | Compound labels survive. 2026 tyres are 25-30mm narrower with new construction, so degradation curves and the cliff are different. Keep the structure, refit the thresholds. |
| `tyre_cliff_risk` | Currently hardcoded cliff ages (28 laps and similar). Those numbers are certainly wrong for 2026. Refit in-era or drop. |
| Sector deltas, `strongest_sector` | Relative, so partial transfer. |
| `gap_pressure_ratio`, `rear_pressure_ratio` | Ratios, so relatively stable. |
| `prior_pair_overtakes`, `overtakes_so_far` | Structure transfers. Level does not, because it scales with the era's overtake rate. Normalise by the in-race or in-era rate. |

### Tier 3: drop, and learn only from 2026

| Feature | Reason |
|---|---|
| `is_in_drs_zone`, `drs_zone_length` | The mechanism no longer exists. |
| `attacker_team`, `defender_team` as raw categories | Grid membership changed: Cadillac is an 11th team with no history, Audi replaced Sauber, and team identity carries a 2022-2025 performance level that is no longer true. Replace with in-era computed team performance (pace rank, constructor rank), which is a method rather than a value. |
| All absolute speeds: `speed_i1`, `speed_i2`, `finish_line_speed`, `straight_speed` for both cars | Different cars, different top speeds, and active aero means straight-line speed is now partly a mode choice rather than a fixed car property. Keep the deltas in Tier 2, drop the absolutes. |
| Absolute `lap_time` for both cars | Different cars. Use `pace_delta` instead. |
| `attacker_overtake_rate_last5`, `defender_defend_rate_last5` | Label-derived, and the label was measured under DRS. Recompute in-era only. |
| `pit_stop_involved` | Looks at future laps. Never a feature. Becomes the censoring signal instead. |

### `drs_train_size`: redefine rather than drop

Worth calling out separately. The feature counted cars within 1.0s ahead, which was a DRS-range concept. But 2026 Overtake Mode also unlocks on being within 1 second at a detection point. So the 1 second threshold survives by coincidence, and the concept of being stuck in a queue survives outright. Redefine it against the Overtake Mode rule rather than deleting it.

### Circuit passability: the highest-leverage single idea

2026 gives roughly one race per circuit. Treating `track` as a 23-level categorical against ~600 rows per level is hopeless, and it is the kind of thing that quietly eats all the model's capacity.

Instead, collapse it to one number per circuit, fitted on 2018-2025:

- estimate a per-circuit overtake-per-battle rate across 8 seasons,
- shrink it toward the global mean (partial pooling), so low-sample circuits are not overfitted,
- feed the shrunk value into the 2026 model as a single continuous feature.

The absolute rate changed in 2026, but the *ranking* is driven by geometry: straight length, heavy braking zones, track width, corner sequence. Monaco is hard to pass at under any ruleset, Monza and Interlagos are easy. That ranking is what we are borrowing, and the model rescales it in-era.

**One gap:** Madrid is new for 2026 and has no F1 history (Imola was dropped to make room). Every other circuit on the calendar has pre-2026 data. So we need a fallback for exactly one circuit: estimate passability from geometry (longest straight, number of heavy braking zones, corner count, track width) fitted against the known circuits, and use the prediction. That is a small, bounded, honest piece of work rather than a hole in the design.

### Mitigation strategies, ranked

Everything above is about *what* to transfer. These are the *mechanisms*, ranked by expected value over implementation cost.

1. **Old-model-as-feature (stacking).** Train a model on 2018-2025 restricted to Tier 1 and Tier 2 features, score every 2026 row with it, use that score as a single input to the 2026 model. The 2026 model then only has to learn the correction, which is a far smaller function than the whole problem. Cheap, low risk, trivially ablatable, and it degrades gracefully: if the old era is useless, the 2026 model simply ignores the feature.
2. **Circuit passability prior.** As above. Largest variance reduction available.
3. **Monotonicity constraints.** LightGBM and XGBoost both support per-feature monotone constraints. Encode what we already know without any data: pass probability rises as the gap falls, as the closing rate rises, as the attacker's tyre advantage grows. This injects physics as a prior and is unusually effective at small sample sizes. Underrated and almost free.
4. **Hierarchical model with an era offset.** Pool 2018-2026 but give each era its own intercept, so the pooled data informs the slopes while the 2026 intercept absorbs the level change. This is the principled version of "levels in-era, shapes from old data".
5. **Recency sample weighting.** Weight 2018-2021 low, 2022-2025 medium, 2026 high. Crude, but a legitimate baseline and it costs nothing to try.
6. **Driver skill priors.** Hierarchical driver effects fitted on 2018-2025 used as priors for 2026 driver effects, for the drivers who have history. Rookies fall back to the population mean.

Strategy 1, 2 and 3 are the recommended starting set. Each is a separable component with a clean on/off switch, so the value of transfer becomes measurable rather than assumed.

**What not to do:** pool all seasons into one table with team identity and absolute speeds in the feature set. That is the default path and it is wrong.

---

## 2. Evaluation

### Formulation: expanding-origin backtest, one race ahead

Adopting the proposed n-1 / 1 shape rather than a fixed 10 / 4 split. Reasoning: it matches what the production pipeline actually does, so the evaluation number and the deployed behaviour are the same thing measured once.

For each round `k` from `k0` to `N`:

1. train on rounds `1 .. k-1` (plus whatever old-era data the transfer components use),
2. predict round `k`,
3. record metrics for round `k`,
4. advance.

Every round is a test set exactly once, and always genuinely out of sample. No separate sealed holdout is needed, which is the elegant part: with a fixed 10 / 4 split we would get one test result from 4 races, and here we get 10 test results from the same 14 rounds.

Suggested `k0 = 5`. Four races is the minimum that produces a trainable model, giving 10 backtest rounds from the 14 currently available.

### Hyperparameters

They cannot be chosen by looking at round `k`. Two acceptable options:

- **Recommended:** fix hyperparameters once on an early block (rounds 1 to 4), then freeze them for the entire backtest. Simple, honest, standard practice in forecasting, and it matches how the production pipeline will behave.
- Alternative: a nested inner split per fold (train `1..k-2`, validate `k-1`). More faithful, considerably more compute, and at this sample size the inner validation set is one race, which is too noisy to tune against.

Take the first.

### Reporting

- Primary: PR-AUC and calibration error, aggregated across backtest rounds with a bootstrap interval.
- Secondary: ROC-AUC, Brier.
- **The per-round metric series is a headline result, not a diagnostic.** It shows whether the model improves as the season accumulates, which is the project's entire premise. It is the plot to put in any write-up.
- Always against the baseline ladder: base rate, gap alone, logistic regression on five features. A complex model reports its gain over that table or it does not ship.

### The noise problem, and its consequence for the pipeline

One race is roughly 600 candidate rows with about 50 positives. PR-AUC on that has wide error bars. A single race cannot distinguish a better model from a luckier one.

Two consequences, both architectural:

- Never promote or demote a model on one race.
- The promotion gate uses a **rolling window** (last 5 rounds) rather than the most recent race. This has to be in the pipeline design from the start, not patched in after the first bad week.

---

## 3. Architecture

### 3.1 Prediction unit and target

**Discrete-time hazard over battle episodes.**

- A **battle episode** is one attacker-defender pair across consecutive laps where they are adjacent and inside the candidate threshold.
- Each lap of the episode is one row.
- Per row the outcome is: pass on the next lap (event), episode continues (survives), or episode ends for another reason (**censored**: pit stop, retirement, safety car neutralisation, gap opening beyond threshold, race end).
- The model predicts the per-lap hazard `h(t) = P(pass on lap t+1 | still battling at t)`.
- Any horizon comes out by composition: `P(pass within k laps) = 1 - prod(1 - h)`.

Why this rather than the legacy binary within-3-laps target:

| Problem in `legacy/` | Fixed by hazard framing |
|---|---|
| Consecutive laps of one pair shared an overlapping 3-lap label window, so rows were heavily correlated and the effective sample size was far below the row count | One row, one conditional outcome, no window overlap |
| Rows where a car pitted were deleted, throwing away real situations | Pitting becomes censoring, which is information rather than a row to drop |
| Three separate label columns for three horizons | One model, horizon is a query parameter |

### 3.2 Layers

Each layer is a separable component with a clean interface, so any of them can be ablated to measure what it contributes.

```
Layer 0   Feature builder
          Emits three blocks, explicitly tagged by transfer tier.
          Tier tags are not documentation, they are what the transfer
          components in Layer 1 operate on.
             |
Layer 1   Transfer components (fitted on 2018-2025, frozen)
          a. circuit passability prior      -> one scalar per circuit
          b. old-era model score            -> one scalar per row
          Both enter Layer 2 as ordinary features. Switchable.
             |
Layer 2   In-era hazard model (fitted on 2026 rounds 1..k-1)
          Gradient boosting (LightGBM), with:
            - monotone constraints on known physical directions
            - the Layer 1 scalars
            - shallow trees, strong regularisation (n is small)
             |
Layer 3   Calibration
          Isotonic or Platt, fitted on the most recent rounds,
          refitted every round.
             |
Layer 4   Uncertainty
          Conformal wrapper over the calibrated hazard.
          Per-row intervals that narrow as rounds accumulate.
```

Layer 4 is not decoration. At 700 positives a bare point probability is not defensible, and interval width shrinking over the season is a direct demonstration of the project's premise.

### 3.3 On the sequence model

Since this was an open question, stating plainly what it would be and why it is not first.

**What it is.** Instead of hand-computing `gap_delta_1/2/3`, `gap_mean_3`, `battle_duration` and the rest, feed the model the raw per-lap sequence of the episode (gap, pace delta, speeds, tyre state, lap by lap) and let a small recurrent network or transformer encoder learn the temporal pattern, with a hazard head on top. Mathematically it is the same model as above. The only difference is a learned encoder in place of hand-crafted aggregates.

**Why not now.** With roughly 8,000 rows and 700 positives, a learned encoder has far more parameters than the data supports. The hand-crafted aggregates already capture most of what a 3 to 5 lap window contains, and they encode it with zero parameters. The honest expectation is that it loses to LightGBM, and reporting that is a legitimate result rather than a failure.

**When it becomes interesting.** Two conditions, either of which changes the answer:

- End of 2027, at roughly 40 races and 2,000 positives.
- **Pretraining the encoder on 2018-2025 episode sequences**, then refitting only the hazard head on 2026. This is the one setting where a sequence model plausibly wins, because the encoder learns what a closing battle looks like from 8 seasons while only the small head is exposed to the 14-race sample. It is also the natural home for the pretraining question, and it fits the transfer philosophy above exactly: the encoder learns the shape, the head learns the level.

So the sequence model is deferred, not dismissed, and the layered architecture leaves room for it: it replaces Layer 0's aggregates and Layer 2's estimator, leaving Layers 1, 3 and 4 untouched.

### 3.4 Pipeline architecture

Race-triggered, which follows directly from the n-1 / 1 evaluation shape.

```
race weekend ends
      |
ingest round k                      -> raw store (parquet, partitioned by season/round)
      |
extract events and candidates       -> episode store
      |
build features (tiered)             -> feature store
      |
score round k with the INCUMBENT    -> this is both the out-of-sample
      |                                test result and the drift signal
retrain challenger on rounds 1..k
      |
promotion gate: rolling 5-round comparison, challenger vs incumbent
      |
promote or archive                  -> registry (versioned, data snapshot hash)
      |
serve predictions for round k+1
```

The load-bearing idea: **scoring the new race with the old model, before retraining, is simultaneously the evaluation and the monitoring.** One step, two purposes, and it is the same computation the backtest performs. The backtest and the production loop are the same code path, which is what keeps the reported number honest over time.

Components:

| Concern | Choice | Reason |
|---|---|---|
| Orchestration | GitHub Actions cron | Runners have open internet. The F1 live timing endpoint is blocked from both the cloud container and the sandboxed device shell, so CI is the only automatable option. |
| Storage | Parquet, partitioned by season and round | Incremental append, no database needed at this scale. |
| Experiment tracking | **Weights & Biases** | One run per round, grouped per backtest sweep. The per-round metric series is the project's headline result, and W&B plots it natively. Needs an API key as a CI secret. |
| Data and model versioning | **W&B Artifacts** | Covers dataset snapshots and model versions in the same place as the runs, so "the model as of round 14" resolves to one lineage. Replaces DVC: one fewer moving part, and the snapshot is attached to the run that used it. |
| Registry | W&B Artifact aliases (`incumbent`, `challenger`) | A single model, so the registry is one pointer rather than a version table. |
| Config | YAML, one file per experiment | Hyperparameters are frozen per the evaluation protocol, so they belong in version control, not in a notebook cell. |
| Drift | Feature distributions and base rate tracked per round | A regulation era is not stationary. Teams develop, and 2027 will not resemble 2026. |

Everything is a Python package with a CLI, so the same code runs locally and in CI. **No notebooks in the training path.** Notebooks are for exploration and they do not produce artifacts.

---

---

## 4. Interface

Confirmed as real scope rather than an afterthought, and it replaces the legacy app rather than extending it.

### One model, so the UI's job changes

The legacy app spent most of its surface on a schema-driven form with dozens of fields, plus a model switcher across v2 to v6. Both go away:

- **No version switcher.** One model, one incumbent pointer. Provenance (which rounds it was trained on, when it was promoted) is shown as metadata, not as a choice.
- **No feature entry form.** This is the biggest improvement available. Features are derived from real session data, so a user picks a race and a battle rather than typing 97 numbers. The legacy form existed because the model was disconnected from the data; it does not need to exist here.

### What the UI actually shows

Four views, in priority order:

| View | What it shows | Why it earns space |
|---|---|---|
| **Race replay** | A completed 2026 round. Every battle episode on a lap timeline, predicted hazard per lap against what actually happened. | The natural visual for this model. Hazard is a per-lap curve, so it wants a timeline, not a gauge. Also the fastest way to see where the model is wrong. |
| **Model health** | The per-round backtest metric series, calibration curve, and conformal interval width over rounds. | This is the project's thesis made visible: does the model improve as the era accumulates data. The single most important chart in the project. |
| **Battle inspector** | One episode, lap by lap: hazard curve, the features driving it, the transfer scalars' contribution. | Where the explanation lives. Replaces the legacy sensitivity chart with something tied to a real situation. |
| **Next round** | Standing predictions for the upcoming race, with intervals. | The only forward-looking view, and the reason the pipeline exists. |

### Stack

Same React and TypeScript baseline as legacy, since that part worked. The difference is in the information design rather than the framework: timelines and small multiples instead of forms and tables, and semantic colour for state (event, censored, predicted) separate from the accent hue. Charts are the product here, so they get the care.

Built last, after the model has numbers worth showing. A UI built against a model that does not exist yet ends up designing the model.

---

## Open questions

**Settled.** `k0 = 5` for the backtest start. Transfer set starts as old-model-as-feature, circuit passability prior and monotone constraints, with the hierarchical era offset, recency weighting and driver priors held as experiments behind the same switches. Tracking is Weights & Biases. One model, no version switcher. The UI is in scope.

**Open.**

1. **Old-era corpus depth.** Proposal: start with 2022-2025, because `legacy/data/v6/scenarios_*.csv` already carries the Tier 1 and Tier 2 columns the transfer components need, which means Layer 1 can be built and tested today with no F1 API access at all. Extending back to 2018 is a later, separable job that only sharpens the circuit prior. Proceeding on this unless told otherwise.
2. **Phase 0 must run outside the sandbox.** `livetiming.formula1.com` is blocked from both the cloud container and the device shell, so `era2026/recon.py` has to be run in a normal local terminal. Its output gates the extraction layer, because we do not yet know what 2026 exposes in place of DRS.
