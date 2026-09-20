# Overtake Prediction for the 2026-2030 Formula One Era

**Status:** planning, nothing built yet
**Scope:** 2026 regulations onward. No reuse of the 2022-2025 models or datasets.
**See also:** [`DESIGN.md`](DESIGN.md) settles the transfer strategy, the evaluation
protocol and the architecture. Where it differs from Phases 2 and 3 below, it wins.

---

## 1. What this project is

Predict on-track overtaking in the Formula One regulation era that began in 2026, using only in-era data, and keep the model current as the era unfolds.

The era is ongoing. As of September 2026, 14 of 23 rounds of the first season are complete. Every race weekend adds data. That fact is not an inconvenience to work around, it is the defining property of the problem and the reason this project exists as something separate from `legacy/`.

---

## 2. Why this is not a continuation of `legacy/`

The 2026 regulations changed the mechanism of overtaking, not just the lap times.

| Change | Consequence for the model |
|---|---|
| DRS removed, replaced by Overtake Mode (extra energy deployment unlocked when within 1s at a detection point) | Every DRS feature in `legacy/` is dead: `is_in_drs_zone`, `drs_zone_length`, `drs_train_size`. The replacement mechanism is energy-based and state-dependent, not zone-based. |
| Active aero (straight mode / corner mode) on front and rear wings | Straight-line speed advantage is now partly a driver and strategy choice, not a fixed car property. Speed-trap deltas mean something different. |
| 50/50 split between combustion and electric power | Energy management becomes a first-class predictor. A car can be fast on one lap and flat the next because it spent its deployment. |
| Cars 30kg lighter, 200mm shorter wheelbase, narrower floor and tyres | Following distance and dirty-air behaviour change. The relationship between gap and pass probability is re-learned from scratch. |

A model trained on 2022-2025 does not transfer as-is. Retraining the legacy pipeline on 2026 data would silently keep features that no longer exist and a battle definition built around DRS zones. Greenfield is the correct call.

`legacy/` stays in the repository as a reference point and as a source of comparison numbers, not as a codebase to extend.

---

## 3. The central constraint

**Roughly 14 races of in-era data exist today.**

Extrapolating from legacy extraction rates (about 600 candidate scenarios and 50 filtered overtakes per race), that is on the order of 8,000 candidate rows and 700 positive events. The legacy project had four full seasons and about 50,000 rows, and even that was optimistic because rows from consecutive laps of the same battle are heavily correlated.

This project starts with roughly one sixth of that, against a target whose mechanics are unfamiliar.

So the honest framing of the technical problem is:

> Build a useful, calibrated overtake model from a small and growing sample in a regime with no history, and improve it automatically as the regime accumulates data.

That framing makes three things load-bearing rather than optional:

1. **Uncertainty.** With this sample size, a point probability without an interval is not a defensible output.
2. **Transfer.** Pre-2026 data is wrong in the details but not worthless. Using it correctly is a research question, not a preprocessing step.
3. **The MLOps loop.** Automatic refetch and retrain is not a bolt-on at the end. It is the mechanism by which the model becomes good. It belongs in the design from day one.

---

## 4. Phases

### Phase 0: data reconnaissance (blocks everything else)

Nothing gets designed until we know what the 2026 data actually contains. Open questions:

- Is the `DRS` telemetry channel gone, still present but always zero, or repurposed?
- Is there any channel, race control message, or timing field exposing Overtake Mode availability or use?
- Is any energy deployment or ERS state exposed?
- Is there any signal for active aero mode?
- Do lap and timing schemas differ from 2022-2025 in any way that breaks the extraction assumptions?

**Deliverable:** a reconnaissance script plus a written findings note listing, channel by channel, what exists for 2026 and what does not.

**Note on where this runs:** the F1 live timing endpoint is blocked from both the cloud container and the sandboxed device shell. All FastF1 extraction has to run on your own machine, or in CI (GitHub-hosted runners have open internet, which is what makes Phase 4 viable).

### Phase 1: extraction layer

Written fresh, informed by Phase 0.

- Session loading, caching, and a local raw data store keyed by season and round.
- Overtake event extraction. Start from the pairwise order-flip rule that the legacy audit validated against public season totals, since that logic is regulation-independent. Re-validate it against published 2026 overtake counts before trusting it.
- Candidate scenario generation. The legacy `<3.0s` adjacency rule was tuned around DRS range. Re-derive the threshold from 2026 data instead of inheriting it.
- Feature groups: race state, pair dynamics, tyre and stint, track, weather, driver and team form. Overtake Mode and energy features only if Phase 0 says the data exists.
- Idempotent and incremental: re-running after a new race extends the store without recomputing history.

### Phase 2: labels and evaluation protocol

Decided and frozen **before** any model is trained. The legacy project got this wrong and the cost was a holdout that had been tuned against six times.

- **Label:** discrete-time hazard per battle episode. For each lap the pair is still battling, predict P(pass on the next lap | still battling), with censoring at pit stop, retirement, safety car neutralisation, and race end. This gives next-lap, within-2 and within-3 from one model, handles the competing risks properly instead of deleting those rows, and avoids the overlapping-label-window problem that inflated the legacy row count.
- **Splits:** forward-chaining by round, never random. Train on rounds 1..k, validate on k+1..k+m, test on the most recent held-back rounds. Test rounds stay sealed.
- **Grouping:** any cross-validation groups by race, never by row.
- **Baseline ladder, established first:** base rate, gap alone, logistic regression on five features, then anything more complex. Every later model reports its gain over this table or it does not ship.
- **Metrics:** PR-AUC and calibration as primary, ROC-AUC as secondary, plus a per-race breakdown. Thresholds are selected on validation only, never on test.

### Phase 3: models

A ladder, run against the frozen protocol:

1. Regularised logistic regression and a small gradient-boosted model. At this sample size these are genuinely competitive and they set the bar.
2. **Transfer from the pre-2026 era.** Options to test against each other: pretrain on 2018-2025 and fine-tune on 2026; train on old data and use its output as a single feature in the 2026 model; sample-weight old seasons by recency; or a shared representation with an era indicator. This is the most interesting experiment in the project and the one most likely to produce a result worth writing up.
3. **Sequence model over the battle episode** (GRU or a small transformer over per-lap history). Only if the data supports it. With ~700 positives it may well lose to boosting, and that negative result is worth reporting honestly.
4. **Uncertainty.** Conformal prediction for per-scenario intervals, or a Bayesian treatment. Intervals should visibly narrow as the season accumulates, which is a good demonstration of the whole premise.
5. **Calibration** as an explicit step on the validation split, not an afterthought.

### Phase 4: the continual learning loop

This is what the ongoing era buys us, and the reason MLOps sits inside the design rather than at the end.

- Scheduled job (GitHub Actions cron) triggered after each race weekend.
- Refetch the new round, extend the store, regenerate features.
- Retrain, evaluate against the frozen protocol, compare to the incumbent model.
- **Promotion gate:** the challenger replaces the incumbent only if it beats it on the sealed evaluation. Otherwise it is archived and flagged.
- **Drift monitoring:** track the feature distributions and the base overtake rate round by round. A regulation era is not stationary. Teams develop, and 2027 will not look like 2026.
- Experiment tracking and a model registry, so every promoted model is traceable to the exact data snapshot and config that produced it.
- Data versioning, so "the model as of round 14" is reproducible.

### A note on scope: the loop is dormant

Phase 4 is **implemented and runnable, but not running**. The schedule in
`.github/workflows/race-weekend.yml` is commented out and the workflow is
manual dispatch only.

That is deliberate. This is a demonstration that the loop is understood and
built, not a service anyone depends on. An enabled cron on an unattended
repository accumulates weekly failures and teaches nobody anything, and a job
that pushes to `main` with no one watching is a liability rather than a
feature. Activation is uncommenting four lines and adding one secret; the
manual path already exercises every step.

What is worth demonstrating is the reasoning, not the uptime:

* the incumbent scores each new race **before** anything is retrained, so
  evaluation and drift detection are one computation rather than two,
* the gate compares configurations walked forward over five rounds, never
  fitted objects over rounds the challenger has already trained on,
* the registry ties every promoted model to the exact data snapshot behind it,
* drift reports only what its sample size can actually support, and stays
  silent otherwise.

### Phase 5: serving

Deliberately last, and deliberately thin. A minimal API exposing per-horizon hazard output with intervals, plus the model's provenance (which rounds it was trained on, when it was promoted). No frontend until the model is worth looking at.

---

## 5. Open decisions

1. **How far back does transfer reach?** 2018-2025 is available and doubles as a pretraining corpus, but it costs a long extraction run. Alternative is 2022-2025 only, which is already extracted in `legacy/data/`.
2. **Is the first season the test set, or is it all training data?** Sealing the last few rounds of 2026 gives an honest number now. Using everything gives a better model but nothing trustworthy to report until 2027.
3. **Sequence model: in scope or stretch?** With this sample size it is a real risk of being a negative result.
4. **What is the deadline and is this assessed?** Phases 0, 1, 2 and 4 are the defensible core. Phase 3 items 2 and 3 are where the interesting work is.
5. **Language and stack.** Python is assumed. Worth deciding early whether the extraction layer stays pure Python or whether the heavier parts are worth writing differently.

---

## 6. Reuse policy

| From `legacy/` | Decision |
|---|---|
| Order-flip overtake extraction logic | Reuse the idea, rewrite the code, re-validate against 2026 published totals |
| Battle and candidate definitions | Rebuild. Built around DRS range. |
| Feature set | Reference only. Re-derive, drop all DRS features. |
| Trained models and datasets | Not used. Pre-2026 data may be used for transfer experiments only. |
| FastAPI and React app | Not carried over. Phase 5 starts clean. |
| Evaluation approach | Explicitly not reused. Rebuilt in Phase 2. |
