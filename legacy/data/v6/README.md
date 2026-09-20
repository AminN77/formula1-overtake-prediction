# data/v6 — Broad Scenario Overtake Dataset

**Generated from:** `pipeline/v6_pipeline.py`  
**Primary target:** `label` = `overtake_within_3`  
**Train/test convention:** train on 2022-2024, test on 2025

## What v6 changes

v1-v5 used a narrow battle detector:

- adjacent cars only,
- gap under 1.0 s,
- next-lap overtake target.

v6 keeps the problem supervised, but broadens both the positives and the candidate universe:

- positives come from the wider order-flip extraction used in `notebooks/pipeline_testing.ipynb`
- candidates are adjacent attacker-defender scenarios within **3.0 s**
- labels are written for `next_lap`, `within_2`, and `within_3`
- the default binary target is `label` = `overtake_within_3`

## Files

- `scenarios_<year>.csv`
  - supervised training table with engineered features and labels
- `raw_overtakes_<year>.csv`
  - raw pairwise order-flip events before filtering
- `filtered_overtakes_<year>.csv`
  - positive events after notebook-style filtering
- `audit_<year>.csv`
  - per-race extraction audit
- `summary.csv`
  - season-level row counts and positive rates

## Key label columns

- `label`: primary v6 target (`overtake_within_3`)
- `overtake_next_lap`
- `overtake_within_2`
- `overtake_within_3`

## Feature groups

- race state: `lap_number`, `race_progress`, `laps_remaining`
- local battle: `gap_ahead`, `pace_delta`, speed-trap deltas
- short-horizon dynamics: `gap_delta_1`, `gap_delta_2`, `gap_delta_3`, `gap_mean_3`, `pace_delta_avg_3`, `battle_duration`
- tyre/strategy: compounds, tyre ages, stints, `tyre_cliff_risk`, `compound_advantage`
- track context: `track`, `track_type`, `sector_type`, `is_in_drs_zone`, `drs_zone_length`
- race context: `gap_to_car_ahead`, `gap_to_car_behind`, `drs_train_size`, `overtakes_so_far`
- form/team context: rolling driver features, team pace ranks, constructor ranks

## Regeneration

From the repository root:

```bash
python3 -m pipeline.main \
  --dataset-version v6 \
  --years 2022 2023 2024 2025 \
  --output-dir data/v6 \
  --cache cache
```

## Current summary

| Year | Candidate rows | Positive rows | Positive rate |
|------|----------------|---------------|---------------|
| 2022 | 11,380 | 1,256 | 11.04% |
| 2023 | 12,334 | 1,289 | 10.45% |
| 2024 | 13,229 | 1,226 | 9.27% |
| 2025 | 14,565 | 1,194 | 8.20% |
