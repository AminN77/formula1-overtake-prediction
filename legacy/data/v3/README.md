# Data v3 — IP02 improvements

- **Source**: FastF1 pipeline with IP02 fixes applied
- **Columns**: 51
- **Files**: `battles_2022.csv`, `battles_2023.csv`, `battles_2024.csv`, `battles_2025.csv`

## Changes vs v2

### Data quality fixes (IP02 P0)
- **`gap_ahead`** now uses actual inter-car gap from `LapStartTime`, not pace difference
- **`tyre_age_difference`** is now signed (attacker − defender; negative = attacker on fresher tyres)
- **`pit_stop_involved`** flags rows where either driver pits on the current or next lap

### New features (IP02 P1–P2)
- **`pace_delta`**: `defender_lap_time − attacker_lap_time` (positive = attacker faster)
- **`speed_i1_delta`**: attacker − defender speed at intermediate 1
- **`speed_i2_delta`**: attacker − defender speed at intermediate 2
- **`speed_fl_delta`**: attacker − defender finish-line speed
- **`speed_st_delta`**: attacker − defender straight speed

### Impact
- Row counts are lower than v2 because the actual-gap filter is stricter than the
  pace-difference filter: many v2 "battles" involved cars that were close in pace
  but far apart on track.

## Regenerate

```bash
python3 -m pipeline.main --years 2022 --output data/v3/battles_2022.csv --cache cache
python3 -m pipeline.main --years 2023 --output data/v3/battles_2023.csv --cache cache
python3 -m pipeline.main --years 2024 --output data/v3/battles_2024.csv --cache cache
python3 -m pipeline.main --years 2025 --output data/v3/battles_2025.csv --cache cache
```
