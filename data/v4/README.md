# Data v4 — IP03 improvements

- **Source**: FastF1 pipeline with IP03 extensions
- **Raw columns** (`BattleRecord`): 60
- **CSV columns**: 62 (adds `attacker_overtake_rate_last5`, `defender_defend_rate_last5` from rolling history)
- **Files**: `battles_2022.csv` … `battles_2025.csv`

## Changes vs v3

### §3.2 Track typing
- Expanded `track_info.TRACK_TYPES` so circuits are not silently labelled `street`.
- Unknown circuits default to `medium-speed` instead of `street`.

### §1.2 Multi-horizon labels
- `overtake_within_2`, `overtake_within_3`: attacker gains position within 2 / 3 laps.

### §3.4 Sector micro-features
- `sector1_delta`, `sector2_delta`, `sector3_delta` (defender − attacker sector time; positive = attacker faster).
- `strongest_sector`: 0–2 (S1–S3) or −1 if no positive advantage.

### Metadata
- `round_number`, `event_date` for calendar ordering.
- `compound_mismatch`: attacker vs defender compound differs.

### §3.3 Driver proxies (CSV enrichment)
- Computed **after** collecting all requested seasons in **one** run, then split by year.
- `attacker_overtake_rate_last5`, `defender_defend_rate_last5` use the previous five **races** (not laps) only.

## Regenerate (all years — required for correct driver features)

```bash
python3 -m pipeline.main --years 2022 2023 2024 2025 --output-dir data/v4 --cache cache
```

## Model notebook

Train and evaluate with `notebooks/model_testing_4.ipynb` (IP03: SHAP pruning, battle-pair model, temporal / LOCO checks).
