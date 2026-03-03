# Data v2 — Enriched 45-column battles

- **Source**: FastF1 (pipeline) or derived from v1 with v2 schema
- **Columns**: 45 (adds year, race_name, race_progress, positions, gap_ahead, speed traps, stint/fresh_tyre, weather)
- **Current file**: `battles_2022.csv` — 35,168 rows, v2 schema (year=2022, race_name from track; speed traps/weather as placeholders when derived from v1)
- **Generate full v2**: `cd src && python3 -m pipeline.main --years 2022 2023 2024 --output ../data/v2/battles_2022_2024.csv --cache cache`
- **Changes vs v1**:
  - Added: year, race_name, total_laps, race_progress, attacker/defender position
  - Added: speed trap data (SpeedI1, SpeedI2, SpeedFL, SpeedST) per driver
  - Added: stint number, fresh_tyre flag per driver
  - Added: air_temp, track_temp, humidity, rainfall, wind_speed
  - Renamed: attacker_speed -> attacker_lap_time, speed_difference -> gap_ahead
