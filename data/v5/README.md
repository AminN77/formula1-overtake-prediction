# data/v5 — IP04 Context-Aware Dataset

**Generated from:** `src/pipeline` with IP04 extensions  
**Columns per row:** 79  

## What changed from v4 (62 cols → 79 cols)

### BattleRecord additions (8 new fields in detector)
| Column | Type | Description |
|--------|------|-------------|
| `attacker_team` | str | Constructor / team name |
| `defender_team` | str | Constructor / team name |
| `same_team` | bool | Teammate battle flag |
| `gap_to_car_ahead` | float | Gap (s) from defender to car in P−1 |
| `gap_to_car_behind` | float | Gap (s) from attacker to car in P+1 |
| `drs_train_size` | int | Number of cars within 1 s chain |
| `race_phase` | str | opening / middle / closing |
| `stint_phase` | str | fresh / mid / degraded / cliff |

### Enrichment additions (9 new columns)
| Column | Source | Description |
|--------|--------|-------------|
| `attacker_team_pace_rank` | team_features.py | Season-to-date team pace rank |
| `defender_team_pace_rank` | team_features.py | Season-to-date team pace rank |
| `team_delta` | team_features.py | attacker rank − defender rank |
| `attacker_positions_gained_avg` | team_features.py | Rolling 5-race mean(grid − finish) |
| `defender_positions_gained_avg` | team_features.py | Rolling 5-race mean(grid − finish) |
| `attacker_quali_vs_teammate` | team_features.py | Rolling quali advantage vs teammate |
| `defender_quali_vs_teammate` | team_features.py | Rolling quali advantage vs teammate |
| `attacker_race_pace_vs_teammate` | team_features.py | Rolling race pace advantage vs teammate |
| `defender_race_pace_vs_teammate` | team_features.py | Rolling race pace advantage vs teammate |

## Regeneration

```bash
cd src/
python -m pipeline.main --years 2022 2023 2024 2025 --output-dir ../data/v5 --cache cache
```
