# 2026 data reconnaissance

- generated: `2026-09-20T11:44:36+00:00`
- fastf1: `3.8.3`
- python: `3.12.12`

## 1. 2026 schedule and which rounds are complete

- rounds on calendar: **23**
- rounds complete: **14**

| R | Event | Date | Format | Complete |
|---|---|---|---|---|
| 1 | Australian Grand Prix | 2026-03-08 | conventional | yes |
| 2 | Chinese Grand Prix | 2026-03-15 | sprint_qualifying | yes |
| 3 | Japanese Grand Prix | 2026-03-29 | conventional | yes |
| 4 | Miami Grand Prix | 2026-05-03 | sprint_qualifying | yes |
| 5 | Canadian Grand Prix | 2026-05-24 | sprint_qualifying | yes |
| 6 | Monaco Grand Prix | 2026-06-07 | conventional | yes |
| 7 | Barcelona Grand Prix | 2026-06-14 | conventional | yes |
| 8 | Austrian Grand Prix | 2026-06-28 | conventional | yes |
| 9 | British Grand Prix | 2026-07-05 | sprint_qualifying | yes |
| 10 | Belgian Grand Prix | 2026-07-19 | conventional | yes |
| 11 | Hungarian Grand Prix | 2026-07-26 | conventional | yes |
| 12 | Dutch Grand Prix | 2026-08-23 | sprint_qualifying | yes |
| 13 | Italian Grand Prix | 2026-09-06 | conventional | yes |
| 14 | Spanish Grand Prix | 2026-09-13 | conventional | yes |
| 15 | Azerbaijan Grand Prix | 2026-09-26 | conventional | no |
| 16 | Bahrain Grand Prix | 2026-10-04 | conventional | no |
| 17 | Singapore Grand Prix | 2026-10-11 | sprint_qualifying | no |
| 18 | United States Grand Prix | 2026-10-25 | conventional | no |
| 19 | Mexico City Grand Prix | 2026-11-01 | conventional | no |
| 20 | São Paulo Grand Prix | 2026-11-08 | conventional | no |
| 21 | Las Vegas Grand Prix | 2026-11-21 | conventional | no |
| 22 | Qatar Grand Prix | 2026-11-29 | conventional | no |
| 23 | Abu Dhabi Grand Prix | 2026-12-06 | conventional | no |

Sample round for deep inspection: **R14**

## 2. Session loading

- **2026**: 1106 lap rows, 22 drivers, event `Spanish Grand Prix`
- **2024**: 1008 lap rows, 20 drivers, event `Italian Grand Prix`

## 3. Lap schema, 2026 versus 2024

- columns 2026: **31**, 2024: **31**
- new in 2026: `none`
- gone from 2026: `none`

2026 lap columns:

```
Time, Driver, DriverNumber, LapTime, LapNumber, Stint, PitOutTime, PitInTime, Sector1Time, Sector2Time, Sector3Time, Sector1SessionTime, Sector2SessionTime, Sector3SessionTime, SpeedI1, SpeedI2, SpeedFL, SpeedST, IsPersonalBest, Compound, TyreLife, FreshTyre, Team, LapStartTime, LapStartDate, TrackStatus, Position, Deleted, DeletedReason, FastF1Generated, IsAccurate
```

## 4. Car telemetry channels, and the fate of DRS

- **2026** channels (10): `Date, RPM, Speed, nGear, Throttle, Brake, DRS, Source, Time, SessionTime`
  - `DRS` present. value counts: `{0: 35553}`
  - **DRS channel is present but entirely zero: dead field.**
- **2024** channels (10): `Date, RPM, Speed, nGear, Throttle, Brake, DRS, Source, Time, SessionTime`
  - `DRS` present. value counts: `{0: 17335, 1: 10778, 3: 1, 8: 209, 10: 4, 12: 852, 14: 39}`
  - **DRS channel carries non-zero values: {1: 10778, 3: 1, 8: 209, 10: 4, 12: 852, 14: 39}**
    Investigate before assuming it is dead.

- channels new in 2026: `none`
- channels gone in 2026: `none`

Position data channels:
- `Date, Status, X, Y, Z, Source, Time, SessionTime`

Weather columns:
- `Time, AirTemp, Humidity, Pressure, Rainfall, TrackTemp, WindDirection, WindSpeed`

## 5. Race control messages: any sign of the new mechanisms

- **2026**: 183 messages. Keyword hits:
  - `{'overtake': 2, 'deploy': 1, 'ers': 1}`
- **2024**: 59 messages. Keyword hits:
  - `{'drs': 2}`

## 6. Every completed 2026 round, laps only

| R | Event | Status | Lap rows | Drivers |
|---|---|---|---|---|
| 1 | Australian Grand Prix | ok | 1006 | 20 |
| 2 | Chinese Grand Prix | ok | 920 | 18 |
| 3 | Japanese Grand Prix | ok | 1107 | 22 |
| 4 | Miami Grand Prix | ok | 1040 | 22 |
| 5 | Canadian Grand Prix | ok | 1211 | 21 |
| 6 | Monaco Grand Prix | ok | 1452 | 22 |
| 7 | Barcelona Grand Prix | ok | 1236 | 22 |
| 8 | Austrian Grand Prix | ok | 1339 | 22 |
| 9 | British Grand Prix | ok | 1113 | 22 |
| 10 | Belgian Grand Prix | ok | 872 | 22 |
| 11 | Hungarian Grand Prix | ok | 1431 | 22 |
| 12 | Dutch Grand Prix | ok | 1368 | 22 |
| 13 | Italian Grand Prix | ok | 1054 | 22 |
| 14 | Spanish Grand Prix | ok | 1106 | 22 |

- loaded **14/14** completed rounds, **16255** lap rows total

## 7. What this means

- DRS channel **present but dead**. Drop all DRS features.
- No race control mentions of Overtake Mode or active aero. Expect no direct feature for the new mechanism; proxy it with gap-at-detection-point and speed-trap deltas instead.
- 14 completed rounds is the in-era sample. Confirms the small-sample design.

---

## Addendum: follow-up dig, and a correction

Section 7 above got one verdict wrong. The script's example printer only looked
for the literal phrases `overtake mode`, `override`, `active aero` and `boost`,
so the two plain `overtake` hits were counted but never displayed, and the
verdict logic used the same narrow list. Fixed in `recon.py`; the finding below
is what a correct run reports.

### Overtake Mode is observable after all

Race control on 2026 R14 carries:

```
lap 1   OVERTAKE DISABLED
lap 1   OVERTAKE ENABLED
```

This is the exact structural analogue of the old `DRS ENABLED` / `DRS DISABLED`
pair, with a populated `Lap` column. So the 2026 replacement mechanism is **not**
invisible. It is not a telemetry channel, but it is a lap-stamped event stream,
which is enough to derive an `overtake_mode_enabled` state per lap.

That is a real Tier 1 feature rather than the proxy section 7 assumed.

### Two keyword hits were noise

- `deploy` matched `VSC DEPLOYED`, which is safety car, not energy deployment.
- `ers` matched `HOLDERS` inside `ALL PASS HOLDERS MAY ACCESS THE PIT LANE`.

Both fixed by word-boundary matching. There is **no** energy or ERS state exposed.

### Race control schema

```
Time, Category, Message, Status, Flag, Scope, Sector, RacingNumber, Lap
```

2026 R14 categories: `Flag` 108, `Other` 73, `SafetyCar` 2. The jump from 59
messages in 2024 R16 to 183 in 2026 R14 is mostly `Flag` volume, not new
message types.

### Lap-level fields confirmed for extraction

- `Position` populated on 1105 of 1106 rows, so order-flip detection works.
- `TrackStatus` uses the same concatenated-code format as 2022-2025 (`1`, `12`, `126`, `21`, `26`, `671`).
- Compounds present as usual (`SOFT`, `MEDIUM`, `HARD` in this dry race).
- Speed traps `SpeedI1`, `SpeedI2`, `SpeedFL`, `SpeedST` are lap-level columns,
  so **car telemetry is not required** for the planned feature set. Only laps,
  weather and race control are needed.

### Grid composition

11 teams, 22 cars: Alpine, Aston Martin, **Audi**, **Cadillac**, Ferrari, Haas,
McLaren, Mercedes, Racing Bulls, Red Bull Racing, Williams.

Audi and Cadillac have no pre-2026 history, and the rest carry a 2022-2025
performance level that is no longer true. Confirms that team identity cannot
cross the era boundary as a raw category.

### Cache state

The run left `era2026/.fastf1_cache` at 364MB, holding **laps for all 14 rounds**
plus full data for R14 and the 2024 reference. Laps alone cover most of the
feature set, so the extraction layer can be developed offline against this cache.
Weather and race control are still only cached for R14.
