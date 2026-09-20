"""
Tier-tagged feature builder.

Every feature declares which transfer tier it belongs to, because the transfer
components in Layer 1 operate on tiers rather than on individual names:

``TIER_1``
    Regulation-invariant. Properties of the world, the calendar or the session,
    plus the 2026 mechanism state. Safe to pool across eras.
``TIER_2``
    Same meaning in both eras but a shifted relationship to the target. Usable
    for shape, not for level.

Tier 3 features are simply absent: no DRS, no absolute car speeds, no absolute
lap times, no team identity. Only *deltas* of speed and pace appear, because the
absolute values describe a car that no longer exists.

Nothing here needs telemetry. The speed traps are lap-level columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TIER_1 = "tier1"
TIER_2 = "tier2"

#: Feature name to transfer tier. The single source of truth for Layer 1.
FEATURE_TIERS: dict[str, str] = {
    # ---- race state ----------------------------------------------------
    "lap_number": TIER_1,
    "total_laps": TIER_1,
    "race_progress": TIER_1,
    "laps_remaining": TIER_1,
    "race_phase_opening": TIER_1,
    "race_phase_closing": TIER_1,
    "neutralised": TIER_1,
    "lap1_or_restart_like": TIER_1,
    # the 2026 mechanism, recovered from race control
    "overtake_mode_enabled": TIER_1,
    # ---- weather -------------------------------------------------------
    "air_temp": TIER_1,
    "track_temp": TIER_1,
    "humidity": TIER_1,
    "rainfall": TIER_1,
    "wind_speed": TIER_1,
    # ---- field position ------------------------------------------------
    "attacker_position": TIER_1,
    "defender_position": TIER_1,
    "position_percentile": TIER_1,
    # ---- pair state ----------------------------------------------------
    "gap_ahead": TIER_2,
    "pace_delta": TIER_2,
    "speed_i1_delta": TIER_2,
    "speed_i2_delta": TIER_2,
    "speed_fl_delta": TIER_2,
    "speed_st_delta": TIER_2,
    # ---- pair dynamics over the episode --------------------------------
    "battle_lap": TIER_2,
    "gap_delta_1": TIER_2,
    "gap_delta_2": TIER_2,
    "gap_delta_3": TIER_2,
    "gap_mean_3": TIER_2,
    "gap_min_3": TIER_2,
    "is_closing": TIER_2,
    "closing_laps": TIER_2,
    "closing_rate": TIER_2,
    "pace_delta_mean_3": TIER_2,
    # ---- surrounding traffic -------------------------------------------
    "gap_to_car_ahead": TIER_2,
    "gap_to_car_behind": TIER_2,
    "queue_ahead": TIER_2,
    "gap_pressure_ratio": TIER_2,
    "rear_pressure_ratio": TIER_2,
    # ---- tyre and strategy ---------------------------------------------
    "attacker_tyre_age": TIER_2,
    "defender_tyre_age": TIER_2,
    "tyre_age_difference": TIER_2,
    "attacker_stint": TIER_2,
    "defender_stint": TIER_2,
    "attacker_on_newer_stint": TIER_2,
    "attacker_fresh_tyre": TIER_2,
    "defender_fresh_tyre": TIER_2,
    "compound_advantage": TIER_2,
    "same_compound": TIER_2,
}

#: Pairs that measurement shows are near-duplicates, recorded so the modelling
#: step can decide rather than being handed two collinear columns unknowingly.
#:
#: ``overtake_mode_enabled`` vs ``neutralised`` correlate at **-0.954** across
#: 2026. Overtake Mode is switched off essentially only when racing is already
#: neutralised, so its availability carries almost no information of its own:
#: of 6,669 trainable rows, just 49 are green-track-with-mode-off, and those 49
#: hold 6 events. The mechanism is observable, but its *effect* is not
#: identifiable from availability, because the aid is withdrawn only when passing
#: is forbidden anyway.
#:
#: The feature is kept rather than dropped. Those 49 rows are the explicit
#: disable periods (Monaco laps 57-70, Silverstone 45-52), which are genuinely
#: informative situations and will accumulate across the era. It just should not
#: be credited with signal it does not yet carry.
KNOWN_COLLINEAR: list[tuple[str, str, float]] = [
    ("overtake_mode_enabled", "neutralised", -0.954),
]

#: Relative tyre pace, softest fastest. Used only for a *difference*, so the
#: absolute scale does not matter and the 2026 construction change is tolerable.
COMPOUND_PACE = {"SOFT": 3.0, "MEDIUM": 2.0, "HARD": 1.0, "INTERMEDIATE": 0.5, "WET": 0.0}

#: Gap within which a car ahead counts as part of a queue. One second is the
#: Overtake Mode activation range, so the old DRS-train concept survives with
#: the same threshold for a different reason.
QUEUE_GAP = 1.0


#: Snapshot of the per-race features, taken before any later module extends the
#: registry. build_features is checked against this, not the live registry.
BASE_FEATURES: tuple[str, ...] = tuple(FEATURE_TIERS)

#: Features held constant for a whole grand prix. Measurement showed the model
#: uses them as circuit proxies rather than as weather: excluding them raises
#: PR-AUC from 0.489 to 0.515. That is memorisation of which race it is looking
#: at, which cannot generalise to a circuit it has not seen, so they are kept in
#: the registry (the dashboard and drift report still use them) but excluded
#: from what the model trains on.
#:
#: circuit_* features are deliberately NOT in this list. They are also constant
#: within a race, but they encode validated physical character rather than an
#: incidental fingerprint.
RACE_CONSTANT: tuple[str, ...] = (
    "total_laps", "air_temp", "track_temp", "humidity", "rainfall",
    "attacker_prior_rounds", "defender_prior_rounds",
)


def model_features() -> list[str]:
    """What the model trains on: everything declared, minus the race-constant
    proxies. Evaluated live so modules registering later are included."""
    return [name for name in FEATURE_TIERS if name not in RACE_CONSTANT]


def features_by_tier(tier: str) -> list[str]:
    return [name for name, value in FEATURE_TIERS.items() if value == tier]


def all_features() -> list[str]:
    return list(FEATURE_TIERS)


def _grid(laps: pd.DataFrame, column: str, aggfunc="first") -> pd.DataFrame:
    if column not in laps.columns:
        raise KeyError(
            f"lap column {column!r} is missing; real FastF1 sessions always carry it, "
            f"so this is a malformed session or an incomplete test fixture. "
            f"available: {sorted(laps.columns)}"
        )
    return laps.pivot_table(index="LapNumber", columns="Driver", values=column, aggfunc=aggfunc)


def _seconds(grid: pd.DataFrame) -> pd.DataFrame:
    return grid.apply(lambda col: pd.to_timedelta(col).dt.total_seconds())


def _weather_by_lap(session) -> pd.DataFrame:
    """Field-wide weather at each lap, matched on session time."""
    laps, weather = session.laps, session.weather_data
    columns = ["air_temp", "track_temp", "humidity", "rainfall", "wind_speed"]
    lap_index = pd.Index(sorted(laps["LapNumber"].dropna().astype(int).unique()), name="LapNumber")
    if weather is None or not len(weather):
        return pd.DataFrame(0.0, index=lap_index, columns=columns)

    lap_time = _seconds(_grid(laps, "Time")).median(axis=1)
    wx = weather.copy()
    wx["_t"] = pd.to_timedelta(wx["Time"]).dt.total_seconds()
    wx = wx.sort_values("_t")
    target = pd.DataFrame({"_t": lap_time.reindex(lap_index).to_numpy()}, index=lap_index)
    merged = pd.merge_asof(
        target.reset_index().sort_values("_t"), wx, on="_t", direction="nearest"
    ).set_index("LapNumber")
    out = pd.DataFrame(index=lap_index)
    out["air_temp"] = merged.get("AirTemp", 0.0)
    out["track_temp"] = merged.get("TrackTemp", 0.0)
    out["humidity"] = merged.get("Humidity", 0.0)
    out["rainfall"] = merged.get("Rainfall", 0.0).astype(float)
    out["wind_speed"] = merged.get("WindSpeed", 0.0)
    return out.astype(float).fillna(0.0)


def _traffic_by_lap(laps: pd.DataFrame) -> pd.DataFrame:
    """For each (lap, driver): gap to the car ahead, gap behind, and queue size."""
    positions = _grid(laps, "Position")
    times = _seconds(_grid(laps, "Time"))
    rows: list[dict] = []
    for lap in positions.index:
        pos, tim = positions.loc[lap], times.loc[lap]
        present = pos.notna() & tim.notna()
        if not present.any():
            continue
        order = pos[present].sort_values()
        drivers = list(order.index)
        gaps = [np.nan] + [float(tim[b] - tim[a]) for a, b in zip(drivers, drivers[1:])]
        for i, driver in enumerate(drivers):
            ahead = gaps[i]
            behind = gaps[i + 1] if i + 1 < len(gaps) else np.nan
            # how many consecutive cars ahead are each within QUEUE_GAP
            queue = 0
            for j in range(i, 0, -1):
                if pd.notna(gaps[j]) and gaps[j] <= QUEUE_GAP:
                    queue += 1
                else:
                    break
            rows.append({
                "LapNumber": int(lap), "Driver": driver,
                "gap_ahead_own": ahead, "gap_behind_own": behind, "queue": queue,
            })
    return pd.DataFrame(rows).set_index(["LapNumber", "Driver"])


def build_features(session, rows: pd.DataFrame) -> pd.DataFrame:
    """Attach every declared feature to a frame of hazard rows."""
    if rows is None or rows.empty:
        return rows

    laps = session.laps
    out = rows.copy()

    lap_time = _seconds(_grid(laps, "LapTime"))
    tyre_life = _grid(laps, "TyreLife")
    stint = _grid(laps, "Stint")
    fresh = _grid(laps, "FreshTyre")
    compound = _grid(laps, "Compound")
    traps = {name: _grid(laps, column) for name, column in
             [("i1", "SpeedI1"), ("i2", "SpeedI2"), ("fl", "SpeedFL"), ("st", "SpeedST")]}
    traffic = _traffic_by_lap(laps)
    weather = _weather_by_lap(session)
    field_size = laps.groupby("LapNumber")["Driver"].nunique()

    def pick(grid: pd.DataFrame, lap: int, driver: str, default=np.nan):
        try:
            value = grid.loc[lap, driver]
        except KeyError:
            return default
        return default if pd.isna(value) else value

    # ---- race state --------------------------------------------------
    out["race_progress"] = out["lap_number"] / out["total_laps"].clip(lower=1)
    out["laps_remaining"] = (out["total_laps"] - out["lap_number"]).clip(lower=0)
    out["race_phase_opening"] = (out["race_progress"] <= 0.25).astype(int)
    out["race_phase_closing"] = (out["race_progress"] >= 0.75).astype(int)
    out["lap1_or_restart_like"] = (out["lap_number"] <= 2).astype(int)
    out["neutralised"] = out["neutralised"].astype(int)
    out["overtake_mode_enabled"] = out["overtake_mode_enabled"].astype(int)
    out["position_percentile"] = out["attacker_position"] / out["lap_number"].map(field_size).clip(lower=1)

    for column in weather.columns:
        out[column] = out["lap_number"].map(weather[column]).astype(float).fillna(0.0)

    # ---- per-row pair state ------------------------------------------
    records: list[dict] = []
    for row in out.itertuples():
        lap, attacker, defender = row.lap_number, row.attacker, row.defender
        record: dict[str, float] = {}

        a_time, d_time = pick(lap_time, lap, attacker), pick(lap_time, lap, defender)
        record["pace_delta"] = (a_time - d_time) if pd.notna(a_time) and pd.notna(d_time) else 0.0

        for name, grid in traps.items():
            a, d = pick(grid, lap, attacker), pick(grid, lap, defender)
            record[f"speed_{name}_delta"] = float(a - d) if pd.notna(a) and pd.notna(d) else 0.0

        a_age, d_age = pick(tyre_life, lap, attacker, 0.0), pick(tyre_life, lap, defender, 0.0)
        record["attacker_tyre_age"] = float(a_age)
        record["defender_tyre_age"] = float(d_age)
        record["tyre_age_difference"] = float(a_age - d_age)

        a_stint, d_stint = pick(stint, lap, attacker, 1.0), pick(stint, lap, defender, 1.0)
        record["attacker_stint"] = float(a_stint)
        record["defender_stint"] = float(d_stint)
        record["attacker_on_newer_stint"] = int(a_stint > d_stint)

        record["attacker_fresh_tyre"] = int(bool(pick(fresh, lap, attacker, False)))
        record["defender_fresh_tyre"] = int(bool(pick(fresh, lap, defender, False)))

        a_comp = str(pick(compound, lap, attacker, "")).upper()
        d_comp = str(pick(compound, lap, defender, "")).upper()
        record["compound_advantage"] = float(
            COMPOUND_PACE.get(a_comp, 1.0) - COMPOUND_PACE.get(d_comp, 1.0)
        )
        record["same_compound"] = int(a_comp == d_comp and a_comp != "")

        key_d, key_a = (lap, defender), (lap, attacker)
        record["gap_to_car_ahead"] = float(
            traffic["gap_ahead_own"].get(key_d, np.nan) if key_d in traffic.index else np.nan
        )
        record["gap_to_car_behind"] = float(
            traffic["gap_behind_own"].get(key_a, np.nan) if key_a in traffic.index else np.nan
        )
        record["queue_ahead"] = float(traffic["queue"].get(key_d, 0) if key_d in traffic.index else 0)
        records.append(record)

    out = pd.concat([out, pd.DataFrame(records, index=out.index)], axis=1)

    out["gap_to_car_ahead"] = out["gap_to_car_ahead"].fillna(99.0)
    out["gap_to_car_behind"] = out["gap_to_car_behind"].fillna(99.0)
    safe_gap = out["gap_ahead"].clip(lower=0.05)
    out["gap_pressure_ratio"] = out["gap_to_car_ahead"] / safe_gap
    out["rear_pressure_ratio"] = out["gap_to_car_behind"] / safe_gap

    # ---- dynamics within the episode ---------------------------------
    out = out.sort_values(["episode_id", "lap_number"], kind="mergesort")
    grouped = out.groupby("episode_id", sort=False)["gap_ahead"]
    for lag in (1, 2, 3):
        out[f"gap_delta_{lag}"] = grouped.diff(lag).fillna(0.0)
    out["gap_mean_3"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["gap_min_3"] = grouped.transform(lambda s: s.rolling(3, min_periods=1).min())
    out["closing_rate"] = -out["gap_delta_1"]
    out["is_closing"] = (out["gap_delta_1"] < 0).astype(int)
    out["closing_laps"] = (
        out.groupby("episode_id", sort=False)["is_closing"]
        .transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    out["pace_delta_mean_3"] = (
        out.groupby("episode_id", sort=False)["pace_delta"]
        .transform(lambda s: s.rolling(3, min_periods=1).mean())
    )

    out = out.sort_values(["lap_number", "attacker_position"], kind="mergesort").reset_index(drop=True)

    # Only the per-race features are this function's responsibility. Form
    # features are attached later, once the whole season is available.
    missing = [name for name in BASE_FEATURES if name not in out.columns]
    if missing:
        raise RuntimeError(f"declared features not produced: {missing}")
    return out
