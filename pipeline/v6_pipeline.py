from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from typing import Iterable

import fastf1
import numpy as np
import pandas as pd

from . import fastf1_utils as ffu
from . import track_info


DEFAULT_CANDIDATE_GAP = 3.0
DEFAULT_POSITIVE_FILTERS = {
    "exclude_pit_related": True,
    "exclude_lap1": False,
    "exclude_safety_car": False,
    "exclude_yellow_flag": False,
    "require_accurate_timing": False,
    "adjacency_rule": "none",
    "max_position_gain": None,
}
DEFAULT_HORIZONS = (1, 2, 3)

TYRE_PACE_RANK = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 1.5,
    "WET": 2,
    "UNKNOWN": 1,
}
TYRE_CLIFF = {"SOFT": 18, "MEDIUM": 28, "HARD": 38, "INTERMEDIATE": 30, "WET": 25}


def race_event_rows(year: int) -> pd.DataFrame:
    schedule = fastf1.get_event_schedule(year)
    out = schedule.copy()
    out = out[out["RoundNumber"].fillna(0).astype(int) > 0].copy()
    return out.reset_index(drop=True)


def load_race_session(year: int, event_name: str, cache_path: str | None = None):
    return ffu.load_session(year=year, gp=event_name, identifier="R", cache_path=cache_path)


def session_year(session, fallback_year: int) -> int:
    try:
        val = session.event.get("Year")
        if val is not None and not pd.isna(val):
            return int(val)
    except Exception:
        pass
    try:
        dt = session.event.get("EventDate")
        if dt is not None and not pd.isna(dt):
            return int(pd.Timestamp(dt).year)
    except Exception:
        pass
    return int(fallback_year)


def lap_frame(session) -> pd.DataFrame:
    cols = [
        "Driver",
        "LapNumber",
        "Position",
        "LapTime",
        "PitInTime",
        "PitOutTime",
        "LapStartTime",
        "Time",
        "TrackStatus",
        "IsAccurate",
        "SpeedI1",
        "SpeedI2",
        "SpeedFL",
        "SpeedST",
        "Compound",
        "TyreLife",
        "Stint",
        "FreshTyre",
        "Sector",
        "Sector1Time",
        "Sector2Time",
        "Sector3Time",
    ]
    present = [c for c in cols if c in session.laps.columns]
    df = session.laps[present].copy()
    df = df[df["Driver"].notna() & df["LapNumber"].notna() & df["Position"].notna()].copy()
    df["Driver"] = df["Driver"].astype(str)
    df["LapNumber"] = df["LapNumber"].astype(int)
    df["Position"] = df["Position"].astype(int)
    if "IsAccurate" in df.columns:
        df["IsAccurate"] = df["IsAccurate"].fillna(False).astype(bool)
    else:
        df["IsAccurate"] = True
    return df.sort_values(["LapNumber", "Position", "Driver"]).reset_index(drop=True)


def pit_related(row: pd.Series) -> bool:
    return pd.notna(row.get("PitInTime")) or pd.notna(row.get("PitOutTime"))


def extract_raw_overtake_candidates(session, fallback_year: int) -> pd.DataFrame:
    laps = lap_frame(session)
    if laps.empty:
        return pd.DataFrame()

    year = session_year(session, fallback_year)
    race_name = str(session.event.get("EventName") or "Unknown")
    round_number = ffu._safe_int(session.event.get("RoundNumber"), 0)
    event_format = str(session.event.get("EventFormat") or "")
    max_lap = int(laps["LapNumber"].max())

    rows: list[dict] = []
    for lap in range(1, max_lap):
        cur = laps[laps["LapNumber"] == lap].set_index("Driver", drop=False)
        nxt = laps[laps["LapNumber"] == lap + 1].set_index("Driver", drop=False)
        shared = sorted(set(cur.index) & set(nxt.index))
        if len(shared) < 2:
            continue

        safety_car, yellow_flag = ffu.detect_safety_car_and_flags(session, lap + 1)

        for i, a in enumerate(shared):
            for b in shared[i + 1 :]:
                a0, b0 = cur.loc[a], cur.loc[b]
                a1, b1 = nxt.loc[a], nxt.loc[b]
                pa0, pb0 = int(a0["Position"]), int(b0["Position"])
                pa1, pb1 = int(a1["Position"]), int(b1["Position"])

                if pa0 == pb0 or pa1 == pb1:
                    continue
                if (pa0 - pb0) * (pa1 - pb1) >= 0:
                    continue

                if pa0 > pb0 and pa1 < pb1:
                    overtaker, overtaken = a, b
                    ov0, ov1, od0, od1 = a0, a1, b0, b1
                    overtaker_prev, overtaker_next = pa0, pa1
                    overtaken_prev, overtaken_next = pb0, pb1
                elif pb0 > pa0 and pb1 < pa1:
                    overtaker, overtaken = b, a
                    ov0, ov1, od0, od1 = b0, b1, a0, a1
                    overtaker_prev, overtaker_next = pb0, pb1
                    overtaken_prev, overtaken_next = pa0, pa1
                else:
                    continue

                rows.append(
                    {
                        "year": year,
                        "race_name": race_name,
                        "round_number": round_number,
                        "event_format": event_format,
                        "lap_number": lap + 1,
                        "overtaker": overtaker,
                        "overtaken": overtaken,
                        "overtaker_prev_pos": overtaker_prev,
                        "overtaker_next_pos": overtaker_next,
                        "overtaken_prev_pos": overtaken_prev,
                        "overtaken_next_pos": overtaken_next,
                        "position_gain": int(overtaker_prev - overtaker_next),
                        "consecutive_before": bool(overtaker_prev == overtaken_prev + 1),
                        "consecutive_after": bool(overtaken_next == overtaker_next + 1),
                        "pit_related": bool(
                            pit_related(ov0) or pit_related(ov1) or pit_related(od0) or pit_related(od1)
                        ),
                        "accurate_timing": bool(
                            ov0.get("IsAccurate", True)
                            and ov1.get("IsAccurate", True)
                            and od0.get("IsAccurate", True)
                            and od1.get("IsAccurate", True)
                        ),
                        "safety_car": bool(safety_car),
                        "yellow_flag": bool(yellow_flag),
                        "lap1_or_restart_like": bool(lap + 1 <= 2),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["year", "round_number", "lap_number", "overtaker", "overtaken"]
    ).reset_index(drop=True)


def apply_raw_overtake_filters(
    df: pd.DataFrame,
    *,
    exclude_pit_related: bool = True,
    exclude_lap1: bool = False,
    exclude_safety_car: bool = False,
    exclude_yellow_flag: bool = False,
    require_accurate_timing: bool = False,
    adjacency_rule: str = "none",
    max_position_gain: int | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    out = df.copy()
    if exclude_pit_related:
        out = out[~out["pit_related"]]
    if exclude_lap1:
        out = out[~out["lap1_or_restart_like"]]
    if exclude_safety_car:
        out = out[~out["safety_car"]]
    if exclude_yellow_flag:
        out = out[~out["yellow_flag"]]
    if require_accurate_timing:
        out = out[out["accurate_timing"]]
    if adjacency_rule == "before":
        out = out[out["consecutive_before"]]
    elif adjacency_rule == "after":
        out = out[out["consecutive_after"]]
    elif adjacency_rule == "either":
        out = out[out["consecutive_before"] | out["consecutive_after"]]
    if max_position_gain is not None:
        out = out[out["position_gain"] <= max_position_gain]
    return out.reset_index(drop=True)


def _safe_total_seconds(value) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        if hasattr(value, "total_seconds"):
            return float(value.total_seconds())
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_date_str(session) -> str:
    dt = session.event.get("EventDate")
    if dt is not None and not pd.isna(dt):
        return str(pd.Timestamp(dt).date())
    return ""


def _build_positive_lookup(filtered_overtakes: pd.DataFrame) -> dict[tuple[str, str], list[int]]:
    lookup: dict[tuple[str, str], list[int]] = {}
    if filtered_overtakes.empty:
        return lookup
    for (overtaker, overtaken), group in filtered_overtakes.groupby(["overtaker", "overtaken"], sort=False):
        lookup[(str(overtaker), str(overtaken))] = sorted(group["lap_number"].astype(int).tolist())
    return lookup


def _has_event_within(laps: list[int], current_lap: int, horizon: int) -> bool:
    idx = bisect_left(laps, current_lap + 1)
    if idx >= len(laps):
        return False
    return laps[idx] <= current_lap + horizon


def _prior_event_count(laps: list[int], current_lap: int) -> int:
    return bisect_left(laps, current_lap)


def _gap_to_leader_seconds(row: pd.Series) -> float:
    start = _safe_total_seconds(row.get("LapStartTime"))
    if start is None:
        return 99.0
    return float(start)


def build_v6_candidates(
    session,
    fallback_year: int,
    filtered_overtakes: pd.DataFrame,
    *,
    gap_threshold: float = DEFAULT_CANDIDATE_GAP,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    laps = lap_frame(session)
    if laps.empty:
        return pd.DataFrame()

    lap_position_gap = ffu.build_position_and_gap_map(session)
    weather_lookup = ffu.build_weather_lookup(session)
    total_laps = ffu.get_total_laps(session)
    year = session_year(session, fallback_year)
    race_name = str(session.event.get("EventName") or "Unknown")
    round_number = ffu._safe_int(session.event.get("RoundNumber"), 0)
    event_date = _event_date_str(session)
    track = str(session.event.get("Location") or "").upper()
    positive_lookup = _build_positive_lookup(filtered_overtakes)
    max_horizon = max(int(h) for h in horizons)

    by_lap = {
        lap_number: frame.set_index("Driver", drop=False)
        for lap_number, frame in laps.groupby("LapNumber", sort=True)
    }

    rows: list[dict] = []
    for lap_number in range(1, int(laps["LapNumber"].max()) + 1):
        cur = by_lap.get(lap_number)
        if cur is None or len(cur) < 2:
            continue

        ordered = cur.reset_index(drop=True).sort_values(["Position", "Driver"])
        lap_weather = weather_lookup.get(lap_number, ffu.DEFAULT_WEATHER)
        safety_car, yellow_flag = ffu.detect_safety_car_and_flags(session, lap_number)
        lap_data = lap_position_gap.get(lap_number, {})

        for idx in range(len(ordered) - 1):
            defender_lap = ordered.iloc[idx]
            attacker_lap = ordered.iloc[idx + 1]

            defender = str(defender_lap["Driver"])
            attacker = str(attacker_lap["Driver"])
            defender_pos = int(defender_lap["Position"])
            attacker_pos = int(attacker_lap["Position"])
            if attacker_pos != defender_pos + 1:
                continue

            if not ffu.is_on_track(defender_lap) or not ffu.is_on_track(attacker_lap):
                continue
            if defender not in lap_data or attacker not in lap_data:
                continue

            actual_gap = ffu.calculate_actual_gap(lap_data[defender], lap_data[attacker])
            if actual_gap is None or actual_gap > gap_threshold:
                continue

            att_speeds = ffu.get_speed_trap_data(attacker_lap)
            def_speeds = ffu.get_speed_trap_data(defender_lap)
            attacker_laptime = _safe_total_seconds(attacker_lap.get("LapTime")) or 0.0
            defender_laptime = _safe_total_seconds(defender_lap.get("LapTime")) or 0.0
            pace_delta = defender_laptime - attacker_laptime
            attacker_tyre = (
                str(attacker_lap.get("Compound", "UNKNOWN"))
                if pd.notna(attacker_lap.get("Compound"))
                else "UNKNOWN"
            )
            defender_tyre = (
                str(defender_lap.get("Compound", "UNKNOWN"))
                if pd.notna(defender_lap.get("Compound"))
                else "UNKNOWN"
            )
            attacker_tyre_age = ffu._safe_int(attacker_lap.get("TyreLife"), 0)
            defender_tyre_age = ffu._safe_int(defender_lap.get("TyreLife"), 0)
            attacker_stint, attacker_fresh = ffu.get_stint_info(attacker_lap)
            defender_stint, defender_fresh = ffu.get_stint_info(defender_lap)

            sector = ffu._safe_int(attacker_lap.get("Sector"), 1)
            is_in_drs_zone, drs_zone_length = track_info.get_drs_zone_info(track, sector)
            attacker_quali = ffu.get_driver_qualification_rank(session, attacker)
            defender_quali = ffu.get_driver_qualification_rank(session, defender)
            att_s1, att_s2, att_s3 = ffu.get_sector_times(attacker_lap)
            def_s1, def_s2, def_s3 = ffu.get_sector_times(defender_lap)
            sector1_delta = def_s1 - att_s1
            sector2_delta = def_s2 - att_s2
            sector3_delta = def_s3 - att_s3
            strongest_sector = ffu.strongest_sector_index([sector1_delta, sector2_delta, sector3_delta])
            attacker_team = ffu.get_driver_team(session, attacker)
            defender_team = ffu.get_driver_team(session, defender)

            pair_event_laps = positive_lookup.get((attacker, defender), [])
            labels = {
                f"overtake_within_{h}": _has_event_within(pair_event_laps, lap_number, int(h))
                for h in horizons
            }
            label = labels.get(f"overtake_within_{max_horizon}", False)

            rows.append(
                {
                    "year": year,
                    "race_name": race_name,
                    "round_number": round_number,
                    "event_date": event_date,
                    "lap_number": lap_number,
                    "total_laps": total_laps,
                    "race_progress": round(lap_number / total_laps, 4) if total_laps > 0 else 0.0,
                    "attacker": attacker,
                    "defender": defender,
                    "attacker_position": attacker_pos,
                    "defender_position": defender_pos,
                    "attacker_lap_time": attacker_laptime,
                    "defender_lap_time": defender_laptime,
                    "gap_ahead": float(actual_gap),
                    "gap_to_leader": _gap_to_leader_seconds(attacker_lap),
                    "pace_delta": float(pace_delta),
                    "attacker_speed_i1": att_speeds["speed_i1"],
                    "defender_speed_i1": def_speeds["speed_i1"],
                    "attacker_speed_i2": att_speeds["speed_i2"],
                    "defender_speed_i2": def_speeds["speed_i2"],
                    "attacker_finish_line_speed": att_speeds["finish_line_speed"],
                    "defender_finish_line_speed": def_speeds["finish_line_speed"],
                    "attacker_straight_speed": att_speeds["straight_speed"],
                    "defender_straight_speed": def_speeds["straight_speed"],
                    "speed_i1_delta": att_speeds["speed_i1"] - def_speeds["speed_i1"],
                    "speed_i2_delta": att_speeds["speed_i2"] - def_speeds["speed_i2"],
                    "speed_fl_delta": att_speeds["finish_line_speed"] - def_speeds["finish_line_speed"],
                    "speed_st_delta": att_speeds["straight_speed"] - def_speeds["straight_speed"],
                    "safety_car": bool(safety_car),
                    "yellow_flag": bool(yellow_flag),
                    "attacker_tyre_compound": attacker_tyre,
                    "defender_tyre_compound": defender_tyre,
                    "attacker_tyre_age": attacker_tyre_age,
                    "defender_tyre_age": defender_tyre_age,
                    "tyre_age_difference": attacker_tyre_age - defender_tyre_age,
                    "attacker_stint": attacker_stint,
                    "defender_stint": defender_stint,
                    "attacker_fresh_tyre": attacker_fresh,
                    "defender_fresh_tyre": defender_fresh,
                    "pit_stop_involved": bool(
                        ffu.is_next_lap_pit(session, attacker, lap_number)
                        or ffu.is_next_lap_pit(session, defender, lap_number)
                        or ffu.is_next_lap_pit(session, attacker, lap_number + 1)
                        or ffu.is_next_lap_pit(session, defender, lap_number + 1)
                    ),
                    "track": track,
                    "sector": sector,
                    "sector_type": track_info.get_sector_type(track),
                    "is_in_drs_zone": is_in_drs_zone,
                    "drs_zone_length": drs_zone_length,
                    "track_type": track_info.get_track_type(track),
                    "attacker_qualification_rank": attacker_quali,
                    "defender_qualification_rank": defender_quali,
                    "air_temp": lap_weather["air_temp"],
                    "track_temp": lap_weather["track_temp"],
                    "humidity": lap_weather["humidity"],
                    "rainfall": lap_weather["rainfall"],
                    "wind_speed": lap_weather["wind_speed"],
                    "sector1_delta": sector1_delta,
                    "sector2_delta": sector2_delta,
                    "sector3_delta": sector3_delta,
                    "strongest_sector": strongest_sector,
                    "attacker_team": attacker_team,
                    "defender_team": defender_team,
                    "same_team": attacker_team == defender_team,
                    "gap_to_car_ahead": (
                        ffu.gap_to_position(lap_data, defender_pos, defender_pos - 1)
                        if defender_pos > 1
                        else 99.0
                    ),
                    "gap_to_car_behind": ffu.gap_to_position(lap_data, attacker_pos, attacker_pos + 1),
                    "drs_train_size": ffu.count_cars_within_drs(lap_data, attacker_pos, threshold=1.0),
                    "race_phase": (
                        "opening"
                        if lap_number / total_laps <= 0.25
                        else "middle"
                        if lap_number / total_laps <= 0.75
                        else "closing"
                    )
                    if total_laps > 0
                    else "middle",
                    "stint_phase": (
                        "fresh"
                        if attacker_tyre_age <= 5
                        else "mid"
                        if attacker_tyre_age <= 15
                        else "degraded"
                        if attacker_tyre_age <= 25
                        else "cliff"
                    ),
                    "accurate_timing": bool(
                        attacker_lap.get("IsAccurate", True) and defender_lap.get("IsAccurate", True)
                    ),
                    "lap1_or_restart_like": bool(lap_number <= 2),
                    "candidate_gap_threshold": float(gap_threshold),
                    "prior_pair_overtakes": int(_prior_event_count(pair_event_laps, lap_number)),
                    "label": bool(label),
                    "overtake_next_lap": bool(labels.get("overtake_within_1", False)),
                    "overtake_within_2": bool(labels.get("overtake_within_2", False)),
                    "overtake_within_3": bool(labels.get("overtake_within_3", False)),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["year", "round_number", "lap_number", "attacker_position", "attacker", "defender"]
    ).reset_index(drop=True)


def engineer_v6_features(df: pd.DataFrame, filtered_overtakes: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.sort_values(
        ["year", "round_number", "race_name", "attacker", "defender", "lap_number"],
        kind="mergesort",
    ).reset_index(drop=True)

    out["_pair"] = (
        out["attacker"].astype(str)
        + "_vs_"
        + out["defender"].astype(str)
        + "_"
        + out["race_name"].astype(str)
        + "_"
        + out["year"].astype(str)
    )
    out["_lap_gap"] = out.groupby("_pair")["lap_number"].diff().fillna(99)
    out["_break"] = (out["_lap_gap"] != 1).astype(int)
    out["_seq"] = out.groupby("_pair")["_break"].cumsum()
    out["_bg"] = out["_pair"] + "_" + out["_seq"].astype(str)

    grp_gap = out.groupby("_bg")["gap_ahead"]
    grp_pace = out.groupby("_bg")["pace_delta"]
    grp_speed_st = out.groupby("_bg")["speed_st_delta"]

    out["gap_delta_1"] = grp_gap.diff(1).fillna(0.0)
    out["gap_delta_2"] = grp_gap.diff(2).fillna(0.0)
    out["gap_delta_3"] = grp_gap.diff(3).fillna(0.0)
    out["is_closing"] = (out["gap_delta_1"] < 0).astype(int)
    out["closing_laps"] = (
        out.groupby("_bg")["is_closing"].transform(lambda s: s.rolling(3, min_periods=1).sum())
    )
    out["gap_mean_3"] = grp_gap.transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["gap_min_3"] = grp_gap.transform(lambda s: s.rolling(3, min_periods=1).min())
    out["pace_delta_avg_3"] = grp_pace.transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["pace_delta_std_3"] = grp_pace.transform(lambda s: s.rolling(3, min_periods=1).std()).fillna(0.0)
    out["speed_st_delta_avg_3"] = grp_speed_st.transform(lambda s: s.rolling(3, min_periods=1).mean())
    out["battle_duration"] = out.groupby("_bg").cumcount() + 1

    out["qualification_rank_difference"] = (
        out["attacker_qualification_rank"] - out["defender_qualification_rank"]
    )
    att_pace = out["attacker_tyre_compound"].astype(str).str.upper().map(TYRE_PACE_RANK).fillna(1.0)
    def_pace = out["defender_tyre_compound"].astype(str).str.upper().map(TYRE_PACE_RANK).fillna(1.0)
    out["compound_advantage"] = def_pace - att_pace
    cliff = out["defender_tyre_compound"].astype(str).str.upper().map(TYRE_CLIFF).fillna(28)
    out["tyre_cliff_risk"] = (out["defender_tyre_age"] > cliff).astype(int)
    out["attacker_on_newer_stint"] = (out["attacker_stint"] > out["defender_stint"]).astype(int)
    out["gap_pressure_ratio"] = out["gap_to_car_ahead"] / out["gap_ahead"].clip(lower=0.05)
    out["rear_pressure_ratio"] = out["gap_to_car_behind"] / out["gap_ahead"].clip(lower=0.05)
    out["closing_rate"] = out["gap_delta_1"]
    out["laps_remaining"] = (out["total_laps"] - out["lap_number"]).clip(lower=0)

    if filtered_overtakes.empty:
        out["overtakes_so_far"] = 0
    else:
        race_to_event_laps: dict[tuple[int, str], list[int]] = {}
        for key, group in filtered_overtakes.groupby(["year", "race_name"], sort=False):
            race_to_event_laps[(int(key[0]), str(key[1]))] = sorted(group["lap_number"].astype(int).tolist())
        out["overtakes_so_far"] = [
            _prior_event_count(race_to_event_laps.get((int(y), str(rn)), []), int(lap))
            for y, rn, lap in zip(out["year"], out["race_name"], out["lap_number"])
        ]

    out.drop(columns=["_pair", "_lap_gap", "_break", "_seq", "_bg"], inplace=True, errors="ignore")
    return out


def generate_v6_dataset(
    years: list[int],
    *,
    cache_path: str | None = None,
    output_dir: str | Path | None = None,
    gap_threshold: float = DEFAULT_CANDIDATE_GAP,
    positive_filters: dict | None = None,
) -> dict[str, pd.DataFrame]:
    positive_filters = positive_filters or DEFAULT_POSITIVE_FILTERS
    scenario_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    filtered_frames: list[pd.DataFrame] = []
    audit_frames: list[pd.DataFrame] = []

    for year in years:
        season_events = race_event_rows(year)
        season_scenarios: list[pd.DataFrame] = []
        season_raw: list[pd.DataFrame] = []
        season_filtered: list[pd.DataFrame] = []
        audit_rows: list[dict] = []

        print(f"\nProcessing v6 season {year}")
        for _, event in season_events.iterrows():
            event_name = str(event["EventName"])
            event_format = str(event.get("EventFormat") or "")
            round_number = int(event.get("RoundNumber") or 0)
            print(f"  {event_name}")
            try:
                session = load_race_session(year, event_name, cache_path=cache_path)
                raw = extract_raw_overtake_candidates(session, year)
                filtered = apply_raw_overtake_filters(raw, **positive_filters)
                scenarios = build_v6_candidates(
                    session,
                    year,
                    filtered,
                    gap_threshold=gap_threshold,
                    horizons=DEFAULT_HORIZONS,
                )

                audit_rows.append(
                    {
                        "year": year,
                        "round_number": round_number,
                        "event_name": event_name,
                        "event_format": event_format,
                        "status": "ok",
                        "raw_overtakes": int(len(raw)),
                        "filtered_overtakes": int(len(filtered)),
                        "candidate_rows": int(len(scenarios)),
                        "positive_rows": int(scenarios["label"].sum()) if not scenarios.empty else 0,
                        "error": None,
                    }
                )
                if not raw.empty:
                    season_raw.append(raw)
                if not filtered.empty:
                    season_filtered.append(filtered)
                if not scenarios.empty:
                    season_scenarios.append(scenarios)
                print(
                    f"    raw={len(raw):,} filtered={len(filtered):,} scenarios={len(scenarios):,}"
                )
            except Exception as exc:
                audit_rows.append(
                    {
                        "year": year,
                        "round_number": round_number,
                        "event_name": event_name,
                        "event_format": event_format,
                        "status": "error",
                        "raw_overtakes": 0,
                        "filtered_overtakes": 0,
                        "candidate_rows": 0,
                        "positive_rows": 0,
                        "error": str(exc),
                    }
                )
                print(f"    Skipped: {exc}")

        raw_year = pd.concat(season_raw, ignore_index=True) if season_raw else pd.DataFrame()
        filtered_year = pd.concat(season_filtered, ignore_index=True) if season_filtered else pd.DataFrame()
        scenarios_year = pd.concat(season_scenarios, ignore_index=True) if season_scenarios else pd.DataFrame()
        audit_year = pd.DataFrame(audit_rows).sort_values(["round_number", "event_name"]).reset_index(drop=True)

        raw_frames.append(raw_year)
        filtered_frames.append(filtered_year)
        scenario_frames.append(scenarios_year)
        audit_frames.append(audit_year)

    raw_all = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
    filtered_all = pd.concat(filtered_frames, ignore_index=True) if filtered_frames else pd.DataFrame()
    scenarios_all = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()
    audit_all = pd.concat(audit_frames, ignore_index=True) if audit_frames else pd.DataFrame()

    if not scenarios_all.empty:
        from .driver_features import enrich_driver_features
        from .team_features import enrich_team_features

        scenarios_all = engineer_v6_features(scenarios_all, filtered_all)
        scenarios_all = enrich_driver_features(scenarios_all, label_col="label")
        scenarios_all = enrich_team_features(scenarios_all)

    summary = (
        scenarios_all.groupby("year", as_index=False)
        .agg(
            candidate_rows=("label", "size"),
            positive_rows=("label", "sum"),
            positive_rate=("label", "mean"),
        )
        if not scenarios_all.empty
        else pd.DataFrame(columns=["year", "candidate_rows", "positive_rows", "positive_rate"])
    )

    outputs = {
        "scenarios": scenarios_all,
        "raw_overtakes": raw_all,
        "filtered_overtakes": filtered_all,
        "audit": audit_all,
        "summary": summary,
    }

    if output_dir is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for year in years:
            if not scenarios_all.empty:
                scenarios_all[scenarios_all["year"] == year].to_csv(
                    out_dir / f"scenarios_{year}.csv", index=False
                )
            if not raw_all.empty:
                raw_all[raw_all["year"] == year].to_csv(out_dir / f"raw_overtakes_{year}.csv", index=False)
            if not filtered_all.empty:
                filtered_all[filtered_all["year"] == year].to_csv(
                    out_dir / f"filtered_overtakes_{year}.csv", index=False
                )
            year_audit = audit_all[audit_all["year"] == year] if not audit_all.empty else pd.DataFrame()
            if not year_audit.empty:
                year_audit.to_csv(out_dir / f"audit_{year}.csv", index=False)
        if not summary.empty:
            summary.to_csv(out_dir / "summary.csv", index=False)

    return outputs
