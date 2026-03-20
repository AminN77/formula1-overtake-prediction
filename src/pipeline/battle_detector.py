import pandas as pd
from typing import List

from . import fastf1_utils as ffu
from .models import BattleRecord
from . import track_info


def _timedelta_to_seconds(td) -> float:
    if pd.isna(td):
        return 0.0
    if isinstance(td, pd.Timedelta):
        return td.total_seconds()
    return float(td)


def _attacker_leads_over_defender_at_lap(session, attacker: str, defender: str, lap_n: int) -> bool:
    """True if attacker position number is lower (ahead) than defender at the given lap row."""
    try:
        max_lap = int(session.laps["LapNumber"].max())
    except Exception:
        return False
    if lap_n > max_lap or lap_n < 1:
        return False
    laps = session.laps
    next_lap_data = laps[
        (laps["Driver"].isin([attacker, defender])) & (laps["LapNumber"] == lap_n)
    ]
    if len(next_lap_data) != 2:
        return False
    att_row = next_lap_data[next_lap_data["Driver"] == attacker]
    def_row = next_lap_data[next_lap_data["Driver"] == defender]
    if att_row.empty or def_row.empty:
        return False
    try:
        return int(att_row.iloc[0]["Position"]) < int(def_row.iloc[0]["Position"])
    except (ValueError, TypeError):
        return False


def _overtake_within_horizon(session, attacker: str, defender: str, lap_start: int, horizon: int) -> bool:
    """IP03 §1.2: attacker passes within `horizon` completed laps (1=next lap only)."""
    for k in range(1, horizon + 1):
        if _attacker_leads_over_defender_at_lap(session, attacker, defender, lap_start + k):
            return True
    return False


def detect_battles(session, year: int, race_name: str,
                   gap_threshold: float = 1.0,
                   start_lap: int = 2) -> List[BattleRecord]:
    """
    Detect battles between consecutive-position drivers within *gap_threshold*.

    v3 improvements over v2:
      - gap_ahead uses actual inter-car gap (LapStartTime), not pace difference
      - tyre_age_difference is signed (negative = attacker on fresher tyres)
      - pit_stop_involved flags battles where either driver pits on current/next lap
      - speed delta features (attacker − defender) added
      - pace_delta separated from gap_ahead

    v4 (IP03) additions:
      - overtake_within_2 / overtake_within_3 multi-horizon labels
      - sector1/2/3_delta and strongest_sector (IP03 §3.4)
      - compound_mismatch; round_number & event_date for temporal ordering
    """
    lap_position_gap = ffu.build_position_and_gap_map(session)
    max_lap = max(lap_position_gap.keys())
    battles: List[BattleRecord] = []

    try:
        rn = session.event.get("RoundNumber")
        round_number = int(rn) if rn is not None and not pd.isna(rn) else 0
    except (TypeError, ValueError):
        round_number = 0
    ed = session.event.get("EventDate")
    if ed is not None and not pd.isna(ed):
        event_date = str(pd.Timestamp(ed).date())
    else:
        event_date = ""

    track = session.event['Location']
    total_laps = ffu.get_total_laps(session)
    weather_lookup = ffu.build_weather_lookup(session)

    for lap_number in range(start_lap, max_lap + 1):
        lap_data = lap_position_gap.get(lap_number, {})
        drivers_by_position = sorted(lap_data.items(), key=lambda x: x[1]['position'])

        lap_weather = weather_lookup.get(lap_number, ffu.DEFAULT_WEATHER)

        for i in range(len(drivers_by_position) - 1):
            defender, defender_info = drivers_by_position[i]
            attacker, attacker_info = drivers_by_position[i + 1]

            if attacker_info['position'] != defender_info['position'] + 1:
                continue

            # §1.1: use actual inter-car gap for battle detection
            actual_gap = ffu.calculate_actual_gap(defender_info, attacker_info)
            if actual_gap is None or actual_gap >= gap_threshold:
                continue

            defender_lap = ffu.get_driver_info_at_lap(session, defender, lap_number)
            attacker_lap = ffu.get_driver_info_at_lap(session, attacker, lap_number)

            if not ffu.is_on_track(defender_lap) or not ffu.is_on_track(attacker_lap):
                continue
            if not ffu.are_on_same_lap(session, defender, attacker, lap_number):
                continue

            # ── lap times ────────────────────────────────────────
            attacker_laptime = _timedelta_to_seconds(attacker_info.get('laptime'))
            defender_laptime = _timedelta_to_seconds(defender_info.get('laptime'))

            # §2.2: pace_delta (positive = attacker faster per lap)
            pace_delta = defender_laptime - attacker_laptime

            # ── positions ────────────────────────────────────────
            attacker_pos = ffu._safe_int(attacker_info.get('position'), 0)
            defender_pos = ffu._safe_int(defender_info.get('position'), 0)

            # ── speed traps ──────────────────────────────────────
            att_speeds = ffu.get_speed_trap_data(attacker_lap)
            def_speeds = ffu.get_speed_trap_data(defender_lap)

            # §2.3: speed deltas (attacker − defender)
            speed_i1_delta = att_speeds['speed_i1'] - def_speeds['speed_i1']
            speed_i2_delta = att_speeds['speed_i2'] - def_speeds['speed_i2']
            speed_fl_delta = att_speeds['finish_line_speed'] - def_speeds['finish_line_speed']
            speed_st_delta = att_speeds['straight_speed'] - def_speeds['straight_speed']

            # ── tyres ────────────────────────────────────────────
            attacker_tyre = (str(attacker_lap.get('Compound', 'UNKNOWN'))
                             if pd.notna(attacker_lap.get('Compound')) else 'UNKNOWN')
            defender_tyre = (str(defender_lap.get('Compound', 'UNKNOWN'))
                             if pd.notna(defender_lap.get('Compound')) else 'UNKNOWN')
            attacker_tyre_age = ffu._safe_int(attacker_lap.get('TyreLife'), 0)
            defender_tyre_age = ffu._safe_int(defender_lap.get('TyreLife'), 0)

            # §1.4: signed tyre age difference (negative = attacker on fresher tyres)
            tyre_age_diff = attacker_tyre_age - defender_tyre_age

            att_stint, att_fresh = ffu.get_stint_info(attacker_lap)
            def_stint, def_fresh = ffu.get_stint_info(defender_lap)

            # ── track / DRS ──────────────────────────────────────
            sector = ffu._safe_int(attacker_lap.get('Sector'), 1)
            is_in_drs_zone, drs_zone_length = track_info.get_drs_zone_info(track, sector)

            # ── qualification ────────────────────────────────────
            attacker_quali = ffu.get_driver_qualification_rank(session, attacker)
            defender_quali = ffu.get_driver_qualification_rank(session, defender)

            # ── overtake detection (next lap + multi-horizon, IP03 §1.2) ──
            overtake = _attacker_leads_over_defender_at_lap(
                session, attacker, defender, lap_number + 1
            )
            overtake_within_2 = _overtake_within_horizon(session, attacker, defender, lap_number, 2)
            overtake_within_3 = _overtake_within_horizon(session, attacker, defender, lap_number, 3)

            # ── sector micro-features (IP03 §3.4): defender − attacker (positive = attacker faster) ──
            att_s1, att_s2, att_s3 = ffu.get_sector_times(attacker_lap)
            def_s1, def_s2, def_s3 = ffu.get_sector_times(defender_lap)
            sector1_delta = def_s1 - att_s1
            sector2_delta = def_s2 - att_s2
            sector3_delta = def_s3 - att_s3
            strongest_sector = ffu.strongest_sector_index([sector1_delta, sector2_delta, sector3_delta])

            compound_mismatch = attacker_tyre.strip().upper() != defender_tyre.strip().upper()

            # §1.2: flag battles where either driver pits on current or next lap
            pit_stop_involved = (
                ffu.is_next_lap_pit(session, attacker, lap_number)
                or ffu.is_next_lap_pit(session, defender, lap_number)
                or ffu.is_next_lap_pit(session, attacker, lap_number + 1)
                or ffu.is_next_lap_pit(session, defender, lap_number + 1)
            )

            safety_car, yellow_flag = ffu.detect_safety_car_and_flags(session, lap_number)

            # ── race progress ────────────────────────────────────
            race_progress = round(lap_number / total_laps, 4) if total_laps > 0 else 0.0

            battles.append(BattleRecord(
                attacker=attacker,
                defender=defender,
                overtake=overtake,
                year=year,
                race_name=race_name,
                round_number=round_number,
                event_date=event_date,
                lap_number=lap_number,
                total_laps=total_laps,
                race_progress=race_progress,
                attacker_position=attacker_pos,
                defender_position=defender_pos,
                attacker_lap_time=attacker_laptime,
                defender_lap_time=defender_laptime,
                gap_ahead=actual_gap,
                pace_delta=pace_delta,
                attacker_speed_i1=att_speeds['speed_i1'],
                defender_speed_i1=def_speeds['speed_i1'],
                attacker_speed_i2=att_speeds['speed_i2'],
                defender_speed_i2=def_speeds['speed_i2'],
                attacker_finish_line_speed=att_speeds['finish_line_speed'],
                defender_finish_line_speed=def_speeds['finish_line_speed'],
                attacker_straight_speed=att_speeds['straight_speed'],
                defender_straight_speed=def_speeds['straight_speed'],
                speed_i1_delta=speed_i1_delta,
                speed_i2_delta=speed_i2_delta,
                speed_fl_delta=speed_fl_delta,
                speed_st_delta=speed_st_delta,
                safety_car=safety_car,
                yellow_flag=yellow_flag,
                attacker_tyre_compound=attacker_tyre,
                defender_tyre_compound=defender_tyre,
                attacker_tyre_age=attacker_tyre_age,
                defender_tyre_age=defender_tyre_age,
                tyre_age_difference=tyre_age_diff,
                attacker_stint=att_stint,
                defender_stint=def_stint,
                attacker_fresh_tyre=att_fresh,
                defender_fresh_tyre=def_fresh,
                pit_stop_involved=pit_stop_involved,
                track=track.upper(),
                sector=sector,
                sector_type=track_info.get_sector_type(track),
                is_in_drs_zone=is_in_drs_zone,
                drs_zone_length=drs_zone_length,
                track_type=track_info.get_track_type(track),
                attacker_qualification_rank=attacker_quali,
                defender_qualification_rank=defender_quali,
                air_temp=lap_weather['air_temp'],
                track_temp=lap_weather['track_temp'],
                humidity=lap_weather['humidity'],
                rainfall=lap_weather['rainfall'],
                wind_speed=lap_weather['wind_speed'],
                overtake_within_2=overtake_within_2,
                overtake_within_3=overtake_within_3,
                sector1_delta=sector1_delta,
                sector2_delta=sector2_delta,
                sector3_delta=sector3_delta,
                strongest_sector=strongest_sector,
                compound_mismatch=compound_mismatch,
            ))

    return battles


def detect_races_battles(year: int, gp: str, identifier: str = "R",
                         cache_path: str = None, gap_threshold: float = 1.0,
                         start_lap: int = 1) -> List[BattleRecord]:
    session = ffu.load_session(year, gp, identifier, cache_path)
    race_name = session.event['EventName']
    return detect_battles(session, year, race_name, gap_threshold, start_lap)
