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


def detect_battles(session, year: int, race_name: str,
                   gap_threshold: float = 1.0,
                   start_lap: int = 2) -> List[BattleRecord]:
    """
    Detect battles between consecutive-position drivers within *gap_threshold*.

    A battle requires:
      1. Consecutive positions (defender P, attacker P+1)
      2. Lap-time gap < gap_threshold
      3. Both cars on track (no pit in/out)
      4. Both cars on the same lap
    """
    lap_position_gap = ffu.build_position_and_gap_map(session)
    max_lap = max(lap_position_gap.keys())
    battles: List[BattleRecord] = []

    track = session.event['Location']
    laps = session.laps
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

            gap = ffu.calculate_gap_between_drivers(defender_info, attacker_info)
            if gap is None or gap >= gap_threshold:
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

            # ── positions ────────────────────────────────────────
            attacker_pos = ffu._safe_int(attacker_info.get('position'), 0)
            defender_pos = ffu._safe_int(defender_info.get('position'), 0)

            # ── speed traps ──────────────────────────────────────
            att_speeds = ffu.get_speed_trap_data(attacker_lap)
            def_speeds = ffu.get_speed_trap_data(defender_lap)

            # ── tyres ────────────────────────────────────────────
            attacker_tyre = (str(attacker_lap.get('Compound', 'UNKNOWN'))
                             if pd.notna(attacker_lap.get('Compound')) else 'UNKNOWN')
            defender_tyre = (str(defender_lap.get('Compound', 'UNKNOWN'))
                             if pd.notna(defender_lap.get('Compound')) else 'UNKNOWN')
            attacker_tyre_age = ffu._safe_int(attacker_lap.get('TyreLife'), 0)
            defender_tyre_age = ffu._safe_int(defender_lap.get('TyreLife'), 0)

            att_stint, att_fresh = ffu.get_stint_info(attacker_lap)
            def_stint, def_fresh = ffu.get_stint_info(defender_lap)

            # ── track / DRS ──────────────────────────────────────
            sector = ffu._safe_int(attacker_lap.get('Sector'), 1)
            is_in_drs_zone, drs_zone_length = track_info.get_drs_zone_info(track, sector)

            # ── qualification ────────────────────────────────────
            attacker_quali = ffu.get_driver_qualification_rank(session, attacker)
            defender_quali = ffu.get_driver_qualification_rank(session, defender)

            # ── overtake detection ───────────────────────────────
            overtake = False
            next_lap_data = laps[
                (laps['Driver'].isin([attacker, defender])) &
                (laps['LapNumber'] == lap_number + 1)
            ]
            if len(next_lap_data) == 2:
                att_next = next_lap_data[next_lap_data['Driver'] == attacker]
                def_next = next_lap_data[next_lap_data['Driver'] == defender]
                if not att_next.empty and not def_next.empty:
                    try:
                        overtake = int(att_next.iloc[0]['Position']) < int(def_next.iloc[0]['Position'])
                    except (ValueError, TypeError):
                        pass

            safety_car, yellow_flag = ffu.detect_safety_car_and_flags(session, lap_number)

            # ── race progress ────────────────────────────────────
            race_progress = round(lap_number / total_laps, 4) if total_laps > 0 else 0.0

            battles.append(BattleRecord(
                attacker=attacker,
                defender=defender,
                overtake=overtake,
                year=year,
                race_name=race_name,
                lap_number=lap_number,
                total_laps=total_laps,
                race_progress=race_progress,
                attacker_position=attacker_pos,
                defender_position=defender_pos,
                attacker_lap_time=attacker_laptime,
                defender_lap_time=defender_laptime,
                gap_ahead=attacker_laptime - defender_laptime,
                attacker_speed_i1=att_speeds['speed_i1'],
                defender_speed_i1=def_speeds['speed_i1'],
                attacker_speed_i2=att_speeds['speed_i2'],
                defender_speed_i2=def_speeds['speed_i2'],
                attacker_finish_line_speed=att_speeds['finish_line_speed'],
                defender_finish_line_speed=def_speeds['finish_line_speed'],
                attacker_straight_speed=att_speeds['straight_speed'],
                defender_straight_speed=def_speeds['straight_speed'],
                safety_car=safety_car,
                yellow_flag=yellow_flag,
                attacker_tyre_compound=attacker_tyre,
                defender_tyre_compound=defender_tyre,
                attacker_tyre_age=attacker_tyre_age,
                defender_tyre_age=defender_tyre_age,
                tyre_age_difference=abs(attacker_tyre_age - defender_tyre_age),
                attacker_stint=att_stint,
                defender_stint=def_stint,
                attacker_fresh_tyre=att_fresh,
                defender_fresh_tyre=def_fresh,
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
            ))

    return battles


def detect_races_battles(year: int, gp: str, identifier: str = "R",
                         cache_path: str = None, gap_threshold: float = 1.0,
                         start_lap: int = 1) -> List[BattleRecord]:
    session = ffu.load_session(year, gp, identifier, cache_path)
    race_name = session.event['EventName']
    return detect_battles(session, year, race_name, gap_threshold, start_lap)
