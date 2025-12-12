import pandas as pd
from datetime import datetime, UTC
from typing import List
import fast_f1_utils as ffu
from models import BattleRecord
import track_info

def detect_battles(session, gap_threshold=1.0, start_lap=2) -> List[BattleRecord]:
    """
    Detect all battles between pairs of drivers throughout a race.

    A battle is defined as:
    1. Consecutive positions (defender at P, attacker at P+1)
    2. Time gap < gap_threshold (default 1.0 second)
    3. Both cars on track (not in pit lane)
    4. Both cars on the same lap

    Args:
        session: FastF1 session object
        gap_threshold: Maximum time gap in seconds (default 1.0)
        start_lap: Lap to start detection (default 2 to skip race start)

    Returns:
        list: List of BattleRecord objects
    """
    lap_position_gap = ffu.build_position_and_gap_map(session)
    max_lap = max(lap_position_gap.keys())
    battles = []

    track = session.event['Location']
    laps = session.laps

    for lap_number in range(start_lap, max_lap + 1):
        lap_data = lap_position_gap.get(lap_number, {})

        # Sort drivers by position
        drivers_by_position = sorted(
            lap_data.items(),
            key=lambda x: x[1]['position']
        )

        # Check consecutive position pairs
        for i in range(len(drivers_by_position) - 1):
            defender, defender_info = drivers_by_position[i]
            attacker, attacker_info = drivers_by_position[i + 1]

            defender_pos = defender_info['position']
            attacker_pos = attacker_info['position']

            # Condition 1: Consecutive positions
            if attacker_pos != defender_pos + 1:
                continue

            # Condition 2: Gap threshold
            gap = ffu.calculate_gap_between_drivers(defender_info, attacker_info)
            if gap is None or gap >= gap_threshold:
                continue

            # Condition 3 & 4: Both on track and same lap
            defender_lap = ffu.get_driver_info_at_lap(session, defender, lap_number)
            attacker_lap = ffu.get_driver_info_at_lap(session, attacker, lap_number)

            if not ffu.is_on_track(defender_lap) or not ffu.is_on_track(attacker_lap):
                continue

            if not ffu.are_on_same_lap(session, defender, attacker, lap_number):
                continue

            # Get additional features
            def _timedelta_to_seconds(td):
                if pd.isna(td):
                    return 0.0
                if isinstance(td, pd.Timedelta):
                    return td.total_seconds()
                return float(td)

            attacker_laptime = _timedelta_to_seconds(attacker_info.get('laptime'))
            defender_laptime = _timedelta_to_seconds(defender_info.get('laptime'))

            attacker_speed = attacker_laptime
            defender_speed = defender_laptime
            speed_difference = attacker_laptime - defender_laptime

            # Get tyre info
            attacker_tyre = str(attacker_lap.get('Compound', 'UNKNOWN')) if pd.notna(
                attacker_lap.get('Compound')) else 'UNKNOWN'
            defender_tyre = str(defender_lap.get('Compound', 'UNKNOWN')) if pd.notna(
                defender_lap.get('Compound')) else 'UNKNOWN'
            attacker_tyre_age = int(attacker_lap.get('TyreLife', 0)) if pd.notna(attacker_lap.get('TyreLife', 0)) else 0
            defender_tyre_age = int(defender_lap.get('TyreLife', 0)) if pd.notna(defender_lap.get('TyreLife', 0)) else 0
            tyre_age_difference = abs(attacker_tyre_age - defender_tyre_age)

            # Get sector and track info
            sector = int(attacker_lap.get('Sector', 1)) if pd.notna(attacker_lap.get('Sector', 1)) else 1
            sector_type = track_info.get_sector_type(track)
            track_type = track_info.get_track_type(track)

            # DRS zone info
            is_in_drs_zone, drs_zone_length = track_info.get_drs_zone_info(track, sector)

            # Get quali ranks
            attacker_quali = ffu.get_driver_qualification_rank(session, attacker)
            defender_quali = ffu.get_driver_qualification_rank(session, defender)

            # Determine if overtake happened
            overtake = False
            next_lap_data = laps[(laps['Driver'].isin([attacker, defender])) & (laps['LapNumber'] == lap_number + 1)]
            if len(next_lap_data) == 2:
                attacker_next = next_lap_data[next_lap_data['Driver'] == attacker]
                defender_next = next_lap_data[next_lap_data['Driver'] == defender]
                if not attacker_next.empty and not defender_next.empty:
                    try:
                        attacker_next_pos = int(attacker_next.iloc[0]['Position'])
                        defender_next_pos = int(defender_next.iloc[0]['Position'])
                        overtake = attacker_next_pos < defender_next_pos
                    except:
                        pass

            # Get safety car and flags
            safety_car, yellow_flag = ffu.detect_safety_car_and_flags(session, lap_number)


            # Create BattleRecord
            battle = BattleRecord(
                attacker=attacker,
                defender=defender,
                overtake=overtake,
                attacker_speed=attacker_speed,
                defender_speed=defender_speed,
                speed_difference=speed_difference,
                lap_number=lap_number,
                safety_car=safety_car,
                yellow_flag=yellow_flag,
                attacker_tyre_compound=attacker_tyre,
                defender_tyre_compound=defender_tyre,
                attacker_tyre_age=attacker_tyre_age,
                defender_tyre_age=defender_tyre_age,
                tyre_age_difference=tyre_age_difference,
                track=track.upper(),
                sector=sector,
                sector_type=sector_type,
                is_in_drs_zone=is_in_drs_zone,
                drs_zone_length=drs_zone_length,
                track_type=track_type,
                attacker_qualification_rank=attacker_quali,
                defender_qualification_rank=defender_quali
            )

            battles.append(battle)

    return battles


def detect_races_battles(year: int, gp: str, identifier: str = "R",
                         cache_path=None, gap_threshold: float = 1.0,
                         start_lap: int = 1) -> List[BattleRecord]:
    """
    Main function to detect all battles in a race session.

    Args:
        year: Season year
        gp: Grand Prix name or round number
        identifier: Session identifier (default "R" for race)
        cache_path: Path to cache directory
        gap_threshold: Maximum time gap in seconds (default 1.0)
        start_lap: Lap to start detection (default 2 to skip race start)

    Returns:
        list: List of BattleRecord objects
    """
    # Load session
    session = ffu.load_session(year, gp, identifier, cache_path)

    # Detect battles
    battles = detect_battles(session, gap_threshold, start_lap)

    return battles