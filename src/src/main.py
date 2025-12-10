import pandas as pd
from pathlib import Path
from datetime import datetime, UTC
from typing import Optional, Dict
import fast_f1_utils as ffu
import track_info

# Constants
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"


def driver_code_to_number(code: str) -> int:
    """Convert driver code to number."""
    try:
        return int(code.replace('0', ''))
    except:
        return 0


def build_battle_features(
        session,
        attacker: str,
        defender: str,
        lap: int,
        track: str
) -> Optional[Dict]:
    """Build complete feature set for a battle."""
    laps = session.laps

    # Get lap data for both drivers
    attacker_lap = laps[(laps['Driver'] == attacker) & (laps['LapNumber'] == lap)]
    defender_lap = laps[(laps['Driver'] == defender) & (laps['LapNumber'] == lap)]

    if attacker_lap.empty or defender_lap.empty:
        return None

    attacker_rec = attacker_lap.iloc[0]
    defender_rec = defender_lap.iloc[0]

    # Skip if either driver retired
    if pd.isna(attacker_rec['LapTime']) or pd.isna(defender_rec['LapTime']):
        return None

    # Get tyre info
    attacker_tyre, attacker_tyre_age = ffu.get_tyre_info(session, attacker, lap)
    defender_tyre, defender_tyre_age = ffu.get_tyre_info(session, defender, lap)

    # Calculate speeds (approximation from lap times)
    track_length = 5000
    try:
        attacker_speed = int(track_length / attacker_rec['LapTime'].total_seconds() * 3.6)
        defender_speed = int(track_length / defender_rec['LapTime'].total_seconds() * 3.6)
    except:
        attacker_speed = 0
        defender_speed = 0

    speed_difference = abs(attacker_speed - defender_speed)

    # Get sector info
    sector = int(attacker_rec.get('Sector', 1)) if pd.notna(attacker_rec.get('Sector', 1)) else 1
    is_in_drs_zone, drs_zone_length = track_info.get_drs_zone_info(track, sector)

    # Get quali ranks
    attacker_quali = ffu.get_driver_qualification_rank(session, attacker)
    defender_quali = ffu.get_driver_qualification_rank(session, defender)

    # Determine if overtake happened (position swap in next lap)
    overtake = False
    next_lap_data = laps[(laps['Driver'].isin([attacker, defender])) & (laps['LapNumber'] == lap + 1)]
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
    safety_car, yellow_flag = ffu.detect_safety_car_and_flags(session, lap)

    # Timestamp
    timestamp = datetime.now(UTC)

    return {
        'attacker': driver_code_to_number(attacker),
        'defender': driver_code_to_number(defender),
        'overtake': overtake,
        'time_stamp': timestamp,
        'attcker_speed': attacker_speed,
        'defender_speed': defender_speed,
        'speed_difference': speed_difference,
        'lap_number': lap,
        'safety_car': safety_car,
        'yellow_flag': yellow_flag,
        'attacker_tyre_compound': attacker_tyre,
        'defender_tyre_compound': defender_tyre,
        'attacker_tyre_age': attacker_tyre_age,
        'defender_tyre_age': defender_tyre_age,
        'tyre_age_difference': abs(attacker_tyre_age - defender_tyre_age),
        'track': track.upper(),
        'sector': sector,
        'sector_type': track_info.get_sector_type(track),
        'is_in_drs_zone': is_in_drs_zone,
        'drs_zone_length': drs_zone_length,
        'track_type': track_info.get_track_type(track),
        'attacker_qualification_rank': attacker_quali,
        'defender_qualification_rank': defender_quali
    }