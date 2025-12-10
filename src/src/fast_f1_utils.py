import fastf1
from pathlib import Path
import pandas as pd
from typing import Tuple

def load_session(year, gp, identifier="R", cache_path=None):
    """
    Load F1 session data with caching enabled.

    Args:
        year: Season year
        gp: Grand Prix name or round number
        identifier: Session identifier (R=Race, Q=Qualifying, etc.)
        cache_path: Path to cache directory

    Returns:
        Loaded FastF1 session object
    """
    if cache_path is not None:
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(cache_path)

    session = fastf1.get_session(year=year, gp=gp, identifier=identifier)
    session.load()
    return session


def get_lap_data(laps, driver, lap_number):
    """
    Get lap data for a specific driver and lap number.

    Args:
        laps: FastF1 laps dataframe
        driver: Driver code
        lap_number: Lap number

    Returns:
        Lap data row or None if not found
    """
    lap_data = laps[(laps['Driver'] == driver) & (laps['LapNumber'] == lap_number)]
    return lap_data.iloc[0] if not lap_data.empty else None


def build_position_and_gap_map(session):
    """
    Build a nested dictionary mapping lap number to driver positions and lap times.

    Args:
        session: FastF1 session object

    Returns:
        dict: {lap_number: {driver: {'position': int, 'laptime': float}}}
    """
    lap_position_gap = {}
    laps = session.laps

    for _, lap in laps.iterrows():
        lap_number = int(lap['LapNumber'])
        driver = lap['Driver']
        position = lap['Position']
        laptime = lap['LapTime']

        lap_position_gap.setdefault(lap_number, {})
        lap_position_gap[lap_number][driver] = {
            'position': position,
            'laptime': laptime
        }

    return lap_position_gap



def build_position_map(laps):
    """
    Build a nested dictionary mapping lap number to driver positions.

    Args:
        laps: FastF1 laps dataframe

    Returns:
        dict: {lap_number: {driver: position}}
    """
    lap_position = {}

    for _, lap in laps.iterrows():
        lap_number = int(lap['LapNumber'])
        driver = lap['Driver']
        position = lap['Position']

        lap_position.setdefault(lap_number, {})
        lap_position[lap_number][driver] = position

    return lap_position


def is_pitstop_lap(lap_data):
    """
    Check if a lap involves a pitstop.

    Args:
        lap_data: Single lap data row

    Returns:
        bool: True if pitstop occurred
    """
    if lap_data is None:
        return False
    return pd.notna(lap_data['PitInTime']) or pd.notna(lap_data['PitOutTime'])


def is_valid_lap(lap_data):
    """
    Check if a lap is valid (completed without issues).

    Args:
        lap_data: Single lap data row

    Returns:
        bool: True if lap is valid
    """
    if lap_data is None:
        return False
    return pd.notna(lap_data['LapTime'])


def get_driver_info_at_lap(session, driver, lap_number):
    """
    Get driver information at a specific lap.

    Args:
        session: FastF1 session object
        driver: Driver code
        lap_number: Lap number

    Returns:
        Driver lap data row or None if not found
    """
    lap_data = session.laps[(session.laps['Driver'] == driver) &
                            (session.laps['LapNumber'] == lap_number)]
    return lap_data.iloc[0] if not lap_data.empty else None


def is_on_track(driver_lap):
    """
    Check if a driver is on track (not in pit lane).

    Args:
        driver_lap: Single lap data row

    Returns:
        bool: True if driver is on track
    """
    if driver_lap is None:
        return False
    # Driver is on track if they have valid lap time and are not pitting
    pit_in = pd.notna(driver_lap['PitInTime'])
    pit_out = pd.notna(driver_lap['PitOutTime'])
    return pd.notna(driver_lap['LapTime']) and not (pit_in or pit_out)


def calculate_gap_between_drivers(driver_a_info, driver_b_info):
    """
    Calculate time gap between two drivers based on lap times.

    Args:
        driver_a_info: Dictionary with 'laptime' for first driver
        driver_b_info: Dictionary with 'laptime' for second driver

    Returns:
        float: Time gap in seconds, or None if unavailable
    """
    laptime_a = driver_a_info.get('laptime')
    laptime_b = driver_b_info.get('laptime')

    # Handle None and NaT values
    if laptime_a is None or laptime_b is None:
        return None
    if pd.isna(laptime_a) or pd.isna(laptime_b):
        return None

    # Convert to seconds
    try:
        if hasattr(laptime_a, 'total_seconds'):
            laptime_a_seconds = laptime_a.total_seconds()
        else:
            laptime_a_seconds = float(laptime_a)

        if hasattr(laptime_b, 'total_seconds'):
            laptime_b_seconds = laptime_b.total_seconds()
        else:
            laptime_b_seconds = float(laptime_b)

        return abs(laptime_a_seconds - laptime_b_seconds)
    except (TypeError, ValueError):
        return None


def are_on_same_lap(session, driver_a, driver_b, lap_number):
    """
    Check if both drivers are on the same lap.

    Args:
        session: FastF1 session object
        driver_a: First driver code
        driver_b: Second driver code
        lap_number: Lap number

    Returns:
        bool: True if both drivers completed this lap
    """
    lap_a = get_driver_info_at_lap(session, driver_a, lap_number)
    lap_b = get_driver_info_at_lap(session, driver_b, lap_number)

    return lap_a is not None and lap_b is not None

def driver_continues_racing(laps, driver, lap_number):
    """
    Check if a driver continues racing after a given lap.

    Args:
        laps: FastF1 laps dataframe
        driver: Driver code
        lap_number: Current lap number

    Returns:
        bool: True if driver appears in next lap
    """
    next_lap = laps[(laps['Driver'] == driver) & (laps['LapNumber'] == lap_number + 1)]
    return not next_lap.empty


def is_genuine_overtake(laps, attacker, defender, lap_number):
    """
    Apply filters to determine if a position swap is a genuine overtake.

    Args:
        laps: FastF1 laps dataframe
        attacker: Driver who gained position
        defender: Driver who lost position
        lap_number: Lap where position swap occurred

    Returns:
        tuple: (is_genuine, attacker_lap_data, defender_lap_data)
    """
    # Get lap data for both drivers
    defender_lap = get_lap_data(laps, defender, lap_number)
    attacker_lap = get_lap_data(laps, attacker, lap_number)

    # Skip if we can't find the lap data
    if defender_lap is None or attacker_lap is None:
        return False, None, None

    # Filter 1: Defender pitted on this lap
    if is_pitstop_lap(defender_lap):
        return False, attacker_lap, defender_lap

    # Filter 2: Attacker pitted (gained position via pitstop strategy)
    if is_pitstop_lap(attacker_lap):
        return False, attacker_lap, defender_lap

    # Filter 3: Defender DNF/retired (check if they completed the lap)
    if not is_valid_lap(defender_lap):
        return False, attacker_lap, defender_lap

    # Filter 4: Check if defender appears in next lap (still racing)
    if not driver_continues_racing(laps, defender, lap_number):
        return False, attacker_lap, defender_lap

    return True, attacker_lap, defender_lap


def get_driver_qualification_rank(session, driver: str) -> int:
    """Get driver's qualifying position."""
    try:
        quali = session.qualifying
        driver_quali = quali[quali['Driver'] == driver]
        if not driver_quali.empty:
            return int(driver_quali.iloc[0]['Position'])
    except:
        pass
    return 0


def get_tyre_info(session, driver: str, lap: int) -> Tuple[str, int]:
    """Get tyre compound and age for a driver at a given lap."""
    laps = session.laps
    lap_data = laps[(laps['Driver'] == driver) & (laps['LapNumber'] <= lap)]

    if lap_data.empty:
        return 'UNKNOWN', 0

    latest = lap_data.iloc[-1]
    compound = str(latest.get('Compound', 'UNKNOWN'))
    tyre_life = int(latest.get('TyreLife', 0)) if pd.notna(latest.get('TyreLife', 0)) else 0

    return compound, tyre_life

def detect_safety_car_and_flags(session, lap: int) -> Tuple[bool, bool]:
    """Detect safety car and yellow flags at a specific lap."""
    safety_car = False
    yellow_flag = False

    try:
        events = session.race_control_data
        if events is not None and not events.empty:
            lap_events = events[events['LapNumber'] == lap]
            if not lap_events.empty:
                messages = ' '.join(lap_events['Message'].astype(str))
                safety_car = 'Safety Car' in messages
                yellow_flag = 'Yellow' in messages
    except:
        pass

    return safety_car, yellow_flag