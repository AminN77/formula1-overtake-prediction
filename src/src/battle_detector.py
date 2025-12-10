import fastf1
import pandas as pd
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent if '__file__' in globals() else Path.cwd()
DEFAULT_CACHE_DIR = PROJECT_ROOT / "cache"


def load_session(year, gp, identifier="R", cache_path=None):
    """
    Load F1 session data with caching enabled.

    Args:
        year: Season year
        gp: Grand Prix name or round number
        identifier: Session identifier (R=Race, Q=Qualifying, etc.)
        cache_path: Path to cache directory (defaults to project/cache)

    Returns:
        Loaded FastF1 session object
    """
    if cache_path is None:
        cache_path = str(DEFAULT_CACHE_DIR)

    # Create cache directory if it doesn't exist
    Path(cache_path).mkdir(parents=True, exist_ok=True)

    fastf1.Cache.enable_cache(cache_path)
    session = fastf1.get_session(year=year, gp=gp, identifier=identifier)
    session.load()
    return session


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


def detect_battles(session, gap_threshold=1.0, start_lap=2):
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
        list: List of battle dictionaries
    """
    lap_position_gap = build_position_and_gap_map(session)
    max_lap = max(lap_position_gap.keys())
    battles = []

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
            gap = calculate_gap_between_drivers(defender_info, attacker_info)
            if gap is None or gap >= gap_threshold:
                continue

            # Condition 3 & 4: Both on track and same lap
            defender_lap = get_driver_info_at_lap(session, defender, lap_number)
            attacker_lap = get_driver_info_at_lap(session, attacker, lap_number)

            if not is_on_track(defender_lap) or not is_on_track(attacker_lap):
                continue

            if not are_on_same_lap(session, defender, attacker, lap_number):
                continue

            # Valid battle found
            battles.append({
                'lap': lap_number,
                'defender': defender,
                'attacker': attacker,
                'defender_position': int(defender_pos),
                'attacker_position': int(attacker_pos),
                'gap': gap,
                'defender_laptime': defender_lap['LapTime'].total_seconds() if pd.notna(
                    defender_lap['LapTime']) else None,
                'attacker_laptime': attacker_lap['LapTime'].total_seconds() if pd.notna(
                    attacker_lap['LapTime']) else None
            })

    return battles


def detect_races_battles(year, gp, identifier="R", cache_path=None, gap_threshold=1.0, start_lap=2):
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
        pd.DataFrame: DataFrame of battles
    """
    # Load session
    session = load_session(year, gp, identifier, cache_path)

    # Detect battles
    battles = detect_battles(session, gap_threshold, start_lap)

    # Convert to DataFrame
    df_battles = pd.DataFrame(battles)

    return df_battles


# Example usage
if __name__ == "__main__":
    df_battles = detect_races_battles(year=2024, gp="monza", identifier="R", gap_threshold=1.0)
    print(f"\nTotal battles detected: {len(df_battles)}")
    print(df_battles)

    # Optional: Show summary statistics
    if not df_battles.empty:
        print(f"\nBattles by defender:")
        print(df_battles['defender'].value_counts())
        print(f"\nAverage gap in battles: {df_battles['gap'].mean():.3f}s")