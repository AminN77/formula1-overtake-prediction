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


def detect_position_swaps(lap_position, start_lap=2):
    """
    Detect all position swaps between consecutive laps.

    Args:
        lap_position: Position map from build_position_map()
        start_lap: Lap to start detection (default 2 to skip race start)

    Returns:
        list: List of position swap dictionaries
    """
    max_lap = max(lap_position.keys())
    position_swaps = []

    for lap in range(start_lap, max_lap):
        current = lap_position.get(lap, {})
        nxt = lap_position.get(lap + 1, {})

        # For every driver pair, check if order swapped
        for drv_a, pos_a in current.items():
            # Find which driver was directly ahead of drv_a on lap L
            for drv_b, pos_b in current.items():
                if pos_b == pos_a - 1:  # drv_b was ahead of drv_a
                    # Now compare positions on next lap
                    next_pos_a = nxt.get(drv_a)
                    next_pos_b = nxt.get(drv_b)

                    if next_pos_a is None or next_pos_b is None:
                        continue

                    # Position swap happened: A is ahead of B
                    if next_pos_a < next_pos_b:
                        position_swaps.append({
                            'lap': lap + 1,
                            'attacker': drv_a,
                            'defender': drv_b
                        })

    return position_swaps


def filter_genuine_overtakes(laps, position_swaps):
    """
    Filter position swaps to only include genuine overtakes.

    Args:
        laps: FastF1 laps dataframe
        position_swaps: List of position swap dictionaries

    Returns:
        list: List of genuine overtake dictionaries
    """
    overtakes = []

    for swap in position_swaps:
        lap = swap['lap']
        attacker = swap['attacker']
        defender = swap['defender']

        is_genuine, attacker_lap, defender_lap = is_genuine_overtake(
            laps, attacker, defender, lap
        )

        if is_genuine:
            overtakes.append({
                'lap': lap,
                'attacker': attacker,
                'defender': defender,
                'attacker_laptime': attacker_lap['LapTime'].total_seconds() if pd.notna(
                    attacker_lap['LapTime']) else None,
                'defender_laptime': defender_lap['LapTime'].total_seconds() if pd.notna(
                    defender_lap['LapTime']) else None
            })

    return overtakes


def detect_overtakes(year, gp, identifier="R", cache_path=None, start_lap=2):
    """
    Main function to detect all genuine overtakes in a race session.

    Args:
        year: Season year
        gp: Grand Prix name or round number
        identifier: Session identifier (default "R" for race)
        cache_path: Path to cache directory
        start_lap: Lap to start detection (default 2 to skip race start)

    Returns:
        pd.DataFrame: DataFrame of genuine overtakes
    """
    # Load session
    session = load_session(year, gp, identifier, cache_path)
    laps = session.laps

    # Build position map
    lap_position = build_position_map(laps)

    # Detect position swaps
    position_swaps = detect_position_swaps(lap_position, start_lap)

    # Filter for genuine overtakes
    overtakes = filter_genuine_overtakes(laps, position_swaps)

    # Convert to DataFrame
    df_overtakes = pd.DataFrame(overtakes)

    return df_overtakes


# Example usage
if __name__ == "__main__":
    df_overtakes = detect_overtakes(year=2025, gp="monza", identifier="R")
    print(f"\nTotal overtakes detected: {len(df_overtakes)}")
    print(df_overtakes)