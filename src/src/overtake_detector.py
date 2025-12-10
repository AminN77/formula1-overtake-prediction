import fast_f1_utils as ffu
import pandas as pd

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

        is_genuine, attacker_lap, defender_lap = ffu.is_genuine_overtake(
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
    session = ffu.load_session(year, gp, identifier, cache_path)
    laps = session.laps

    # Build position map
    lap_position = ffu.build_position_map(laps)

    # Detect position swaps
    position_swaps = detect_position_swaps(lap_position, start_lap)

    # Filter for genuine overtakes
    overtakes = filter_genuine_overtakes(laps, position_swaps)

    # Convert to DataFrame
    df_overtakes = pd.DataFrame(overtakes)

    return df_overtakes

