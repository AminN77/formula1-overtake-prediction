import fastf1
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List


def load_session(year: int, gp: str, identifier: str = "R",
                 cache_path: Optional[str] = None):
    if cache_path is not None:
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(cache_path)

    session = fastf1.get_session(year=year, gp=gp, identifier=identifier)
    session.load()
    return session


def get_lap_data(laps, driver: str, lap_number: int):
    lap_data = laps[(laps['Driver'] == driver) & (laps['LapNumber'] == lap_number)]
    return lap_data.iloc[0] if not lap_data.empty else None


def build_position_and_gap_map(session) -> dict:
    """Build {lap_number: {driver: {'position', 'laptime', 'lap_start_time'}}}."""
    lap_position_gap = {}
    for _, lap in session.laps.iterrows():
        if pd.isna(lap.get('LapNumber')) or pd.isna(lap.get('Position')) or pd.isna(lap.get('Driver')):
            continue
        lap_number = int(lap['LapNumber'])
        driver = lap['Driver']
        lap_position_gap.setdefault(lap_number, {})[driver] = {
            'position': lap['Position'],
            'laptime': lap['LapTime'],
            'lap_start_time': lap.get('LapStartTime'),
        }
    return lap_position_gap


def build_position_map(laps) -> dict:
    """Build {lap_number: {driver: position}}."""
    lap_position = {}
    for _, lap in laps.iterrows():
        if pd.isna(lap.get('LapNumber')) or pd.isna(lap.get('Position')) or pd.isna(lap.get('Driver')):
            continue
        lap_number = int(lap['LapNumber'])
        lap_position.setdefault(lap_number, {})[lap['Driver']] = lap['Position']
    return lap_position


def is_pitstop_lap(lap_data) -> bool:
    if lap_data is None:
        return False
    return pd.notna(lap_data['PitInTime']) or pd.notna(lap_data['PitOutTime'])


def is_valid_lap(lap_data) -> bool:
    if lap_data is None:
        return False
    return pd.notna(lap_data['LapTime'])


def get_driver_info_at_lap(session, driver: str, lap_number: int):
    lap_data = session.laps[
        (session.laps['Driver'] == driver) &
        (session.laps['LapNumber'] == lap_number)
    ]
    return lap_data.iloc[0] if not lap_data.empty else None


def is_on_track(driver_lap) -> bool:
    if driver_lap is None:
        return False
    pit_in = pd.notna(driver_lap['PitInTime'])
    pit_out = pd.notna(driver_lap['PitOutTime'])
    return pd.notna(driver_lap['LapTime']) and not (pit_in or pit_out)


def calculate_gap_between_drivers(driver_a_info: dict, driver_b_info: dict) -> Optional[float]:
    """Lap-time-based gap (fallback when LapStartTime is unavailable)."""
    laptime_a = driver_a_info.get('laptime')
    laptime_b = driver_b_info.get('laptime')

    if laptime_a is None or laptime_b is None:
        return None
    if pd.isna(laptime_a) or pd.isna(laptime_b):
        return None

    try:
        secs_a = laptime_a.total_seconds() if hasattr(laptime_a, 'total_seconds') else float(laptime_a)
        secs_b = laptime_b.total_seconds() if hasattr(laptime_b, 'total_seconds') else float(laptime_b)
        return abs(secs_a - secs_b)
    except (TypeError, ValueError):
        return None


def calculate_actual_gap(defender_info: dict, attacker_info: dict) -> Optional[float]:
    """Actual inter-car gap from LapStartTime (seconds).

    The attacker (behind) crosses the start line later than the defender (ahead),
    so the gap = attacker_start - defender_start (positive = attacker behind).
    Falls back to lap-time difference when LapStartTime is unavailable.
    """
    start_def = defender_info.get('lap_start_time')
    start_att = attacker_info.get('lap_start_time')

    if (start_def is not None and start_att is not None
            and not pd.isna(start_def) and not pd.isna(start_att)):
        try:
            secs_def = start_def.total_seconds() if hasattr(start_def, 'total_seconds') else float(start_def)
            secs_att = start_att.total_seconds() if hasattr(start_att, 'total_seconds') else float(start_att)
            gap = abs(secs_att - secs_def)
            if gap < 120:  # sanity: gaps > 2 min are likely errors
                return gap
        except (TypeError, ValueError):
            pass

    return calculate_gap_between_drivers(defender_info, attacker_info)


def is_next_lap_pit(session, driver: str, lap_number: int) -> bool:
    """True if the driver pits on the specified lap."""
    lap_data = get_driver_info_at_lap(session, driver, lap_number)
    if lap_data is None:
        return False
    return pd.notna(lap_data.get('PitInTime')) or pd.notna(lap_data.get('PitOutTime'))


def are_on_same_lap(session, driver_a: str, driver_b: str, lap_number: int) -> bool:
    lap_a = get_driver_info_at_lap(session, driver_a, lap_number)
    lap_b = get_driver_info_at_lap(session, driver_b, lap_number)
    return lap_a is not None and lap_b is not None


def get_driver_qualification_rank(session, driver: str) -> int:
    try:
        return _safe_int(session.get_driver(driver).get("GridPosition"), default=0)
    except Exception:
        return 0


def get_driver_team(session, driver: str) -> str:
    """Team name for a driver in this session (IP04 §1.1)."""
    try:
        return str(session.get_driver(driver)["TeamName"])
    except Exception:
        return "UNKNOWN"


def count_cars_within_drs(
    lap_data: dict, target_position: int, threshold: float = 1.0
) -> int:
    """Count how many cars are within *threshold* seconds of each other in a
    chain that includes *target_position* (IP04 §1.3 DRS-train proxy).

    *lap_data* is the dict {driver: {'position': …, 'lap_start_time': …}} for
    one lap from `build_position_and_gap_map`.
    """
    entries = sorted(lap_data.values(), key=lambda x: x["position"])
    starts: list[float | None] = []
    for e in entries:
        lst = e.get("lap_start_time")
        if lst is not None and not pd.isna(lst):
            try:
                starts.append(lst.total_seconds() if hasattr(lst, "total_seconds") else float(lst))
            except (TypeError, ValueError):
                starts.append(None)
        else:
            starts.append(None)

    target_idx = None
    for j, e in enumerate(entries):
        if int(e["position"]) == target_position:
            target_idx = j
            break
    if target_idx is None:
        return 1

    lo = target_idx
    while lo > 0:
        if starts[lo] is not None and starts[lo - 1] is not None:
            if abs(starts[lo] - starts[lo - 1]) <= threshold:
                lo -= 1
                continue
        break

    hi = target_idx
    while hi < len(starts) - 1:
        if starts[hi] is not None and starts[hi + 1] is not None:
            if abs(starts[hi + 1] - starts[hi]) <= threshold:
                hi += 1
                continue
        break

    return hi - lo + 1


def gap_to_position(
    lap_data: dict, reference_pos: int, target_pos: int
) -> float:
    """Gap (seconds) between two positions on the same lap.

    Positive = target is behind reference.  Returns 99.0 if unavailable.
    """
    ref_start = tgt_start = None
    for info in lap_data.values():
        p = int(info["position"])
        lst = info.get("lap_start_time")
        if p == reference_pos and lst is not None and not pd.isna(lst):
            ref_start = lst.total_seconds() if hasattr(lst, "total_seconds") else float(lst)
        if p == target_pos and lst is not None and not pd.isna(lst):
            tgt_start = lst.total_seconds() if hasattr(lst, "total_seconds") else float(lst)
    if ref_start is not None and tgt_start is not None:
        return abs(tgt_start - ref_start)
    return 99.0


def detect_safety_car_and_flags(session, lap: int) -> Tuple[bool, bool]:
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
    except Exception:
        pass
    return safety_car, yellow_flag


# ── helpers ──────────────────────────────────────────────────────


def _safe_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return bool(value)


def _sector_time_to_seconds(val) -> float:
    if val is None:
        return 0.0
    try:
        if pd.isna(val):
            return 0.0
    except (TypeError, ValueError):
        pass
    try:
        if hasattr(val, "total_seconds"):
            return float(val.total_seconds())
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def get_sector_times(lap_data) -> Tuple[float, float, float]:
    """Sector1/2/3 times in seconds (0 if missing). IP03 §3.4."""
    if lap_data is None:
        return 0.0, 0.0, 0.0
    return (
        _sector_time_to_seconds(lap_data.get("Sector1Time")),
        _sector_time_to_seconds(lap_data.get("Sector2Time")),
        _sector_time_to_seconds(lap_data.get("Sector3Time")),
    )


def strongest_sector_index(deltas: List[float]) -> int:
    """Index 0–2 of max delta (attacker advantage); -1 if all non-positive or invalid."""
    if not deltas or all(d <= 0 for d in deltas):
        return -1
    return int(np.argmax(deltas))


def get_speed_trap_data(lap_data) -> Dict[str, float]:
    """Extract all speed-trap readings from a single lap row."""
    if lap_data is None:
        return {
            'speed_i1': 0.0, 'speed_i2': 0.0,
            'finish_line_speed': 0.0, 'straight_speed': 0.0,
        }
    return {
        'speed_i1': _safe_float(lap_data.get('SpeedI1')),
        'speed_i2': _safe_float(lap_data.get('SpeedI2')),
        'finish_line_speed': _safe_float(lap_data.get('SpeedFL')),
        'straight_speed': _safe_float(lap_data.get('SpeedST')),
    }


def get_stint_info(lap_data) -> Tuple[int, bool]:
    """Return (stint_number, fresh_tyre) from a single lap row."""
    if lap_data is None:
        return 1, False
    stint = _safe_int(lap_data.get('Stint'), default=1)
    fresh = _safe_bool(lap_data.get('FreshTyre'), default=False)
    return stint, fresh


def build_weather_lookup(session) -> Dict[int, Dict[str, float]]:
    """Build a {lap_number: weather_dict} lookup from session weather data."""
    lookup: Dict[int, Dict[str, float]] = {}
    try:
        weather = session.weather_data
        if weather is None or weather.empty:
            return lookup

        laps = session.laps
        if laps.empty:
            return lookup

        weather = weather.copy()
        if 'Time' not in weather.columns:
            return lookup

        weather_times = weather['Time'].values

        for _, lap in laps.iterrows():
            lap_num = int(lap['LapNumber'])
            if lap_num in lookup:
                continue

            lap_start = lap.get('LapStartTime')
            if lap_start is None or pd.isna(lap_start):
                continue

            idx = np.argmin(np.abs(weather_times - lap_start))
            row = weather.iloc[idx]

            lookup[lap_num] = {
                'air_temp': _safe_float(row.get('AirTemp')),
                'track_temp': _safe_float(row.get('TrackTemp')),
                'humidity': _safe_float(row.get('Humidity')),
                'rainfall': _safe_bool(row.get('Rainfall')),
                'wind_speed': _safe_float(row.get('WindSpeed')),
            }
    except Exception:
        pass

    return lookup


DEFAULT_WEATHER: Dict[str, float] = {
    'air_temp': 0.0,
    'track_temp': 0.0,
    'humidity': 0.0,
    'rainfall': False,
    'wind_speed': 0.0,
}


def get_total_laps(session) -> int:
    """Return the scheduled number of race laps."""
    try:
        total = session.total_laps
        if total and int(total) > 0:
            return int(total)
    except Exception:
        pass
    try:
        return int(session.laps['LapNumber'].max())
    except Exception:
        return 0
