import fastf1
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict


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
    """Build {lap_number: {driver: {'position': int, 'laptime': Timedelta}}}."""
    lap_position_gap = {}
    for _, lap in session.laps.iterrows():
        lap_number = int(lap['LapNumber'])
        driver = lap['Driver']
        lap_position_gap.setdefault(lap_number, {})[driver] = {
            'position': lap['Position'],
            'laptime': lap['LapTime'],
        }
    return lap_position_gap


def build_position_map(laps) -> dict:
    """Build {lap_number: {driver: position}}."""
    lap_position = {}
    for _, lap in laps.iterrows():
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


def are_on_same_lap(session, driver_a: str, driver_b: str, lap_number: int) -> bool:
    lap_a = get_driver_info_at_lap(session, driver_a, lap_number)
    lap_b = get_driver_info_at_lap(session, driver_b, lap_number)
    return lap_a is not None and lap_b is not None


def get_driver_qualification_rank(session, driver: str) -> int:
    return int(session.get_driver(driver)["GridPosition"])


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


# ── new helpers ──────────────────────────────────────────────────


def _safe_float(value, default: float = 0.0) -> float:
    """Convert a value to float, returning *default* for NaN / None / errors."""
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


def get_speed_trap_data(lap_data) -> Dict[str, float]:
    """Extract all speed-trap readings from a single lap row.

    Returns dict with keys: speed_i1, speed_i2, finish_line_speed, straight_speed.
    Missing values are returned as 0.0.
    """
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
    """Build a {lap_number: weather_dict} lookup from session weather data.

    Weather samples are matched to each lap by finding the closest weather
    timestamp to each lap's start time.  If weather data is unavailable the
    lookup will be empty.
    """
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
