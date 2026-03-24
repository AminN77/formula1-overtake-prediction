"""Build UI schema from model meta + default row."""

from __future__ import annotations

from typing import Any

from ..schemas.model_info import FeatureSchemaItem
from .feature_builder import build_single_row

RACES = [
    "Abu Dhabi Grand Prix",
    "Australian Grand Prix",
    "Austrian Grand Prix",
    "Azerbaijan Grand Prix",
    "Bahrain Grand Prix",
    "Belgian Grand Prix",
    "British Grand Prix",
    "Canadian Grand Prix",
    "Chinese Grand Prix",
    "Dutch Grand Prix",
    "Emilia Romagna Grand Prix",
    "French Grand Prix",
    "Hungarian Grand Prix",
    "Italian Grand Prix",
    "Japanese Grand Prix",
    "Las Vegas Grand Prix",
    "Mexico City Grand Prix",
    "Miami Grand Prix",
    "Monaco Grand Prix",
    "Qatar Grand Prix",
    "Saudi Arabian Grand Prix",
    "Singapore Grand Prix",
    "Spanish Grand Prix",
    "United States Grand Prix",
]
COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
TEAMS = [
    "Red Bull Racing",
    "McLaren",
    "Ferrari",
    "Mercedes",
    "Aston Martin",
    "Alpine",
    "Williams",
    "RB",
    "Kick Sauber",
    "Haas F1 Team",
]

GROUP_HINTS: dict[str, str] = {
    "race_name": "race",
    "year": "race",
    "lap_number": "race",
    "total_laps": "race",
    "round_number": "race",
    "attacker_position": "positions",
    "defender_position": "positions",
    "gap_ahead": "positions",
    "attacker_tyre_compound": "attacker",
    "attacker_tyre_age": "attacker",
    "attacker_lap_time": "attacker",
    "attacker_fresh_tyre": "attacker",
    "attacker_stint": "attacker",
    "attacker_qualification_rank": "attacker",
    "attacker_team": "teams",
    "defender_tyre_compound": "defender",
    "defender_tyre_age": "defender",
    "defender_lap_time": "defender",
    "defender_fresh_tyre": "defender",
    "defender_stint": "defender",
    "defender_qualification_rank": "defender",
    "defender_team": "teams",
    "sector": "track",
    "air_temp": "weather",
    "track_temp": "weather",
    "humidity": "weather",
    "rainfall": "weather",
    "wind_speed": "weather",
    "safety_car": "flags",
    "yellow_flag": "flags",
    "gap_delta_1": "battle",
    "battle_duration": "battle",
    "overtakes_this_race": "battle",
    "gap_to_car_ahead": "situation",
    "gap_to_car_behind": "situation",
    "drs_train_size": "situation",
}


def _kind_for_feature(name: str, num_cols: set[str], cat_cols: set[str]) -> str:
    if name in cat_cols:
        if name in ("attacker_fresh_tyre", "defender_fresh_tyre"):
            return "boolean"
        return "category"
    if name in num_cols:
        return "number"
    return "string"


def _options_for(name: str) -> list[Any] | None:
    if name == "race_name":
        return list(RACES)
    if name in ("attacker_tyre_compound", "defender_tyre_compound"):
        return list(COMPOUNDS)
    if name in ("attacker_team", "defender_team"):
        return list(TEAMS)
    if name == "stint_phase":
        return ["fresh", "mid", "degraded", "cliff"]
    return None


def _range_for(name: str, default: Any) -> tuple[float | None, float | None]:
    ranges: dict[str, tuple[float, float]] = {
        "year": (2020, 2030),
        "lap_number": (1, 80),
        "total_laps": (1, 100),
        "attacker_position": (1, 20),
        "defender_position": (1, 20),
        "gap_ahead": (0.0, 5.0),
        "attacker_tyre_age": (0, 60),
        "defender_tyre_age": (0, 60),
        "attacker_lap_time": (70.0, 130.0),
        "defender_lap_time": (70.0, 130.0),
        "sector": (1, 3),
        "air_temp": (0.0, 50.0),
        "track_temp": (10.0, 70.0),
        "humidity": (0.0, 100.0),
        "wind_speed": (0.0, 30.0),
        "gap_delta_1": (-2.0, 2.0),
        "battle_duration": (1, 40),
        "gap_to_car_ahead": (0.0, 99.0),
        "gap_to_car_behind": (0.0, 99.0),
        "drs_train_size": (1, 20),
    }
    return ranges.get(name, (None, None))


def build_feature_schema(meta: dict[str, Any]) -> list[FeatureSchemaItem]:
    features: list[str] = list(meta.get("features") or [])
    nums = set(meta.get("num_cols") or [])
    cats = set(meta.get("cat_cols") or [])
    defaults = build_single_row({})

    items: list[FeatureSchemaItem] = []
    for name in features:
        kind = _kind_for_feature(name, nums, cats)
        default = defaults.get(name, 0)
        opts = _options_for(name)
        mn, mx = _range_for(name, default)
        if kind == "boolean":
            default = bool(default)
        items.append(
            FeatureSchemaItem(
                name=name,
                kind=kind,
                default=default,
                min=mn,
                max=mx,
                options=opts,
                group=GROUP_HINTS.get(name, "other"),
            )
        )
    return items
