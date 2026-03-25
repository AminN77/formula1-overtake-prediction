"""Build UI schema from model meta + default row."""

from __future__ import annotations

from typing import Any

from ..schemas.model_info import FeatureSchemaItem
from .circuit_calendar import CIRCUIT_CALENDAR_2025
from .feature_builder import build_single_row
from .feature_metadata import (
    BASIC_FEATURE_NAMES,
    DERIVED_FROM,
    READONLY_FEATURE_NAMES,
    description_for,
    label_for,
)

# Championship order for dropdowns (2025 calendar).
RACES = sorted(
    CIRCUIT_CALENDAR_2025.keys(),
    key=lambda n: int(CIRCUIT_CALENDAR_2025[n]["round"]),
)
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

# Shown in the UI + sent to `build_single_row` even if not yet in older model artifacts (IP06).
SUPPLEMENTAL_FEATURES: tuple[str, ...] = (
    "gap_delta_1",
    "attacker_qualification_rank",
    "defender_qualification_rank",
    "attacker_constructor_rank",
    "defender_constructor_rank",
    "constructor_rank_delta",
)

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
    "attacker_constructor_rank": "teams",
    "defender_constructor_rank": "teams",
    "constructor_rank_delta": "teams",
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
        "attacker_qualification_rank": (1, 20),
        "defender_qualification_rank": (1, 20),
        "attacker_constructor_rank": (1, 20),
        "defender_constructor_rank": (1, 20),
        "constructor_rank_delta": (-20, 20),
        "qualification_rank_difference": (-20, 20),
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
    seen = set(features)
    for name in SUPPLEMENTAL_FEATURES:
        if name not in seen:
            features.append(name)
            seen.add(name)

    nums = set(meta.get("num_cols") or [])
    cats = set(meta.get("cat_cols") or [])
    defaults = build_single_row({})
    for name in SUPPLEMENTAL_FEATURES:
        v = defaults.get(name)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            nums.add(name)

    items: list[FeatureSchemaItem] = []
    for name in features:
        kind = _kind_for_feature(name, nums, cats)
        default = defaults.get(name, 0)
        opts = _options_for(name)
        mn, mx = _range_for(name, default)
        if kind == "boolean":
            default = bool(default)  # type: ignore[assignment]

        readonly = name in READONLY_FEATURE_NAMES
        advanced = name not in BASIC_FEATURE_NAMES
        derived = DERIVED_FROM.get(name)

        items.append(
            FeatureSchemaItem(
                name=name,
                kind=kind,
                default=default,
                min=mn,
                max=mx,
                options=opts,
                group=GROUP_HINTS.get(name, "other"),
                label=label_for(name),
                description=description_for(name),
                readonly=readonly,
                advanced=advanced,
                derived_from=derived,
            )
        )
    return items
