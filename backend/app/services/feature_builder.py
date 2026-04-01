"""
Feature engineering for inference (ported from legacy models/app.py).

Builds a full battle row from user inputs, then aligns to model `features` list.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .circuit_calendar import CIRCUIT_CALENDAR_2025

try:
    from pipeline.constructor_standings import constructor_position_for_team
except ImportError:  # pragma: no cover

    def constructor_position_for_team(year: int, app_team_name: str) -> int | None:
        return None


try:
    from pipeline.track_info import get_drs_zone_info, get_sector_type, get_track_type
except ImportError:  # pragma: no cover

    def get_sector_type(track: str) -> str:
        return "mixed"

    def get_track_type(track: str) -> str:
        return "street"

    def get_drs_zone_info(track: str, sector: int):
        return False, 0


TYRE_PACE_RANK = {
    "SOFT": 0,
    "MEDIUM": 1,
    "HARD": 2,
    "INTERMEDIATE": 1.5,
    "WET": 2,
    "UNKNOWN": 1,
}
TYRE_CLIFF = {"SOFT": 18, "MEDIUM": 28, "HARD": 38, "INTERMEDIATE": 30, "WET": 25}

_CIRCUIT_SPEEDS = {
    "abu dhabi": {"i1": 227.9, "i2": 297.7, "fl": 210.1, "st": 304.4},
    "australian": {"i1": 227.1, "i2": 294.9, "fl": 287.3, "st": 256.3},
    "austrian": {"i1": 240.0, "i2": 270.0, "fl": 260.0, "st": 290.0},
    "azerbaijan": {"i1": 182.8, "i2": 206.2, "fl": 321.0, "st": 318.3},
    "bahrain": {"i1": 171.6, "i2": 249.6, "fl": 279.8, "st": 255.3},
    "belgian": {"i1": 284.7, "i2": 194.1, "fl": 214.9, "st": 237.1},
    "british": {"i1": 244.1, "i2": 251.5, "fl": 242.1, "st": 301.9},
    "canadian": {"i1": 199.5, "i2": 271.8, "fl": 278.6, "st": 305.9},
    "chinese": {"i1": 210.0, "i2": 280.0, "fl": 275.0, "st": 310.0},
    "dutch": {"i1": 204.0, "i2": 263.9, "fl": 298.8, "st": 259.2},
    "emilia": {"i1": 220.0, "i2": 280.0, "fl": 270.0, "st": 310.0},
    "french": {"i1": 239.0, "i2": 274.8, "fl": 290.1, "st": 310.7},
    "hungarian": {"i1": 217.5, "i2": 235.8, "fl": 244.9, "st": 218.3},
    "italian": {"i1": 249.3, "i2": 316.6, "fl": 308.5, "st": 282.2},
    "japanese": {"i1": 209.4, "i2": 297.2, "fl": 253.8, "st": 292.9},
    "las vegas": {"i1": 220.0, "i2": 280.0, "fl": 290.0, "st": 320.0},
    "mexico": {"i1": 255.4, "i2": 279.1, "fl": 243.6, "st": 259.3},
    "miami": {"i1": 210.0, "i2": 270.0, "fl": 280.0, "st": 300.0},
    "monaco": {"i1": 166.0, "i2": 176.4, "fl": 257.2, "st": 273.9},
    "qatar": {"i1": 220.0, "i2": 280.0, "fl": 275.0, "st": 310.0},
    "saudi": {"i1": 209.6, "i2": 303.3, "fl": 295.2, "st": 308.5},
    "singapore": {"i1": 249.9, "i2": 260.8, "fl": 243.4, "st": 228.1},
    "spanish": {"i1": 240.3, "i2": 268.1, "fl": 276.8, "st": 252.3},
    "united states": {"i1": 184.7, "i2": 179.2, "fl": 199.3, "st": 300.9},
    "são paulo": {"i1": 255.0, "i2": 275.0, "fl": 245.0, "st": 260.0},
    "sao paulo": {"i1": 255.0, "i2": 275.0, "fl": 245.0, "st": 260.0},
}
_FALLBACK_SPEEDS = {"i1": 220.0, "i2": 265.0, "fl": 270.0, "st": 285.0}
_COMPOUND_DELTA = {
    "SOFT": 1.005,
    "MEDIUM": 1.000,
    "HARD": 0.995,
    "INTERMEDIATE": 0.925,
    "WET": 0.870,
}


def _circuit_base(race_name: str) -> dict:
    name_lower = race_name.lower()
    for key, speeds in _CIRCUIT_SPEEDS.items():
        if key in name_lower:
            return speeds.copy()
    return _FALLBACK_SPEEDS.copy()


def _estimate_speeds(race: str, compound: str, lap_time: float, ref_time: float) -> dict:
    base = _circuit_base(race)
    c_mult = _COMPOUND_DELTA.get(compound.upper(), 1.0)
    pace = (ref_time / lap_time) if lap_time and ref_time else 1.0
    return {k: round(v * c_mult * pace, 1) for k, v in base.items()}


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v in (None, "", 0, "0", "false", "False"):
        return False
    return bool(v)


def clean_raw_inputs(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop None and blank strings so `raw.get(key, default)` uses defaults (avoids int(''))."""
    out: dict[str, Any] = {}
    for k, v in raw.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        out[k] = v
    return out


def build_single_row(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Construct one logical battle row from UI/API input (snake_case keys)."""
    raw = clean_raw_inputs(dict(raw))
    race = str(raw.get("race_name", "Italian Grand Prix"))
    cal = CIRCUIT_CALENDAR_2025.get(race)
    year = int(raw.get("year", 2025))
    lap = int(raw.get("lap_number", 35))
    if cal is not None:
        total_laps = int(raw.get("total_laps", cal["total_laps"]))
        round_number = int(raw.get("round_number", cal["round"]))
    else:
        total_laps = int(raw.get("total_laps", 53))
        round_number = int(raw.get("round_number", 0))
    attacker_pos = int(raw.get("attacker_position", 8))
    defender_pos = int(raw.get("defender_position", 7))
    gap = float(raw.get("gap_ahead", 0.56))
    attacker_compound = str(raw.get("attacker_tyre_compound", "HARD")).upper()
    defender_compound = str(raw.get("defender_tyre_compound", "HARD")).upper()
    attacker_tyre_age = int(raw.get("attacker_tyre_age", 20))
    defender_tyre_age = int(raw.get("defender_tyre_age", 22))
    attacker_lap_time = float(raw.get("attacker_lap_time", 92.1))
    defender_lap_time = float(raw.get("defender_lap_time", 92.8))
    sector = int(raw.get("sector", 1))
    safety_car = _coerce_bool(raw.get("safety_car", False))
    yellow_flag = _coerce_bool(raw.get("yellow_flag", False))
    attacker_fresh_tyre = _coerce_bool(raw.get("attacker_fresh_tyre", False))
    defender_fresh_tyre = _coerce_bool(raw.get("defender_fresh_tyre", False))
    attacker_stint = int(raw.get("attacker_stint", 2))
    defender_stint = int(raw.get("defender_stint", 2))
    attacker_qual_rank = int(raw.get("attacker_qualification_rank", 8))
    defender_qual_rank = int(raw.get("defender_qualification_rank", 7))
    qual_diff_raw = raw.get("qualification_rank_difference")
    if qual_diff_raw is not None and str(qual_diff_raw).strip() != "":
        qualification_rank_difference = int(float(qual_diff_raw))
    else:
        qualification_rank_difference = attacker_qual_rank - defender_qual_rank
    air_temp = float(raw.get("air_temp", 25.0))
    track_temp = float(raw.get("track_temp", 36.0))
    humidity = float(raw.get("humidity", 50.0))
    rainfall = _coerce_bool(raw.get("rainfall", False))
    wind_speed = float(raw.get("wind_speed", 2.0))
    gap_delta_1 = float(raw.get("gap_delta_1", raw.get("closing_rate", -0.2)))
    battle_duration = int(raw.get("battle_duration", 5))
    overtakes_this_race = int(raw.get("overtakes_this_race", 5))
    attacker_team = str(raw.get("attacker_team", "McLaren"))
    defender_team = str(raw.get("defender_team", "Ferrari"))

    # Avoid network on `build_single_row({})` (schema defaults).
    if raw:
        acr = constructor_position_for_team(year, attacker_team)
        dcr = constructor_position_for_team(year, defender_team)
        att_cons = int(acr) if acr is not None else 10
        def_cons = int(dcr) if dcr is not None else 10
    else:
        att_cons, def_cons = 10, 10
    gap_to_car_ahead = float(raw.get("gap_to_car_ahead", 2.5))
    gap_to_car_behind = float(raw.get("gap_to_car_behind", 1.8))
    drs_train_size = int(raw.get("drs_train_size", 2))

    ref_time = (attacker_lap_time + defender_lap_time) / 2
    att_s = _estimate_speeds(race, attacker_compound, attacker_lap_time, ref_time)
    def_s = _estimate_speeds(race, defender_compound, defender_lap_time, ref_time)
    in_drs, drs_len = get_drs_zone_info(race, sector)

    race_progress = round(lap / total_laps, 4) if total_laps > 0 else 0.0
    pace_delta = defender_lap_time - attacker_lap_time
    is_closing = 1 if gap_delta_1 < 0 else 0
    tyre_age_diff = attacker_tyre_age - defender_tyre_age

    att_pace_rank = TYRE_PACE_RANK.get(attacker_compound, 1)
    def_pace_rank = TYRE_PACE_RANK.get(defender_compound, 1)
    compound_advantage = def_pace_rank - att_pace_rank
    cliff_thresh = TYRE_CLIFF.get(defender_compound, 28)
    tyre_cliff_risk = 1 if defender_tyre_age > cliff_thresh else 0
    attacker_on_newer_stint = 1 if attacker_stint > defender_stint else 0

    if attacker_tyre_age <= 5:
        stint_phase = "fresh"
    elif attacker_tyre_age <= 15:
        stint_phase = "mid"
    elif attacker_tyre_age <= 25:
        stint_phase = "degraded"
    else:
        stint_phase = "cliff"

    row: dict[str, Any] = {
        "year": year,
        "race_name": race,
        "round_number": round_number,
        "lap_number": lap,
        "total_laps": total_laps,
        "race_progress": race_progress,
        "attacker_position": attacker_pos,
        "defender_position": defender_pos,
        "attacker_lap_time": attacker_lap_time,
        "defender_lap_time": defender_lap_time,
        "gap_ahead": gap,
        "pace_delta": pace_delta,
        "attacker_speed_i1": att_s["i1"],
        "defender_speed_i1": def_s["i1"],
        "attacker_speed_i2": att_s["i2"],
        "defender_speed_i2": def_s["i2"],
        "attacker_finish_line_speed": att_s["fl"],
        "defender_finish_line_speed": def_s["fl"],
        "attacker_straight_speed": att_s["st"],
        "defender_straight_speed": def_s["st"],
        "speed_i1_delta": att_s["i1"] - def_s["i1"],
        "speed_i2_delta": att_s["i2"] - def_s["i2"],
        "speed_fl_delta": att_s["fl"] - def_s["fl"],
        "speed_st_delta": att_s["st"] - def_s["st"],
        "safety_car": safety_car,
        "yellow_flag": yellow_flag,
        "attacker_tyre_compound": attacker_compound,
        "defender_tyre_compound": defender_compound,
        "attacker_tyre_age": attacker_tyre_age,
        "defender_tyre_age": defender_tyre_age,
        "tyre_age_difference": tyre_age_diff,
        "attacker_stint": attacker_stint,
        "defender_stint": defender_stint,
        "attacker_fresh_tyre": attacker_fresh_tyre,
        "defender_fresh_tyre": defender_fresh_tyre,
        "sector": sector,
        "sector_type": get_sector_type(race),
        "is_in_drs_zone": in_drs,
        "drs_zone_length": drs_len,
        "track_type": get_track_type(race),
        "air_temp": air_temp,
        "track_temp": track_temp,
        "humidity": humidity,
        "rainfall": rainfall,
        "wind_speed": wind_speed,
        "gap_delta_1": gap_delta_1,
        "gap_delta_3": gap_delta_1,
        "is_closing": is_closing,
        "closing_laps": is_closing,
        "pace_delta_avg_3": pace_delta,
        "battle_duration": battle_duration,
        "attempted_before": 0,
        "overtakes_this_race": overtakes_this_race,
        "compound_advantage": compound_advantage,
        "tyre_cliff_risk": tyre_cliff_risk,
        "attacker_on_newer_stint": attacker_on_newer_stint,
        "qualification_rank_difference": qualification_rank_difference,
        "attacker_team": attacker_team,
        "defender_team": defender_team,
        "attacker_constructor_rank": att_cons,
        "defender_constructor_rank": def_cons,
        "constructor_rank_delta": att_cons - def_cons,
        "same_team": attacker_team == defender_team,
        "gap_to_car_ahead": gap_to_car_ahead,
        "gap_to_car_behind": gap_to_car_behind,
        "drs_train_size": drs_train_size,
        "stint_phase": raw.get("stint_phase", stint_phase),
        "sector1_delta": float(raw.get("sector1_delta", 0.0)),
        "sector2_delta": float(raw.get("sector2_delta", 0.0)),
        "sector3_delta": float(raw.get("sector3_delta", 0.0)),
        "strongest_sector": int(raw.get("strongest_sector", -1)),
        "attacker_overtake_rate_last5": float(raw.get("attacker_overtake_rate_last5", 0.5)),
        "defender_defend_rate_last5": float(raw.get("defender_defend_rate_last5", 0.5)),
        "attacker_team_pace_rank": float(raw.get("attacker_team_pace_rank", 5.0)),
        "defender_team_pace_rank": float(raw.get("defender_team_pace_rank", 5.0)),
        "team_delta": float(raw.get("team_delta", 0.0)),
        "attacker_positions_gained_avg": float(raw.get("attacker_positions_gained_avg", 0.0)),
        "defender_positions_gained_avg": float(raw.get("defender_positions_gained_avg", 0.0)),
        "attacker_quali_vs_teammate": float(raw.get("attacker_quali_vs_teammate", 0.0)),
        "defender_quali_vs_teammate": float(raw.get("defender_quali_vs_teammate", 0.0)),
        "attacker_race_pace_vs_teammate": float(raw.get("attacker_race_pace_vs_teammate", 0.0)),
        "defender_race_pace_vs_teammate": float(raw.get("defender_race_pace_vs_teammate", 0.0)),
        "closing_rate": float(raw.get("closing_rate", gap_delta_1)),
    }
    return row


def engineer_batch_features(df: pd.DataFrame) -> pd.DataFrame:
    """Gap trends, battle groups, tyre features — legacy batch path."""
    df = df.copy()

    if "qualification_rank_difference" not in df.columns:
        if "attacker_qualification_rank" in df.columns and "defender_qualification_rank" in df.columns:
            df["qualification_rank_difference"] = (
                df["attacker_qualification_rank"] - df["defender_qualification_rank"]
            )

    if "pace_delta" not in df.columns:
        if "defender_lap_time" in df.columns and "attacker_lap_time" in df.columns:
            df["pace_delta"] = df["defender_lap_time"] - df["attacker_lap_time"]

    for delta, att, dfn in [
        ("speed_i1_delta", "attacker_speed_i1", "defender_speed_i1"),
        ("speed_i2_delta", "attacker_speed_i2", "defender_speed_i2"),
        ("speed_fl_delta", "attacker_finish_line_speed", "defender_finish_line_speed"),
        ("speed_st_delta", "attacker_straight_speed", "defender_straight_speed"),
    ]:
        if delta not in df.columns and att in df.columns and dfn in df.columns:
            df[delta] = df[att] - df[dfn]

    if "year" in df.columns and "race_name" in df.columns:
        df = df.sort_values(["year", "race_name", "attacker", "defender", "lap_number"]).copy()
        df["_pair"] = (
            df["attacker"].astype(str)
            + "_vs_"
            + df["defender"].astype(str)
            + "_"
            + df["race_name"].astype(str)
            + "_"
            + df["year"].astype(str)
        )
        df["_lap_gap"] = df.groupby("_pair")["lap_number"].diff().fillna(99)
        df["_break"] = (df["_lap_gap"] != 1).astype(int)
        df["_seq"] = df.groupby("_pair")["_break"].cumsum()
        df["_bg"] = df["_pair"] + "_" + df["_seq"].astype(str)
    else:
        df["_bg"] = range(len(df))

    if "gap_ahead" in df.columns:
        grp = df.groupby("_bg")["gap_ahead"]
        df["gap_delta_1"] = grp.diff(1).fillna(0)
        df["gap_delta_3"] = grp.diff(3).fillna(0)
        df["is_closing"] = (df["gap_delta_1"] < 0).astype(int)
        df["closing_laps"] = (
            df.groupby("_bg")["is_closing"].transform(lambda s: s.rolling(3, min_periods=1).sum())
        )

    if "pace_delta" in df.columns:
        df["pace_delta_avg_3"] = (
            df.groupby("_bg")["pace_delta"].transform(lambda s: s.rolling(3, min_periods=1).mean())
        )

    df["battle_duration"] = df.groupby("_bg").cumcount() + 1

    if "overtake" in df.columns:
        race_pair = (
            df["attacker"].astype(str)
            + "_"
            + df["defender"].astype(str)
            + "_"
            + df["race_name"].astype(str)
            + "_"
            + df["year"].astype(str)
        )
        df["_rp"] = race_pair
        df["attempted_before"] = (
            df.sort_values("lap_number")
            .groupby("_rp")["overtake"]
            .transform(lambda s: s.shift(1).cummax().fillna(0))
            .astype(int)
        )
        race_key = df["race_name"].astype(str) + "_" + df["year"].astype(str)
        df["_rk"] = race_key
        df["overtakes_this_race"] = (
            df.sort_values("lap_number")
            .groupby("_rk")["overtake"]
            .transform(lambda s: s.shift(1).cumsum().fillna(0))
            .astype(int)
        )
        df.drop(columns=["_rp", "_rk"], inplace=True, errors="ignore")
    else:
        df["attempted_before"] = 0
        df["overtakes_this_race"] = 0

    att_pace = df["attacker_tyre_compound"].map(TYRE_PACE_RANK).fillna(1)
    def_pace = df["defender_tyre_compound"].map(TYRE_PACE_RANK).fillna(1)
    df["compound_advantage"] = def_pace - att_pace
    cliff = df["defender_tyre_compound"].map(TYRE_CLIFF).fillna(28)
    df["tyre_cliff_risk"] = (df["defender_tyre_age"] > cliff).astype(int)
    df["attacker_on_newer_stint"] = 0
    if "attacker_stint" in df.columns and "defender_stint" in df.columns:
        df["attacker_on_newer_stint"] = (df["attacker_stint"] > df["defender_stint"]).astype(int)

    if "closing_rate" not in df.columns and "gap_delta_1" in df.columns:
        df["closing_rate"] = df["gap_delta_1"]

    df.drop(columns=["_pair", "_lap_gap", "_break", "_seq", "_bg"], inplace=True, errors="ignore")
    return df


def dataframe_for_model(row_dict: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([row_dict])
    for c in feature_names:
        if c not in df.columns:
            df[c] = 0
    return df[feature_names]
