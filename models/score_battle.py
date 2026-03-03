"""
Score a single F1 battle — returns the overtake probability.

Usage (interactive):
    python models/score_battle.py

Usage (as a one-liner, values passed via --option):
    python models/score_battle.py \\
        --race "Italian Grand Prix" \\
        --year 2025 \\
        --lap 35 --total-laps 53 \\
        --attacker-pos 8 --defender-pos 7 \\
        --gap -0.56 \\
        --attacker-compound HARD --attacker-tyre-age 20 \\
        --defender-compound HARD --defender-tyre-age 22 \\
        --attacker-lap-time 92.1 --defender-lap-time 92.8

Required inputs
---------------
  race              Grand Prix name, e.g. "Italian Grand Prix"
  year              Season year
  lap               Current lap number
  total-laps        Total scheduled laps in the race
  attacker-pos      Attacker's current race position
  defender-pos      Defender's current race position (must be attacker-pos - 1)
  gap               Gap to car ahead in seconds (negative = attacker is close behind)
  attacker-compound Tyre compound: SOFT / MEDIUM / HARD / INTERMEDIATE / WET
  attacker-tyre-age Laps on the attacker's current tyre
  defender-compound Tyre compound
  defender-tyre-age Laps on the defender's current tyre
  attacker-lap-time Attacker's last lap time in seconds
  defender-lap-time Defender's last lap time in seconds

All other features (speed traps, weather, stint, track info) are filled with
sensible defaults derived from historical averages and track metadata.
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent
_MODEL_PATH  = _HERE / "artifacts" / "overtake_model_v2.pkl"
_META_PATH   = _HERE / "artifacts" / "overtake_model_v2_meta.json"
_TRACK_INFO  = _HERE.parent / "src" / "pipeline" / "track_info.py"

# ── load track helpers without importing the module directly ───────────────────
sys.path.insert(0, str(_HERE.parent / "src"))
try:
    from pipeline.track_info import get_sector_type, get_track_type, get_drs_zone_info
except ImportError:
    def get_sector_type(track): return "mixed"
    def get_track_type(track):  return "street"
    def get_drs_zone_info(track, sector): return False, 0


# ── per-circuit average speed-trap values (mean over all laps, 2022-2024) ─────
# Key = lowercase words from the race name (partial match is enough)
_CIRCUIT_SPEEDS = {
    "abu dhabi":    {"i1": 227.9, "i2": 297.7, "fl": 210.1, "st": 304.4},
    "australian":   {"i1": 227.1, "i2": 294.9, "fl": 287.3, "st": 256.3},
    "azerbaijan":   {"i1": 182.8, "i2": 206.2, "fl": 321.0, "st": 318.3},
    "bahrain":      {"i1": 171.6, "i2": 249.6, "fl": 279.8, "st": 255.3},
    "belgian":      {"i1": 284.7, "i2": 194.1, "fl": 214.9, "st": 237.1},
    "british":      {"i1": 244.1, "i2": 251.5, "fl": 242.1, "st": 301.9},
    "canadian":     {"i1": 199.5, "i2": 271.8, "fl": 278.6, "st": 305.9},
    "dutch":        {"i1": 204.0, "i2": 263.9, "fl": 298.8, "st": 259.2},
    "french":       {"i1": 239.0, "i2": 274.8, "fl": 290.1, "st": 310.7},
    "hungarian":    {"i1": 217.5, "i2": 235.8, "fl": 244.9, "st": 218.3},
    "italian":      {"i1": 249.3, "i2": 316.6, "fl": 308.5, "st": 282.2},
    "japanese":     {"i1": 209.4, "i2": 297.2, "fl": 253.8, "st": 292.9},
    "mexico":       {"i1": 255.4, "i2": 279.1, "fl": 243.6, "st": 259.3},
    "monaco":       {"i1": 166.0, "i2": 176.4, "fl": 257.2, "st": 273.9},
    "saudi":        {"i1": 209.6, "i2": 303.3, "fl": 295.2, "st": 308.5},
    "singapore":    {"i1": 249.9, "i2": 260.8, "fl": 243.4, "st": 228.1},
    "spanish":      {"i1": 240.3, "i2": 268.1, "fl": 276.8, "st": 252.3},
    "united states":{"i1": 184.7, "i2": 179.2, "fl": 199.3, "st": 300.9},
}
_FALLBACK_SPEEDS = {"i1": 220.0, "i2": 265.0, "fl": 270.0, "st": 285.0}

# Soft tyres are ~0.5% faster than Medium, Hard ~0.5% slower
_COMPOUND_DELTA = {"SOFT": 1.005, "MEDIUM": 1.000, "HARD": 0.995,
                   "INTERMEDIATE": 0.925, "WET": 0.870}

_DEFAULT_WEATHER = {"air_temp": 25.0, "track_temp": 36.0, "humidity": 50.0,
                    "rainfall": False, "wind_speed": 2.0}


def _circuit_base(race_name: str) -> dict:
    name_lower = race_name.lower()
    for key, speeds in _CIRCUIT_SPEEDS.items():
        if key in name_lower:
            return speeds.copy()
    return _FALLBACK_SPEEDS.copy()


def _speeds(race: str, compound: str, lap_time: float, ref_time: float) -> dict:
    """
    Return realistic speed-trap estimates.
    1. Start from circuit-specific averages (captured absolute speed level).
    2. Apply compound delta (SOFT slightly faster than HARD).
    3. Scale by this car's pace relative to the session reference lap time.
    """
    base   = _circuit_base(race)
    c_mult = _COMPOUND_DELTA.get(compound.upper(), 1.0)
    # pace ratio: >1 means this car is faster than the reference
    pace   = (ref_time / lap_time) if lap_time and ref_time else 1.0
    return {k: round(v * c_mult * pace, 1) for k, v in base.items()}


def build_battle_row(
    race: str,
    year: int,
    lap: int,
    total_laps: int,
    attacker_pos: int,
    defender_pos: int,
    gap: float,
    attacker_compound: str,
    attacker_tyre_age: int,
    defender_compound: str,
    defender_tyre_age: int,
    attacker_lap_time: float,
    defender_lap_time: float,
    # optional overrides
    sector: int = 1,
    safety_car: bool = False,
    yellow_flag: bool = False,
    attacker_fresh_tyre: bool = False,
    defender_fresh_tyre: bool = False,
    attacker_stint: int = 1,
    defender_stint: int = 1,
    attacker_qual_rank: int = 10,
    defender_qual_rank: int = 9,
    weather: dict = None,
) -> pd.DataFrame:
    if weather is None:
        weather = _DEFAULT_WEATHER

    ref_time = (attacker_lap_time + defender_lap_time) / 2
    att_s = _speeds(race, attacker_compound, attacker_lap_time, ref_time)
    def_s = _speeds(race, defender_compound, defender_lap_time, ref_time)
    in_drs, drs_len = get_drs_zone_info(race, sector)

    row = {
        "year":                        year,
        "race_name":                   race,
        "lap_number":                  lap,
        "total_laps":                  total_laps,
        "race_progress":               lap / total_laps,
        "attacker_position":           attacker_pos,
        "defender_position":           defender_pos,
        "attacker_lap_time":           attacker_lap_time,
        "defender_lap_time":           defender_lap_time,
        "gap_ahead":                   gap,
        "attacker_speed_i1":           att_s["i1"],
        "defender_speed_i1":           def_s["i1"],
        "attacker_speed_i2":           att_s["i2"],
        "defender_speed_i2":           def_s["i2"],
        "attacker_finish_line_speed":  att_s["fl"],
        "defender_finish_line_speed":  def_s["fl"],
        "attacker_straight_speed":     att_s["st"],
        "defender_straight_speed":     def_s["st"],
        "safety_car":                  safety_car,
        "yellow_flag":                 yellow_flag,
        "attacker_tyre_compound":      attacker_compound.upper(),
        "defender_tyre_compound":      defender_compound.upper(),
        "attacker_tyre_age":           attacker_tyre_age,
        "defender_tyre_age":           defender_tyre_age,
        "tyre_age_difference":         attacker_tyre_age - defender_tyre_age,
        "attacker_stint":              attacker_stint,
        "defender_stint":              defender_stint,
        "attacker_fresh_tyre":         attacker_fresh_tyre,
        "defender_fresh_tyre":         defender_fresh_tyre,
        "sector":                      sector,
        "sector_type":                 get_sector_type(race),
        "is_in_drs_zone":              in_drs,
        "drs_zone_length":             drs_len,
        "track_type":                  get_track_type(race),
        "air_temp":                    weather["air_temp"],
        "track_temp":                  weather["track_temp"],
        "humidity":                    weather["humidity"],
        "rainfall":                    weather["rainfall"],
        "wind_speed":                  weather["wind_speed"],
        "qualification_rank_difference": attacker_qual_rank - defender_qual_rank,
    }
    return pd.DataFrame([row])


def score(row_df: pd.DataFrame) -> float:
    if not _MODEL_PATH.exists():
        print("Model not found. Run all cells in model_testing_2.ipynb first.")
        sys.exit(1)
    pipeline = joblib.load(_MODEL_PATH)
    meta     = json.loads(_META_PATH.read_text())
    proba    = pipeline.predict_proba(row_df[meta["features"]])[:, 1][0]
    return float(proba)


def interpret(proba: float, threshold: float) -> str:
    if proba < 0.05:
        label = "Very unlikely"
    elif proba < 0.15:
        label = "Possible"
    elif proba < 0.30:
        label = "Likely"
    elif proba < 0.50:
        label = "Very likely"
    else:
        label = "Highly likely"
    decision = "OVERTAKE PREDICTED" if proba >= threshold else "no overtake predicted"
    return f"{label} ({decision})"


def print_result(proba: float, threshold: float, row: dict):
    bar_len  = int(proba * 40)
    bar      = "█" * bar_len + "░" * (40 - bar_len)
    verdict  = interpret(proba, threshold)
    print()
    print("┌─────────────────────────────────────────────────┐")
    print(f"│  Race   : {row.get('race_name','?'):<38} │")
    print(f"│  Lap    : {row.get('lap_number','?'):<5}  /  {row.get('total_laps','?'):<5}  "
          f"({row.get('race_progress', 0):.0%} race progress)  │")
    print(f"│  Pos    : P{row.get('attacker_position','?')} (attacker) vs P{row.get('defender_position','?')} (defender)  │")
    print(f"│  Gap    : {row.get('gap_ahead', 0):.3f}s                                  │")
    print(f"│  Tyres  : {row.get('attacker_tyre_compound','?')} ({row.get('attacker_tyre_age','?')} laps) "
          f"vs {row.get('defender_tyre_compound','?')} ({row.get('defender_tyre_age','?')} laps)  │")
    print("├─────────────────────────────────────────────────┤")
    print(f"│  Overtake probability: {proba*100:5.1f}%                    │")
    print(f"│  [{bar}]  │")
    print(f"│  {verdict:<47} │")
    print("└─────────────────────────────────────────────────┘")
    print()


def interactive_mode(threshold: float):
    print("\n=== F1 Overtake Probability Calculator ===")
    print("Press Ctrl-C to quit.\n")

    def ask(prompt, cast=str, default=None):
        suffix = f" [{default}]" if default is not None else ""
        while True:
            raw = input(f"  {prompt}{suffix}: ").strip()
            if raw == "" and default is not None:
                return default
            try:
                return cast(raw)
            except ValueError:
                print(f"    ✗ Expected {cast.__name__}, got {raw!r}")

    race              = ask("Grand Prix name (e.g. Italian Grand Prix)")
    year              = ask("Season year", int, 2025)
    lap               = ask("Current lap", int)
    total_laps        = ask("Total laps in race", int)
    attacker_pos      = ask("Attacker position (e.g. 8)", int)
    defender_pos      = ask("Defender position (e.g. 7)", int)
    gap               = ask("Gap to car ahead in seconds (e.g. -0.6)", float)
    attacker_compound = ask("Attacker tyre compound [SOFT/MEDIUM/HARD]", str, "MEDIUM")
    attacker_tyre_age = ask("Attacker tyre age (laps)", int)
    defender_compound = ask("Defender tyre compound [SOFT/MEDIUM/HARD]", str, "MEDIUM")
    defender_tyre_age = ask("Defender tyre age (laps)", int)
    attacker_lap_time = ask("Attacker last lap time in seconds (e.g. 92.1)", float)
    defender_lap_time = ask("Defender last lap time in seconds (e.g. 92.8)", float)

    print("\n  Optional (press Enter to use defaults):")
    attacker_qual = ask("Attacker quali rank (e.g. 8)", int, attacker_pos)
    defender_qual = ask("Defender quali rank (e.g. 7)", int, defender_pos)
    safety_car    = ask("Safety car active? [y/n]", lambda x: x.lower() == "y", False)
    rainfall      = ask("Wet race? [y/n]", lambda x: x.lower() == "y", False)

    row = build_battle_row(
        race=race, year=year, lap=lap, total_laps=total_laps,
        attacker_pos=attacker_pos, defender_pos=defender_pos, gap=gap,
        attacker_compound=attacker_compound, attacker_tyre_age=attacker_tyre_age,
        defender_compound=defender_compound, defender_tyre_age=defender_tyre_age,
        attacker_lap_time=attacker_lap_time, defender_lap_time=defender_lap_time,
        attacker_qual_rank=attacker_qual, defender_qual_rank=defender_qual,
        safety_car=safety_car, weather={**_DEFAULT_WEATHER, "rainfall": rainfall},
    )
    proba = score(row)
    print_result(proba, threshold, row.iloc[0].to_dict())

    again = input("  Score another battle? [y/n]: ").strip().lower()
    if again == "y":
        interactive_mode(threshold)


def main():
    meta      = json.loads(_META_PATH.read_text()) if _META_PATH.exists() else {}
    threshold = meta.get("threshold", 0.5)

    parser = argparse.ArgumentParser(description="Score a single F1 battle")
    parser.add_argument("--race",               type=str,   default=None)
    parser.add_argument("--year",               type=int,   default=2025)
    parser.add_argument("--lap",                type=int,   default=None)
    parser.add_argument("--total-laps",         type=int,   default=None)
    parser.add_argument("--attacker-pos",       type=int,   default=None)
    parser.add_argument("--defender-pos",       type=int,   default=None)
    parser.add_argument("--gap",                type=float, default=None)
    parser.add_argument("--attacker-compound",  type=str,   default="MEDIUM")
    parser.add_argument("--attacker-tyre-age",  type=int,   default=10)
    parser.add_argument("--defender-compound",  type=str,   default="MEDIUM")
    parser.add_argument("--defender-tyre-age",  type=int,   default=10)
    parser.add_argument("--attacker-lap-time",  type=float, default=92.0)
    parser.add_argument("--defender-lap-time",  type=float, default=92.5)
    args = parser.parse_args()

    # If any required arg is missing → interactive mode
    required = [args.race, args.lap, args.total_laps, args.attacker_pos,
                args.defender_pos, args.gap]
    if any(v is None for v in required):
        interactive_mode(threshold)
        return

    row = build_battle_row(
        race=args.race, year=args.year, lap=args.lap, total_laps=args.total_laps,
        attacker_pos=args.attacker_pos, defender_pos=args.defender_pos, gap=args.gap,
        attacker_compound=args.attacker_compound,
        attacker_tyre_age=args.attacker_tyre_age,
        defender_compound=args.defender_compound,
        defender_tyre_age=args.defender_tyre_age,
        attacker_lap_time=args.attacker_lap_time,
        defender_lap_time=args.defender_lap_time,
    )
    proba = score(row)
    print_result(proba, threshold, row.iloc[0].to_dict())


if __name__ == "__main__":
    main()
