"""
IP03 §1.1 — aggregate lap-level battles into one row per battle sequence.

Group key: (year, race_name, attacker, defender).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_battle_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build battle-pair level rows from lap-level data.

    Label: any(overtake) within the sequence (did an overtake occur in this battle at any lap).
    """
    keys = ["year", "race_name", "attacker", "defender"]
    df = df.sort_values(keys + ["lap_number"], kind="mergesort")

    rows = []
    for _, g in df.groupby(keys, sort=False):
        g = g.sort_values("lap_number")
        gaps = g["gap_ahead"].astype(float).values
        pace = g["pace_delta"].astype(float).values
        st = g["speed_st_delta"].astype(float).values
        laps = np.arange(len(g), dtype=float)

        if len(gaps) > 1:
            gap_slope = float((gaps[-1] - gaps[0]) / (len(gaps) - 1))
            closing = np.sum(np.diff(gaps) < 0)
        else:
            gap_slope = 0.0
            closing = 0

        drs_frac = float(g["is_in_drs_zone"].astype(bool).mean()) if "is_in_drs_zone" in g else 0.0

        row = {
            "year": int(g["year"].iloc[0]),
            "race_name": g["race_name"].iloc[0],
            "attacker": g["attacker"].iloc[0],
            "defender": g["defender"].iloc[0],
            "round_number": int(g["round_number"].iloc[0]) if "round_number" in g else 0,
            "track": g["track"].iloc[0] if "track" in g else "",
            "track_type": g["track_type"].iloc[0] if "track_type" in g else "",
            "min_gap": float(np.min(gaps)),
            "mean_gap": float(np.mean(gaps)),
            "last_gap": float(gaps[-1]),
            "gap_slope": gap_slope,
            "max_pace_delta": float(np.max(pace)),
            "mean_pace_delta": float(np.mean(pace)),
            "n_laps": int(len(g)),
            "n_closing_laps": int(closing),
            "max_speed_delta_st": float(np.max(st)) if len(st) else 0.0,
            "tyre_age_at_start_att": int(g["attacker_tyre_age"].iloc[0]),
            "tyre_age_at_end_att": int(g["attacker_tyre_age"].iloc[-1]),
            "compound_mismatch": bool(g["compound_mismatch"].iloc[0]) if "compound_mismatch" in g else False,
            "drs_available_fraction": drs_frac,
            "overtake_pair": bool(g["overtake"].astype(bool).max()),
            "overtake_within_3_pair": bool(g["overtake_within_3"].astype(bool).max())
            if "overtake_within_3" in g
            else bool(g["overtake"].astype(bool).max()),
        }
        rows.append(row)

    return pd.DataFrame(rows)
