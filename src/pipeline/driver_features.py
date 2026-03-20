"""
IP03 §3.3 — driver skill proxies without identity leakage.

Rolling 5-race rates computed from strictly past races only (ordered by year, round).
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

DEFAULT_RATE = 0.5


def enrich_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add:
      - attacker_overtake_rate_last5: mean(overtake) over prior rows where attacker is attacker,
        in the previous 5 races (calendar order) before this row's race.
      - defender_defend_rate_last5: mean(1 - overtake) = defender held position, same window.
    """
    if df.empty:
        return df

    need = {"year", "round_number", "lap_number", "attacker", "defender", "overtake"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"enrich_driver_features: missing columns {missing}")

    df = df.sort_values(["year", "round_number", "lap_number"], kind="mergesort").reset_index(drop=True)

    race_keys: list[tuple[int, int]] = (
        df[["year", "round_number"]].drop_duplicates().apply(tuple, axis=1).tolist()
    )
    race_key_to_idx = {rk: i for i, rk in enumerate(race_keys)}

    by_race: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, r in df.iterrows():
        by_race[(int(r["year"]), int(r["round_number"]))].append(idx)

    att_rates: list[float] = []
    def_rates: list[float] = []

    for idx, row in df.iterrows():
        rk = (int(row["year"]), int(row["round_number"]))
        ri = race_key_to_idx[rk]
        prev_race_keys = race_keys[max(0, ri - 5) : ri]

        prev_idx: list[int] = []
        for pr in prev_race_keys:
            prev_idx.extend(by_race[pr])

        if prev_idx:
            sub = df.loc[prev_idx]
            att_sub = sub[sub["attacker"] == row["attacker"]]
            def_sub = sub[sub["defender"] == row["defender"]]
            att_r = float(att_sub["overtake"].astype(bool).mean()) if len(att_sub) else DEFAULT_RATE
            def_r = float((~def_sub["overtake"].astype(bool)).mean()) if len(def_sub) else DEFAULT_RATE
        else:
            att_r = def_r = DEFAULT_RATE

        att_rates.append(att_r)
        def_rates.append(def_r)

    out = df.copy()
    out["attacker_overtake_rate_last5"] = att_rates
    out["defender_defend_rate_last5"] = def_rates
    return out
