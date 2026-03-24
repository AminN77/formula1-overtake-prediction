"""
IP04 §1.1 / §1.2 — team-level and richer driver form features.

Computed from strictly prior races (leak-safe), like driver_features.py.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

DEFAULT_RANK = 10.0
DEFAULT_RATE = 0.0


def _race_key(row) -> tuple[int, int]:
    return (int(row["year"]), int(row["round_number"]))


def enrich_team_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add team-level and richer driver form columns to the battle DataFrame.

    New columns (all leak-safe, using only prior races):
      Team performance (IP04 §1.1):
        attacker_team_pace_rank   — team average race-pace rank, season-to-date
        defender_team_pace_rank
        team_delta                — attacker rank − defender rank (negative = attacker in faster team)

      Richer driver form (IP04 §1.2):
        attacker_positions_gained_avg   — mean(grid − finish) over last 5 races
        defender_positions_gained_avg
        attacker_quali_vs_teammate      — mean(teammate grid − driver grid) last 5 races
        defender_quali_vs_teammate
        attacker_race_pace_vs_teammate  — mean(teammate avgLap − driver avgLap) last 5 races
        defender_race_pace_vs_teammate
    """
    if df.empty:
        return df

    need = {
        "year", "round_number", "attacker", "defender",
        "attacker_team", "defender_team",
        "attacker_lap_time", "defender_lap_time",
        "attacker_qualification_rank", "defender_qualification_rank",
        "attacker_position", "defender_position",
    }
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"enrich_team_features: missing columns {missing}")

    df = df.sort_values(
        ["year", "round_number", "lap_number"], kind="mergesort"
    ).reset_index(drop=True)

    race_keys: list[tuple[int, int]] = (
        df[["year", "round_number"]]
        .drop_duplicates()
        .apply(tuple, axis=1)
        .tolist()
    )
    race_key_to_idx = {rk: i for i, rk in enumerate(race_keys)}

    # ── Pre-compute per-race summaries ───────────────────────────
    # team pace rank per race (lower rank = faster team)
    team_avg_pace: dict[tuple[int, int], dict[str, float]] = {}
    # driver grid & approximate finish for positions-gained
    driver_grid: dict[tuple[int, int], dict[str, int]] = {}
    driver_finish: dict[tuple[int, int], dict[str, int]] = {}
    # driver average lap time per race
    driver_avg_lap: dict[tuple[int, int], dict[str, float]] = {}
    # driver-team map per race
    driver_team_map: dict[tuple[int, int], dict[str, str]] = {}

    for rk in race_keys:
        sub = df[(df["year"] == rk[0]) & (df["round_number"] == rk[1])]

        # Team average pace from attacker/defender lap times
        team_times: dict[str, list[float]] = defaultdict(list)
        drv_times: dict[str, list[float]] = defaultdict(list)
        drv_team: dict[str, str] = {}

        for _, row in sub.iterrows():
            for role, team_col, lt_col in [
                ("attacker", "attacker_team", "attacker_lap_time"),
                ("defender", "defender_team", "defender_lap_time"),
            ]:
                t = str(row[team_col])
                lt = float(row[lt_col])
                drv = str(row[role])
                if lt > 0:
                    team_times[t].append(lt)
                    drv_times[drv].append(lt)
                drv_team[drv] = t

        team_means = {t: np.mean(v) for t, v in team_times.items() if v}
        sorted_teams = sorted(team_means.items(), key=lambda x: x[1])
        team_rank = {t: rank + 1 for rank, (t, _) in enumerate(sorted_teams)}
        team_avg_pace[rk] = team_rank

        driver_team_map[rk] = drv_team
        driver_avg_lap[rk] = {d: float(np.mean(v)) for d, v in drv_times.items() if v}

        # Grid (from qualification rank) and approximate finish position
        grids: dict[str, int] = {}
        finishes: dict[str, int] = {}
        for role, q_col, p_col in [
            ("attacker", "attacker_qualification_rank", "attacker_position"),
            ("defender", "defender_qualification_rank", "defender_position"),
        ]:
            for _, row in sub.drop_duplicates(subset=[role]).iterrows():
                drv = str(row[role])
                grids.setdefault(drv, int(row[q_col]))
                finishes[drv] = int(row[p_col])  # last lap position as proxy
        driver_grid[rk] = grids
        driver_finish[rk] = finishes

    # ── Build per-row features ───────────────────────────────────
    att_team_rank = []
    def_team_rank = []
    team_deltas = []
    att_pg = []
    def_pg = []
    att_qvt = []
    def_qvt = []
    att_rpvt = []
    def_rpvt = []

    for _, row in df.iterrows():
        rk = _race_key(row)
        ri = race_key_to_idx[rk]
        season_rks = [
            race_keys[j] for j in range(max(0, ri - 5), ri)
            if race_keys[j][0] == rk[0]
        ]
        all_prev_rks = race_keys[max(0, ri - 5): ri]

        # ── team pace rank (season-to-date) ─────────────────
        att_t = str(row["attacker_team"])
        def_t = str(row["defender_team"])
        ranks_att: list[float] = []
        ranks_def: list[float] = []
        for prk in season_rks:
            tr = team_avg_pace.get(prk, {})
            n_teams = max(len(tr), 1)
            ranks_att.append(tr.get(att_t, n_teams))
            ranks_def.append(tr.get(def_t, n_teams))
        a_rank = float(np.mean(ranks_att)) if ranks_att else DEFAULT_RANK
        d_rank = float(np.mean(ranks_def)) if ranks_def else DEFAULT_RANK
        att_team_rank.append(a_rank)
        def_team_rank.append(d_rank)
        team_deltas.append(a_rank - d_rank)

        # ── positions gained (grid − finish) rolling ────────
        att_drv = str(row["attacker"])
        def_drv = str(row["defender"])
        att_gains: list[float] = []
        def_gains: list[float] = []
        for prk in all_prev_rks:
            g = driver_grid.get(prk, {})
            f = driver_finish.get(prk, {})
            if att_drv in g and att_drv in f:
                att_gains.append(g[att_drv] - f[att_drv])
            if def_drv in g and def_drv in f:
                def_gains.append(g[def_drv] - f[def_drv])
        att_pg.append(float(np.mean(att_gains)) if att_gains else DEFAULT_RATE)
        def_pg.append(float(np.mean(def_gains)) if def_gains else DEFAULT_RATE)

        # ── quali vs teammate & race pace vs teammate ───────
        att_qvt_vals: list[float] = []
        def_qvt_vals: list[float] = []
        att_rpvt_vals: list[float] = []
        def_rpvt_vals: list[float] = []
        for prk in all_prev_rks:
            dtm = driver_team_map.get(prk, {})
            grids = driver_grid.get(prk, {})
            avg_laps = driver_avg_lap.get(prk, {})

            for drv, qvt_list, rpvt_list in [
                (att_drv, att_qvt_vals, att_rpvt_vals),
                (def_drv, def_qvt_vals, def_rpvt_vals),
            ]:
                tm = dtm.get(drv)
                if tm is None:
                    continue
                teammates = [d for d, t in dtm.items() if t == tm and d != drv]
                if not teammates:
                    continue
                tm_mate = teammates[0]
                if drv in grids and tm_mate in grids:
                    qvt_list.append(grids[tm_mate] - grids[drv])
                if drv in avg_laps and tm_mate in avg_laps:
                    rpvt_list.append(avg_laps[tm_mate] - avg_laps[drv])

        att_qvt.append(float(np.mean(att_qvt_vals)) if att_qvt_vals else DEFAULT_RATE)
        def_qvt.append(float(np.mean(def_qvt_vals)) if def_qvt_vals else DEFAULT_RATE)
        att_rpvt.append(float(np.mean(att_rpvt_vals)) if att_rpvt_vals else DEFAULT_RATE)
        def_rpvt.append(float(np.mean(def_rpvt_vals)) if def_rpvt_vals else DEFAULT_RATE)

    out = df.copy()
    out["attacker_team_pace_rank"] = att_team_rank
    out["defender_team_pace_rank"] = def_team_rank
    out["team_delta"] = team_deltas
    out["attacker_positions_gained_avg"] = att_pg
    out["defender_positions_gained_avg"] = def_pg
    out["attacker_quali_vs_teammate"] = att_qvt
    out["defender_quali_vs_teammate"] = def_qvt
    out["attacker_race_pace_vs_teammate"] = att_rpvt
    out["defender_race_pace_vs_teammate"] = def_rpvt
    return out
