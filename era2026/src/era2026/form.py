"""
Driver and team form, computed from strictly prior rounds.

Everything here answers "what did we know about this driver before this race
started". A round-k row may only use rounds 1..k-1, which makes these features
valid for every fold of the expanding-origin backtest without recomputation.

All of this is Tier 1. Rank-based and teammate-relative measures are the
strongest transferable class available, because a rank is ordinal within its own
session and a teammate comparison cancels the car, which is precisely the thing
the regulation change altered.

Team identity is used to *compute* a pace rank and is never itself a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from era2026.features import TIER_1

#: Added to FEATURE_TIERS by :func:`register`.
FORM_FEATURES: dict[str, str] = {
    "attacker_prior_rounds": TIER_1,
    "defender_prior_rounds": TIER_1,
    "attacker_overtake_rate": TIER_1,
    "defender_hold_rate": TIER_1,
    "attacker_positions_gained_avg": TIER_1,
    "defender_positions_gained_avg": TIER_1,
    "attacker_team_pace_rank": TIER_1,
    "defender_team_pace_rank": TIER_1,
    "team_pace_rank_delta": TIER_1,
    "attacker_pace_vs_teammate": TIER_1,
    "defender_pace_vs_teammate": TIER_1,
}

#: What a driver with no history gets. Neutral rather than optimistic: an unseen
#: driver should look average, and `prior_rounds` tells the model how much to
#: trust the estimate.
NEUTRAL_RATE = 0.0
NEUTRAL_TEAM_RANK = 6.0  # mid-field of eleven teams


def driver_round_summary(rows: pd.DataFrame, sessions: dict[int, object]) -> pd.DataFrame:
    """One row per (round, driver): team, pace, start and finish, form counters."""
    records: list[dict] = []
    for round_number, session in sessions.items():
        laps = session.laps
        if laps is None or laps.empty:
            continue
        first_lap, last_lap = int(laps["LapNumber"].min()), int(laps["LapNumber"].max())
        lap_seconds = laps.assign(
            _s=pd.to_timedelta(laps["LapTime"]).dt.total_seconds()
        )
        by_driver = lap_seconds.groupby("Driver")

        start = (laps[laps["LapNumber"] == first_lap]
                 .set_index("Driver")["Position"].to_dict())
        finish = (laps.sort_values("LapNumber").groupby("Driver")["Position"].last().to_dict())
        team = laps.groupby("Driver")["Team"].first().to_dict()
        median_pace = by_driver["_s"].median().to_dict()

        # Censored rows carry label = -1, so summing over them would corrupt
        # every rate. Form is counted only over labelled intervals.
        labelled = rows[~rows["censored"]] if "censored" in rows.columns else rows
        race_rows = labelled[labelled["round_number"] == round_number]
        attacked = race_rows.groupby("attacker")["label"].agg(["size", "sum"]) \
            if len(race_rows) else pd.DataFrame(columns=["size", "sum"])
        defended = race_rows.groupby("defender")["label"].agg(["size", "sum"]) \
            if len(race_rows) else pd.DataFrame(columns=["size", "sum"])

        for driver in team:
            a = attacked.loc[driver] if driver in attacked.index else None
            d = defended.loc[driver] if driver in defended.index else None
            records.append({
                "round_number": round_number,
                "driver": driver,
                "team": team[driver],
                "median_pace": median_pace.get(driver, np.nan),
                "start_position": start.get(driver, np.nan),
                "finish_position": finish.get(driver, np.nan),
                "attack_rows": int(a["size"]) if a is not None else 0,
                "attack_passes": int(a["sum"]) if a is not None else 0,
                "defend_rows": int(d["size"]) if d is not None else 0,
                "defend_passed": int(d["sum"]) if d is not None else 0,
            })
    return pd.DataFrame(records)


def _team_pace_ranks(summary: pd.DataFrame) -> pd.DataFrame:
    """Per round, rank teams by median pace across their drivers. 1 is fastest."""
    frames = []
    for round_number, group in summary.groupby("round_number"):
        team_pace = group.groupby("team")["median_pace"].median().sort_values()
        ranks = pd.Series(range(1, len(team_pace) + 1), index=team_pace.index, dtype=float)
        frames.append(pd.DataFrame({
            "round_number": round_number, "team": ranks.index, "team_pace_rank": ranks.to_numpy()
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["round_number", "team", "team_pace_rank"]
    )


def _pace_vs_teammate(summary: pd.DataFrame) -> pd.Series:
    """Teammate median pace minus this driver's. Positive means faster than
    the teammate, and the car itself cancels out."""
    out = pd.Series(np.nan, index=summary.index)
    for _, group in summary.groupby(["round_number", "team"]):
        if len(group) != 2:
            continue
        a, b = group.index
        pace_a, pace_b = summary.loc[a, "median_pace"], summary.loc[b, "median_pace"]
        if pd.notna(pace_a) and pd.notna(pace_b):
            out.loc[a] = pace_b - pace_a
            out.loc[b] = pace_a - pace_b
    return out


def prior_form(summary: pd.DataFrame) -> pd.DataFrame:
    """For each (round, driver), form aggregated over strictly earlier rounds."""
    if summary.empty:
        return pd.DataFrame()

    summary = summary.copy()
    summary = summary.merge(_team_pace_ranks(summary), on=["round_number", "team"], how="left")
    summary["pace_vs_teammate"] = _pace_vs_teammate(summary).to_numpy()
    summary["positions_gained"] = summary["start_position"] - summary["finish_position"]

    rounds = sorted(summary["round_number"].unique())
    records: list[dict] = []
    for round_number in rounds:
        history = summary[summary["round_number"] < round_number]
        for driver in summary.loc[summary["round_number"] == round_number, "driver"]:
            past = history[history["driver"] == driver]
            attack_rows = past["attack_rows"].sum()
            defend_rows = past["defend_rows"].sum()
            records.append({
                "round_number": round_number,
                "driver": driver,
                "prior_rounds": int(len(past)),
                "overtake_rate": float(past["attack_passes"].sum() / attack_rows)
                if attack_rows else NEUTRAL_RATE,
                "hold_rate": float(1 - past["defend_passed"].sum() / defend_rows)
                if defend_rows else NEUTRAL_RATE,
                "positions_gained_avg": float(past["positions_gained"].mean())
                if len(past) else NEUTRAL_RATE,
                "team_pace_rank": float(past["team_pace_rank"].mean())
                if past["team_pace_rank"].notna().any() else NEUTRAL_TEAM_RANK,
                "pace_vs_teammate": float(past["pace_vs_teammate"].mean())
                if past["pace_vs_teammate"].notna().any() else NEUTRAL_RATE,
            })
    return pd.DataFrame(records)


def attach_form(rows: pd.DataFrame, sessions: dict[int, object]) -> pd.DataFrame:
    """Add every form feature to a frame of hazard rows."""
    if rows is None or rows.empty:
        return rows

    form = prior_form(driver_round_summary(rows, sessions))
    if form.empty:
        for name in FORM_FEATURES:
            rows[name] = NEUTRAL_RATE
        return rows

    indexed = form.set_index(["round_number", "driver"])
    out = rows.copy()

    def lookup(column: str, who: str) -> pd.Series:
        keys = list(zip(out["round_number"], out[who]))
        return pd.Series([indexed[column].get(key, np.nan) for key in keys], index=out.index)

    out["attacker_prior_rounds"] = lookup("prior_rounds", "attacker").fillna(0)
    out["defender_prior_rounds"] = lookup("prior_rounds", "defender").fillna(0)
    out["attacker_overtake_rate"] = lookup("overtake_rate", "attacker").fillna(NEUTRAL_RATE)
    out["defender_hold_rate"] = lookup("hold_rate", "defender").fillna(NEUTRAL_RATE)
    out["attacker_positions_gained_avg"] = lookup("positions_gained_avg", "attacker").fillna(NEUTRAL_RATE)
    out["defender_positions_gained_avg"] = lookup("positions_gained_avg", "defender").fillna(NEUTRAL_RATE)
    out["attacker_team_pace_rank"] = lookup("team_pace_rank", "attacker").fillna(NEUTRAL_TEAM_RANK)
    out["defender_team_pace_rank"] = lookup("team_pace_rank", "defender").fillna(NEUTRAL_TEAM_RANK)
    out["team_pace_rank_delta"] = out["attacker_team_pace_rank"] - out["defender_team_pace_rank"]
    out["attacker_pace_vs_teammate"] = lookup("pace_vs_teammate", "attacker").fillna(NEUTRAL_RATE)
    out["defender_pace_vs_teammate"] = lookup("pace_vs_teammate", "defender").fillna(NEUTRAL_RATE)
    return out


def register() -> None:
    """Add the form features to the global tier registry."""
    from era2026.features import FEATURE_TIERS

    FEATURE_TIERS.update(FORM_FEATURES)
