"""
Battle episodes and discrete-time hazard rows.

The unit of prediction is one lap of one battle, not one battle. For every lap
that an adjacent pair is still fighting, the model estimates

    h(t) = P(pass on lap t+1 | pair still battling at lap t)

and any horizon follows by composition, ``P(pass within k) = 1 - prod(1 - h)``.

The outcome of a row is one of three things, and the third is the point of the
whole design:

``event``
    A filtered on-track pass for this exact pair between lap L and L+1.
``survived``
    No pass. Includes the case where the attacker drops back out of range,
    which is a genuine failure to pass rather than a missing observation.
``censored``
    The opportunity was removed rather than taken: either car pitted, either car
    retired, or the race ended. These rows carry no label and are excluded from
    training, but every earlier lap of the episode is kept. That is what
    "censoring rather than deleting" buys, and it is where the legacy pipeline
    threw information away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from era2026.events import filter_events, extract_order_flips
from era2026.sessions import load_race, neutralised_by_lap, overtake_mode_by_lap

#: Gap in seconds within which an adjacent pair counts as a battle. Derived from
#: 2026 data rather than inherited: see ``scripts`` output in the commit message.
DEFAULT_GAP_THRESHOLD = 3.0

OUTCOME_EVENT = "event"
OUTCOME_SURVIVED = "survived"
OUTCOME_CENSORED_PIT = "censored_pit"
OUTCOME_CENSORED_RETIREMENT = "censored_retirement"
OUTCOME_CENSORED_RACE_END = "censored_race_end"

CENSORED_OUTCOMES = frozenset({
    OUTCOME_CENSORED_PIT,
    OUTCOME_CENSORED_RETIREMENT,
    OUTCOME_CENSORED_RACE_END,
})


def _grid(laps: pd.DataFrame, column: str, aggfunc="first") -> pd.DataFrame:
    return laps.pivot_table(index="LapNumber", columns="Driver", values=column, aggfunc=aggfunc)


def lap_times_seconds(laps: pd.DataFrame) -> pd.DataFrame:
    """Session time at which each driver completed each lap, in seconds.

    The difference between two drivers' values on the same lap is their on-track
    gap, which is how the gap is obtained without telemetry.
    """
    grid = _grid(laps, "Time")
    return grid.apply(lambda col: pd.to_timedelta(col).dt.total_seconds())


def candidate_pairs(session, gap_threshold: float = DEFAULT_GAP_THRESHOLD) -> pd.DataFrame:
    """Adjacent attacker-defender pairs within ``gap_threshold`` at each lap.

    The attacker is the car behind. Both cars must have completed the lap, which
    excludes retirements automatically, and the gap threshold excludes lapped
    traffic without needing an explicit same-lap test.
    """
    laps = session.laps
    if laps is None or laps.empty:
        return pd.DataFrame()

    positions = _grid(laps, "Position")
    times = lap_times_seconds(laps)
    pit = laps.assign(_pit=laps["PitInTime"].notna() | laps["PitOutTime"].notna())
    pits = _grid(pit, "_pit", aggfunc="max").astype(float).fillna(0.0).astype(bool)
    pits = pits.reindex(index=positions.index, columns=positions.columns, fill_value=False)

    rows: list[dict] = []
    for lap in positions.index:
        pos = positions.loc[lap]
        tim = times.loc[lap]
        present = pos.notna() & tim.notna()
        if present.sum() < 2:
            continue
        order = pos[present].sort_values()
        drivers = list(order.index)
        for defender, attacker in zip(drivers, drivers[1:]):
            gap = float(tim[attacker] - tim[defender])
            if not (0.0 <= gap <= gap_threshold):
                continue
            rows.append({
                "lap_number": int(lap),
                "attacker": attacker,
                "defender": defender,
                "attacker_position": int(pos[attacker]),
                "defender_position": int(pos[defender]),
                "gap_ahead": gap,
                "attacker_pitted": bool(pits.loc[lap, attacker]),
                "defender_pitted": bool(pits.loc[lap, defender]),
            })
    return pd.DataFrame(rows)


def assign_episodes(candidates: pd.DataFrame) -> pd.DataFrame:
    """Group a pair's consecutive battling laps into episodes.

    A gap in the lap sequence starts a new episode: the same two cars fighting on
    laps 5-8 and again on laps 30-33 are two separate battles, not one.
    """
    if candidates.empty:
        return candidates.assign(episode_id=pd.Series(dtype=str), battle_lap=pd.Series(dtype=int))

    out = candidates.sort_values(["attacker", "defender", "lap_number"], kind="mergesort").copy()
    pair = out["attacker"] + ">" + out["defender"]
    lap_step = out.groupby(pair, sort=False)["lap_number"].diff()
    new_episode = (lap_step != 1).fillna(True)
    sequence = new_episode.groupby(pair, sort=False).cumsum()
    out["episode_id"] = pair + "#" + sequence.astype(int).astype(str)
    out["battle_lap"] = out.groupby("episode_id", sort=False).cumcount() + 1
    return out.sort_values(["lap_number", "attacker_position"], kind="mergesort").reset_index(drop=True)


def build_hazard_rows(
    year: int,
    round_number: int,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> pd.DataFrame:
    """One row per lap of every battle episode in a race, with its outcome."""
    return hazard_rows_from_session(load_race(year, round_number), year, round_number, gap_threshold)


def hazard_rows_from_session(
    session,
    year: int,
    round_number: int,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> pd.DataFrame:
    """The body of :func:`build_hazard_rows`, separated so it can be tested
    against a synthetic session rather than the 364MB cache."""
    laps = session.laps
    if laps is None or laps.empty:
        return pd.DataFrame()

    candidates = assign_episodes(candidate_pairs(session, gap_threshold))
    if candidates.empty:
        return pd.DataFrame()

    flips = filter_events(extract_order_flips(session, year, round_number))
    passes = set(zip(flips["lap_number"], flips["attacker"], flips["defender"])) if len(flips) else set()

    positions = _grid(laps, "Position")
    pit = laps.assign(_pit=laps["PitInTime"].notna() | laps["PitOutTime"].notna())
    pits = _grid(pit, "_pit", aggfunc="max").astype(float).fillna(0.0).astype(bool)
    pits = pits.reindex(index=positions.index, columns=positions.columns, fill_value=False)

    final_lap = int(positions.index.max())
    neutralised = neutralised_by_lap(session).reindex(positions.index, fill_value=False)
    overtake_mode = overtake_mode_by_lap(session).reindex(positions.index, fill_value=False)

    # laps on which each episode is active, to tell "still battling" from "separated"
    active = {
        episode: set(group["lap_number"])
        for episode, group in candidates.groupby("episode_id", sort=False)
    }

    def present(driver: str, lap: int) -> bool:
        return lap in positions.index and pd.notna(positions.loc[lap, driver])

    def pitted(driver: str, lap: int) -> bool:
        return lap in pits.index and bool(pits.loc[lap, driver])

    outcomes: list[str] = []
    for row in candidates.itertuples():
        lap, nxt = row.lap_number, row.lap_number + 1
        attacker, defender = row.attacker, row.defender

        if (lap, attacker, defender) in passes:
            outcomes.append(OUTCOME_EVENT)
        elif lap >= final_lap:
            outcomes.append(OUTCOME_CENSORED_RACE_END)
        elif not present(attacker, nxt) or not present(defender, nxt):
            outcomes.append(OUTCOME_CENSORED_RETIREMENT)
        elif pitted(attacker, nxt) or pitted(defender, nxt):
            outcomes.append(OUTCOME_CENSORED_PIT)
        else:
            # Still racing and no pass. Includes the attacker dropping out of
            # range, which is a real failure rather than a missing observation.
            outcomes.append(OUTCOME_SURVIVED)

    result = candidates.copy()
    result["outcome"] = outcomes
    result["censored"] = result["outcome"].isin(CENSORED_OUTCOMES)
    result["label"] = np.where(result["outcome"] == OUTCOME_EVENT, 1, 0)
    result.loc[result["censored"], "label"] = -1  # never train on these
    result["year"] = year
    result["round_number"] = round_number
    result["event_name"] = str(session.event["EventName"])
    result["total_laps"] = final_lap
    result["neutralised"] = result["lap_number"].map(neutralised).fillna(False).astype(bool)
    result["overtake_mode_enabled"] = (
        result["lap_number"].map(overtake_mode).fillna(False).astype(bool)
    )
    result["episode_still_open"] = [
        (r.lap_number + 1) in active[r.episode_id] for r in result.itertuples()
    ]
    return result


def trainable(rows: pd.DataFrame) -> pd.DataFrame:
    """Rows that carry a label. Censored intervals are excluded, earlier laps kept."""
    return rows[~rows["censored"]].reset_index(drop=True) if len(rows) else rows


def season_hazard_rows(
    year: int,
    rounds: list[int],
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hazard rows plus a per-race audit for a whole season."""
    frames, audit = [], []
    for round_number in rounds:
        try:
            rows = build_hazard_rows(year, round_number, gap_threshold)
            status, error = "ok", ""
        except Exception as exc:
            rows = pd.DataFrame()
            status, error = "failed", f"{type(exc).__name__}: {exc}"
        frames.append(rows)
        train = trainable(rows) if len(rows) else rows
        audit.append({
            "year": year,
            "round_number": round_number,
            "event_name": rows["event_name"].iloc[0] if len(rows) else "",
            "status": status,
            "episodes": rows["episode_id"].nunique() if len(rows) else 0,
            "rows": len(rows),
            "trainable": len(train),
            "events": int((rows["outcome"] == OUTCOME_EVENT).sum()) if len(rows) else 0,
            "censored": int(rows["censored"].sum()) if len(rows) else 0,
            "error": error,
        })
    frames = [f for f in frames if len(f)]
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return combined, pd.DataFrame(audit)
