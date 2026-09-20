"""
Overtake event extraction by pairwise order flips.

A flip is the only regulation-independent definition of an overtake available
from timing data: driver A is behind driver B at the end of lap L and ahead of
them at the end of lap L+1. It says nothing about DRS, aero or energy, which is
exactly why it survives the 2026 rule change intact.

Raw flips massively overcount, because a pit stop, a retirement or a
slow-puncture crawl all reorder the field without anyone passing anyone on
track. Each flip therefore carries the metadata needed to filter it, and
:func:`filter_events` applies the one filter that matters.

Convention: a flip between lap L and lap L+1 is stamped ``lap_number = L``, the
lap at which the situation was observed. That matches the hazard framing, where
the row at lap L predicts a pass occurring on lap L+1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from era2026.sessions import NEUTRALISING_CODES, load_race, overtake_mode_by_lap

EVENT_COLUMNS = [
    "year", "round_number", "event_name", "lap_number",
    "attacker", "defender",
    "attacker_position_before", "defender_position_before",
    "attacker_position_after", "defender_position_after",
    "position_gain", "adjacent_before", "adjacent_after",
    "pit_related", "accurate_timing", "neutralised",
    "lap1_or_restart_like", "overtake_mode_enabled",
]


def _pivot(laps: pd.DataFrame, value: str, aggfunc="first") -> pd.DataFrame:
    """Lap number by driver grid of one lap-level column."""
    return laps.pivot_table(index="LapNumber", columns="Driver", values=value, aggfunc=aggfunc)


def _pit_involvement(laps: pd.DataFrame) -> pd.DataFrame:
    """True where a driver entered or left the pits on that lap."""
    pit = laps.copy()
    pit["_pit"] = pit["PitInTime"].notna() | pit["PitOutTime"].notna()
    # Cast through float: filling an object-dtype pivot triggers pandas' deprecated
    # silent downcasting.
    return _pivot(pit, "_pit", aggfunc="max").astype(float).fillna(0.0).astype(bool)


def _neutralised_lap_flags(laps: pd.DataFrame) -> pd.Series:
    status = laps.groupby("LapNumber")["TrackStatus"].agg(
        lambda values: "".join(str(v) for v in values.dropna())
    )
    return status.apply(lambda s: any(code in s for code in NEUTRALISING_CODES))


def extract_order_flips(session, year: int, round_number: int) -> pd.DataFrame:
    """Every pairwise order flip in one race, unfiltered, with filter metadata."""
    laps = session.laps
    if laps is None or laps.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    event_name = str(session.event["EventName"])
    positions = _pivot(laps, "Position")
    pits = _pit_involvement(laps).reindex(index=positions.index, columns=positions.columns)
    pits = pits.astype(float).fillna(0.0).astype(bool)
    accurate = _pivot(laps, "IsAccurate", aggfunc="min")
    accurate = accurate.reindex(index=positions.index, columns=positions.columns)
    accurate = accurate.astype(float).fillna(0.0).astype(bool)
    neutralised = _neutralised_lap_flags(laps).reindex(positions.index, fill_value=False)
    overtake_mode = overtake_mode_by_lap(session).reindex(positions.index, fill_value=False)

    drivers = np.array(positions.columns)
    rows: list[dict] = []

    lap_numbers = list(positions.index)
    for current, following in zip(lap_numbers, lap_numbers[1:]):
        if following != current + 1:
            continue  # a gap in the lap sequence is not a comparable pair

        before = positions.loc[current].to_numpy(dtype=float)
        after = positions.loc[following].to_numpy(dtype=float)
        present = ~np.isnan(before) & ~np.isnan(after)
        if present.sum() < 2:
            continue

        idx = np.flatnonzero(present)
        b, a = before[idx], after[idx]

        # attacker i was behind defender j, then ahead of them
        flipped = (b[:, None] > b[None, :]) & (a[:, None] < a[None, :])
        attackers, defenders = np.nonzero(flipped)
        if not len(attackers):
            continue

        pit_now = pits.loc[current].to_numpy(dtype=bool)[idx]
        pit_next = pits.loc[following].to_numpy(dtype=bool)[idx]
        acc_now = accurate.loc[current].to_numpy()[idx]
        acc_next = accurate.loc[following].to_numpy()[idx]
        names = drivers[idx]

        for i, j in zip(attackers, defenders):
            rows.append({
                "year": year,
                "round_number": round_number,
                "event_name": event_name,
                "lap_number": int(current),
                "attacker": str(names[i]),
                "defender": str(names[j]),
                "attacker_position_before": int(b[i]),
                "defender_position_before": int(b[j]),
                "attacker_position_after": int(a[i]),
                "defender_position_after": int(a[j]),
                "position_gain": int(b[i] - a[i]),
                "adjacent_before": bool(b[i] == b[j] + 1),
                "adjacent_after": bool(a[j] == a[i] + 1),
                "pit_related": bool(pit_now[i] or pit_now[j] or pit_next[i] or pit_next[j]),
                "accurate_timing": bool(acc_now[i] and acc_now[j] and acc_next[i] and acc_next[j]),
                "neutralised": bool(neutralised.loc[current] or neutralised.loc[following]),
                "lap1_or_restart_like": bool(current <= 1),
                "overtake_mode_enabled": bool(overtake_mode.loc[current]),
            })

    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


def filter_events(events: pd.DataFrame, exclude_pit_related: bool = True) -> pd.DataFrame:
    """Keep the flips that plausibly describe an on-track pass.

    Only pit involvement is excluded by default. Everything else stays: lap one,
    neutralisations and inaccurate timing all contain real passes, and filtering
    them costs more recall than it buys precision. This matches the filter the
    legacy audit validated against published season totals.
    """
    if events.empty:
        return events
    keep = pd.Series(True, index=events.index)
    if exclude_pit_related:
        keep &= ~events["pit_related"].astype(bool)
    return events[keep].reset_index(drop=True)


def race_events(year: int, round_number: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw and filtered flips for one race."""
    session = load_race(year, round_number)
    raw = extract_order_flips(session, year, round_number)
    return raw, filter_events(raw)


def season_events(year: int, rounds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Raw flips, filtered flips and a per-race audit for a whole season."""
    raws, filtereds, audit = [], [], []
    for round_number in rounds:
        try:
            raw, kept = race_events(year, round_number)
            status, error = "ok", ""
        except Exception as exc:  # a single unloadable race must not sink the season
            raw = kept = pd.DataFrame(columns=EVENT_COLUMNS)
            status, error = "failed", f"{type(exc).__name__}: {exc}"

        raws.append(raw)
        filtereds.append(kept)
        audit.append({
            "year": year,
            "round_number": round_number,
            "event_name": raw["event_name"].iloc[0] if len(raw) else "",
            "status": status,
            "raw_flips": len(raw),
            "filtered_flips": len(kept),
            "pit_related": int(raw["pit_related"].sum()) if len(raw) else 0,
            "error": error,
        })

    def concat(frames):
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=EVENT_COLUMNS)

    return concat(raws), concat(filtereds), pd.DataFrame(audit)
