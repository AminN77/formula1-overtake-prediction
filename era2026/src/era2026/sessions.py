"""
Cached session access and per-lap race state for the 2026 era.

Everything here reads from the local FastF1 cache. Warm it with
``era2026-warm`` on a networked machine; nothing downstream needs the network.

The interesting piece is :func:`overtake_mode_by_lap`. Overtake Mode replaced
DRS in 2026 and, unlike DRS, it has no telemetry channel. It surfaces only as
race control messages, and those messages alone are not enough to reconstruct
the state. See that function for why.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

# FastF1 track status codes. A lap's TrackStatus is the concatenation of every
# code seen during that lap, e.g. "126" means green, yellow and VSC all occurred.
STATUS_GREEN = "1"
STATUS_YELLOW = "2"
STATUS_SAFETY_CAR = "4"
STATUS_RED = "5"
STATUS_VSC = "6"
STATUS_VSC_ENDING = "7"

#: Codes that mean the track was neutralised for part of the lap.
NEUTRALISING_CODES = frozenset({STATUS_SAFETY_CAR, STATUS_RED, STATUS_VSC})

#: Exact race control messages that toggle Overtake Mode. Matched exactly,
#: never by substring: "LAPPED CARS MAY NOW OVERTAKE THE SAFETY CAR" also
#: contains the word OVERTAKE and has nothing to do with the mechanism.
OVERTAKE_ENABLED = "OVERTAKE ENABLED"
OVERTAKE_DISABLED = "OVERTAKE DISABLED"


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


CACHE_DIR = _project_root() / ".fastf1_cache"


def _enable_cache() -> None:
    import fastf1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))


@lru_cache(maxsize=64)
def load_race(year: int, round_number: int):
    """Load one race from cache. Telemetry is never needed: the speed traps the
    feature set uses (SpeedI1/I2/FL/ST) are lap-level columns."""
    import fastf1

    _enable_cache()
    session = fastf1.get_session(year, round_number, "R")
    session.load(telemetry=False, weather=True, messages=True)
    return session


def completed_rounds(year: int) -> pd.DataFrame:
    """Rounds of ``year`` whose event date has passed, in calendar order."""
    import fastf1

    _enable_cache()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    schedule = schedule[schedule["RoundNumber"] > 0].copy()
    now = pd.Timestamp.utcnow().tz_localize(None)
    schedule = schedule[pd.to_datetime(schedule["EventDate"]) < now]
    return schedule[["RoundNumber", "EventName", "EventDate", "EventFormat"]].reset_index(drop=True)


def neutralised_by_lap(session) -> pd.Series:
    """True for each lap where a safety car, VSC or red flag was out.

    Indexed by lap number. A lap counts as neutralised if any neutralising code
    appears in its TrackStatus, since the status string aggregates the whole lap.
    """
    laps = session.laps
    if laps is None or laps.empty or "LapNumber" not in laps.columns:
        return pd.Series(dtype=bool, name="neutralised")
    status = laps.groupby("LapNumber")["TrackStatus"].agg(
        lambda values: "".join(str(v) for v in values.dropna())
    )
    return status.apply(lambda s: any(code in s for code in NEUTRALISING_CODES))


def overtake_mode_messages(session) -> pd.DataFrame:
    """The Overtake Mode toggles, exactly matched, with their lap numbers."""
    messages = session.race_control_messages
    if messages is None or not len(messages):
        return pd.DataFrame(columns=["Lap", "Message"])
    text = messages["Message"].astype(str).str.strip().str.upper()
    toggles = messages[text.isin({OVERTAKE_ENABLED, OVERTAKE_DISABLED})].copy()
    toggles["Message"] = text[toggles.index]
    return toggles[["Lap", "Message"]].dropna(subset=["Lap"])


def overtake_mode_by_lap(session) -> pd.Series:
    """Per-lap Overtake Mode availability, as a boolean Series indexed by lap.

    Reconstructing this needs two sources, because neither is sufficient alone.

    Race control emits an explicit ``OVERTAKE DISABLED`` only at the standing
    start. Later neutralisations disable the mode silently, and the re-enable is
    announced *inconsistently*: Canada 2026 takes a VSC on laps 31, 46 and 53 and
    logs ``OVERTAKE ENABLED`` after each one, while Australia 2026 takes VSCs on
    laps 12, 18 and 34 and logs no re-enable at all. Replaying messages alone
    therefore leaves Australia switched off for the final 46 laps, which is wrong.

    So track status is the primary signal and messages are explicit overrides:

    * an explicit ``OVERTAKE ENABLED`` switches it on, and wins for that lap even
      if the lap was neutralised, since racing resumed partway through it
      (Canada lap 46 carries VSC DEPLOYED, VSC ENDING and OVERTAKE ENABLED),
    * an explicit ``OVERTAKE DISABLED`` switches it off and *persists* until the
      next explicit enable (Monaco is off from lap 60 to 70, Silverstone from
      lap 48 to the flag),
    * otherwise a neutralised lap is off for that lap only, and the baseline
      state resumes afterwards.
    """
    laps = session.laps
    if laps is None or laps.empty or "LapNumber" not in laps.columns:
        return pd.Series(dtype=bool, name="overtake_mode_enabled")
    lap_numbers = pd.Index(sorted(laps["LapNumber"].dropna().astype(int).unique()), name="LapNumber")
    if not len(lap_numbers):
        return pd.Series(dtype=bool, name="overtake_mode_enabled")

    toggles = overtake_mode_messages(session)
    enabled_on = set(toggles.loc[toggles["Message"] == OVERTAKE_ENABLED, "Lap"].astype(int))
    disabled_on = set(toggles.loc[toggles["Message"] == OVERTAKE_DISABLED, "Lap"].astype(int))
    neutralised = neutralised_by_lap(session).reindex(lap_numbers, fill_value=False)

    values: list[bool] = []
    baseline = False  # the grid forms up with the mode off
    for lap in lap_numbers:
        if lap in enabled_on:
            baseline = True
            values.append(True)  # an announced re-enable wins over the lap's status
            continue
        if lap in disabled_on:
            baseline = False
        values.append(baseline and not bool(neutralised[lap]))

    return pd.Series(values, index=lap_numbers, name="overtake_mode_enabled")


@dataclass(frozen=True)
class RaceState:
    """Per-lap race context for one round."""

    year: int
    round_number: int
    event_name: str
    total_laps: int
    overtake_mode: pd.Series
    neutralised: pd.Series

    @property
    def overtake_mode_share(self) -> float:
        return float(self.overtake_mode.mean()) if len(self.overtake_mode) else float("nan")


def race_state(year: int, round_number: int) -> RaceState:
    session = load_race(year, round_number)
    laps = session.laps
    return RaceState(
        year=year,
        round_number=round_number,
        event_name=str(session.event["EventName"]),
        total_laps=int(laps["LapNumber"].max()),
        overtake_mode=overtake_mode_by_lap(session),
        neutralised=neutralised_by_lap(session),
    )
