"""Battle episodes and hazard row outcomes."""

from __future__ import annotations

import pandas as pd
import pytest

from era2026.episodes import (
    OUTCOME_CENSORED_PIT,
    OUTCOME_CENSORED_RACE_END,
    OUTCOME_CENSORED_RETIREMENT,
    OUTCOME_EVENT,
    OUTCOME_SURVIVED,
    assign_episodes,
    candidate_pairs,
    hazard_rows_from_session,
    trainable,
)


class FakeSession:
    """`order` gives drivers P1 downward; `gaps` gives each one's cumulative
    seconds behind the leader on that lap."""

    def __init__(self, laps: list[dict], messages: list[tuple[int, str]] | None = None):
        defaults = {"PitInTime": None, "PitOutTime": None, "IsAccurate": True, "TrackStatus": "1"}
        self.laps = pd.DataFrame([{**defaults, **row} for row in laps])
        self.race_control_messages = pd.DataFrame(
            [{"Lap": lap, "Message": text} for lap, text in (messages or [])],
            columns=["Lap", "Message"],
        )
        self.event = {"EventName": "Test Grand Prix"}
        self.weather_data = pd.DataFrame(
            columns=["Time", "AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"]
        )


#: Lap columns the feature builder reads, with plausible defaults. Real sessions
#: carry all of these, so a fixture missing them would test the wrong thing.
LAP_DEFAULTS = {
    "LapTime": pd.Timedelta("90s"),
    "SpeedI1": 250.0,
    "SpeedI2": 260.0,
    "SpeedFL": 300.0,
    "SpeedST": 320.0,
    "TyreLife": 10.0,
    "Stint": 1.0,
    "FreshTyre": False,
    "Compound": "MEDIUM",
}


def lap(n: int, order: list[str], gaps: list[float], **overrides) -> list[dict]:
    return [
        {
            "LapNumber": n,
            "Driver": driver,
            "Position": i + 1,
            "Time": pd.Timedelta(seconds=60.0 * n + gaps[i]),
            **LAP_DEFAULTS,
            **overrides.get(driver, {}),
        }
        for i, driver in enumerate(order)
    ]


def test_candidate_is_adjacent_and_attacker_is_behind():
    session = FakeSession(lap(1, ["VER", "HAM"], [0.0, 0.8]))
    pairs = candidate_pairs(session)
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["attacker"] == "HAM" and row["defender"] == "VER"
    assert row["gap_ahead"] == pytest.approx(0.8)


def test_gap_beyond_threshold_is_not_a_candidate():
    session = FakeSession(lap(1, ["VER", "HAM"], [0.0, 5.0]))
    assert candidate_pairs(session, gap_threshold=3.0).empty


def test_only_adjacent_pairs_count():
    """P1 and P3 are within the gap but not adjacent, so they are not a battle."""
    session = FakeSession(lap(1, ["VER", "HAM", "LEC"], [0.0, 0.5, 1.0]))
    pairs = candidate_pairs(session)
    assert len(pairs) == 2
    assert set(zip(pairs["attacker"], pairs["defender"])) == {("HAM", "VER"), ("LEC", "HAM")}


def test_episodes_split_on_a_lap_gap():
    """The same pair fighting on laps 1-2 and again on 8-9 is two battles."""
    laps = (
        lap(1, ["VER", "HAM"], [0.0, 0.5])
        + lap(2, ["VER", "HAM"], [0.0, 0.5])
        + lap(8, ["VER", "HAM"], [0.0, 0.5])
        + lap(9, ["VER", "HAM"], [0.0, 0.5])
    )
    episodes = assign_episodes(candidate_pairs(FakeSession(laps)))
    assert episodes["episode_id"].nunique() == 2
    assert sorted(episodes["battle_lap"]) == [1, 1, 2, 2]


def test_pass_is_an_event():
    laps = (
        lap(1, ["VER", "HAM"], [0.0, 0.5])
        + lap(2, ["HAM", "VER"], [0.0, 0.5])
        + lap(3, ["HAM", "VER"], [0.0, 0.5])
    )
    rows = hazard_rows_from_session(FakeSession(laps), 2026, 1)
    first = rows[(rows["lap_number"] == 1) & (rows["attacker"] == "HAM")].iloc[0]
    assert first["outcome"] == OUTCOME_EVENT
    assert first["label"] == 1


def test_still_racing_without_a_pass_survives():
    laps = [r for n in (1, 2, 3) for r in lap(n, ["VER", "HAM"], [0.0, 0.5])]
    rows = hazard_rows_from_session(FakeSession(laps), 2026, 1)
    assert rows[rows["lap_number"] == 1].iloc[0]["outcome"] == OUTCOME_SURVIVED


def test_attacker_dropping_out_of_range_survives_rather_than_censors():
    """Falling back is a genuine failure to pass, not a missing observation.

    Censoring it would bias the hazard upward by discarding the clearest
    negatives in the dataset.
    """
    laps = lap(1, ["VER", "HAM"], [0.0, 0.5]) + lap(2, ["VER", "HAM"], [0.0, 9.0])
    rows = hazard_rows_from_session(FakeSession(laps), 2026, 1)
    row = rows[rows["lap_number"] == 1].iloc[0]
    assert row["outcome"] == OUTCOME_SURVIVED
    assert not row["episode_still_open"], "the pair separated"
    assert row["label"] == 0


def test_pit_stop_next_lap_censors():
    laps = (
        lap(1, ["VER", "HAM"], [0.0, 0.5])
        + lap(2, ["VER", "HAM"], [0.0, 0.5], VER={"PitInTime": pd.Timedelta("1s")})
    )
    rows = hazard_rows_from_session(FakeSession(laps), 2026, 1)
    row = rows[rows["lap_number"] == 1].iloc[0]
    assert row["outcome"] == OUTCOME_CENSORED_PIT
    assert row["censored"] and row["label"] == -1


def test_retirement_censors():
    laps = lap(1, ["VER", "HAM", "LEC"], [0.0, 0.5, 1.0]) + lap(2, ["VER", "HAM"], [0.0, 0.5])
    rows = hazard_rows_from_session(FakeSession(laps), 2026, 1)
    row = rows[(rows["lap_number"] == 1) & (rows["attacker"] == "LEC")].iloc[0]
    assert row["outcome"] == OUTCOME_CENSORED_RETIREMENT


def test_final_lap_censors():
    laps = [r for n in (1, 2) for r in lap(n, ["VER", "HAM"], [0.0, 0.5])]
    rows = hazard_rows_from_session(FakeSession(laps), 2026, 1)
    assert rows[rows["lap_number"] == 2].iloc[0]["outcome"] == OUTCOME_CENSORED_RACE_END


def test_censoring_keeps_the_earlier_laps():
    """The point of censoring: a pit stop on lap 4 must not delete laps 1-3."""
    laps = (
        [r for n in (1, 2, 3) for r in lap(n, ["VER", "HAM"], [0.0, 0.5])]
        + lap(4, ["VER", "HAM"], [0.0, 0.5], VER={"PitInTime": pd.Timedelta("1s")})
        + lap(5, ["VER", "HAM"], [0.0, 0.5])
    )
    rows = hazard_rows_from_session(FakeSession(laps), 2026, 1)
    train = trainable(rows)
    assert len(rows) == 5
    assert set(train["lap_number"]) == {1, 2, 4}, "lap 3 censors (pit on 4), lap 5 is the final lap"
    assert (train["label"] == 0).all()


def test_empty_session():
    assert hazard_rows_from_session(FakeSession([]), 2026, 1).empty
