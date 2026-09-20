"""Order-flip extraction."""

from __future__ import annotations

import pandas as pd

from era2026.events import extract_order_flips, filter_events


class FakeSession:
    def __init__(self, rows: list[dict], messages: list[tuple[int, str]] | None = None):
        defaults = {"PitInTime": None, "PitOutTime": None, "IsAccurate": True, "TrackStatus": "1"}
        self.laps = pd.DataFrame([{**defaults, **row} for row in rows])
        self.race_control_messages = pd.DataFrame(
            [{"Lap": lap, "Message": text} for lap, text in (messages or [])],
            columns=["Lap", "Message"],
        )
        self.event = {"EventName": "Test Grand Prix"}


def lap(n: int, order: list[str], **overrides) -> list[dict]:
    """One lap where `order` lists drivers from P1 downward."""
    return [
        {"LapNumber": n, "Driver": d, "Position": i + 1, **overrides.get(d, {})}
        for i, d in enumerate(order)
    ]


def test_single_clean_pass():
    session = FakeSession(lap(1, ["VER", "HAM"]) + lap(2, ["HAM", "VER"]))
    flips = extract_order_flips(session, 2026, 1)
    assert len(flips) == 1
    row = flips.iloc[0]
    assert row["attacker"] == "HAM"
    assert row["defender"] == "VER"
    assert row["lap_number"] == 1, "stamped with the lap observed, not the lap completed"
    assert row["position_gain"] == 1
    assert row["adjacent_before"] and row["adjacent_after"]
    assert not row["pit_related"]


def test_no_flip_when_order_holds():
    session = FakeSession(lap(1, ["VER", "HAM"]) + lap(2, ["VER", "HAM"]))
    assert extract_order_flips(session, 2026, 1).empty


def test_pit_stop_flip_is_flagged_and_filtered():
    """The defender pits, so the position change is not an on-track pass."""
    session = FakeSession(
        lap(1, ["VER", "HAM"]) + lap(2, ["HAM", "VER"], VER={"PitInTime": pd.Timedelta("1s")})
    )
    flips = extract_order_flips(session, 2026, 1)
    assert len(flips) == 1
    assert flips.iloc[0]["pit_related"]
    assert filter_events(flips).empty


def test_pit_on_either_lap_counts():
    """A stop on lap L+1 reorders the pair just as one on lap L does."""
    rows = lap(1, ["VER", "HAM"], VER={"PitOutTime": pd.Timedelta("1s")}) + lap(2, ["HAM", "VER"])
    assert extract_order_flips(FakeSession(rows), 2026, 1).iloc[0]["pit_related"]


def test_multi_position_gain_yields_one_flip_per_pair():
    """Gaining three places produces three flips, one against each car passed."""
    session = FakeSession(
        lap(1, ["VER", "HAM", "LEC", "NOR"]) + lap(2, ["NOR", "VER", "HAM", "LEC"])
    )
    flips = extract_order_flips(session, 2026, 1)
    assert len(flips) == 3
    assert set(flips["defender"]) == {"VER", "HAM", "LEC"}
    assert (flips["attacker"] == "NOR").all()
    assert (flips["position_gain"] == 3).all()


def test_net_zero_gain_is_still_a_real_flip():
    """Passing one car while being passed by another nets zero, but both are real.

    Guards against filtering on position_gain, which would drop these.
    """
    session = FakeSession(
        lap(1, ["VER", "HAM", "LEC"]) + lap(2, ["HAM", "VER", "LEC"]) + lap(3, ["HAM", "LEC", "VER"])
    )
    flips = extract_order_flips(session, 2026, 1)
    second = flips[flips["lap_number"] == 2]
    assert len(second) == 1
    assert second.iloc[0]["attacker"] == "LEC" and second.iloc[0]["defender"] == "VER"


def test_retirement_is_not_compared():
    """A driver absent from the later lap cannot be flipped past."""
    session = FakeSession(lap(1, ["VER", "HAM", "LEC"]) + lap(2, ["VER", "HAM"]))
    assert extract_order_flips(session, 2026, 1).empty


def test_gap_in_lap_sequence_is_skipped():
    session = FakeSession(lap(1, ["VER", "HAM"]) + lap(5, ["HAM", "VER"]))
    assert extract_order_flips(session, 2026, 1).empty


def test_neutralisation_and_overtake_mode_are_recorded():
    rows = lap(1, ["VER", "HAM"]) + lap(2, ["HAM", "VER"], VER={"TrackStatus": "46"})
    session = FakeSession(rows, messages=[(1, "OVERTAKE ENABLED")])
    row = extract_order_flips(session, 2026, 1).iloc[0]
    assert row["neutralised"], "lap 2 was under VSC"
    assert row["overtake_mode_enabled"], "lap 1 itself was green with the mode on"


def test_filter_keeps_everything_but_pit_stops():
    """Lap one, neutralisations and inaccurate timing all contain real passes."""
    rows = (
        lap(1, ["VER", "HAM"])
        + lap(2, ["HAM", "VER"], VER={"TrackStatus": "4", "IsAccurate": False})
    )
    flips = extract_order_flips(FakeSession(rows), 2026, 1)
    assert len(filter_events(flips)) == 1


def test_empty_session():
    session = FakeSession([])
    assert extract_order_flips(session, 2026, 1).empty
