"""Overtake Mode reconstruction.

The rule is subtle because the 2026 race control feed is inconsistent: some
races announce every re-enable after a neutralisation, others announce none.
These cases are drawn from real 2026 rounds.
"""

from __future__ import annotations

import pandas as pd
import pytest

from era2026.sessions import overtake_mode_by_lap


class FakeSession:
    """Minimal stand-in: overtake_mode_by_lap only reads laps and messages."""

    def __init__(self, track_status: dict[int, str], messages: list[tuple[int, str]]):
        self.laps = pd.DataFrame(
            [{"LapNumber": lap, "TrackStatus": status} for lap, status in track_status.items()],
            columns=["LapNumber", "TrackStatus"],
        )
        self.race_control_messages = pd.DataFrame(
            [{"Lap": lap, "Message": text} for lap, text in messages],
            columns=["Lap", "Message"],
        )


def green(*laps: int) -> dict[int, str]:
    return {lap: "1" for lap in laps}


def test_grid_forms_up_disabled():
    """No enable message means the mode never comes on."""
    session = FakeSession(green(1, 2, 3), [])
    assert not overtake_mode_by_lap(session).any()


def test_standing_start_disable_then_enable():
    """Both toggles land on lap 1; the lap ends enabled."""
    session = FakeSession(green(1, 2, 3), [(1, "OVERTAKE DISABLED"), (1, "OVERTAKE ENABLED")])
    assert list(overtake_mode_by_lap(session)) == [True, True, True]


def test_neutralisation_without_re_enable_resumes():
    """Australia 2026: VSCs on laps 12, 18 and 34 with no re-enable messages.

    Replaying messages alone would leave the mode off for the final 46 laps.
    Track status has to drive it, so the mode resumes once the VSC clears.
    """
    status = {1: "1", 2: "1", 3: "16", 4: "671", 5: "1", 6: "1"}
    session = FakeSession(status, [(1, "OVERTAKE DISABLED"), (1, "OVERTAKE ENABLED")])
    assert list(overtake_mode_by_lap(session)) == [True, True, False, False, True, True]


def test_announced_re_enable_wins_over_neutralised_lap():
    """Canada 2026 lap 46 carries VSC DEPLOYED, VSC ENDING and OVERTAKE ENABLED.

    Racing resumed partway through that lap, so it counts as enabled.
    """
    status = {1: "1", 2: "6", 3: "671", 4: "1"}
    messages = [(1, "OVERTAKE DISABLED"), (1, "OVERTAKE ENABLED"), (3, "OVERTAKE ENABLED")]
    session = FakeSession(status, messages)
    assert list(overtake_mode_by_lap(session)) == [True, False, True, True]


def test_explicit_disable_persists_until_explicit_enable():
    """Monaco 2026: disabled on lap 60, re-enabled on 71, green track throughout.

    A neutralisation is transient, but an announced disable is not.
    """
    session = FakeSession(
        green(1, 2, 3, 4, 5),
        [(1, "OVERTAKE ENABLED"), (2, "OVERTAKE DISABLED"), (5, "OVERTAKE ENABLED")],
    )
    assert list(overtake_mode_by_lap(session)) == [True, False, False, False, True]


def test_explicit_disable_to_the_flag():
    """Silverstone 2026: disabled on lap 48 and never re-enabled."""
    session = FakeSession(green(1, 2, 3), [(1, "OVERTAKE ENABLED"), (2, "OVERTAKE DISABLED")])
    assert list(overtake_mode_by_lap(session)) == [True, False, False]


def test_lapped_cars_message_is_not_a_toggle():
    """'LAPPED CARS MAY NOW OVERTAKE THE SAFETY CAR' contains the word OVERTAKE
    but is a safety car instruction. Substring matching would misread it."""
    session = FakeSession(
        green(1, 2, 3),
        [
            (1, "OVERTAKE ENABLED"),
            (2, "OVERTAKE DISABLED"),
            (3, "LAPPED CARS MAY NOW OVERTAKE THE SAFETY CAR: 6, 63, 41"),
        ],
    )
    assert list(overtake_mode_by_lap(session)) == [True, False, False]


def test_yellow_flag_does_not_disable():
    """Only safety car, VSC and red flag neutralise. A local yellow does not."""
    session = FakeSession({1: "1", 2: "12", 3: "26"}, [(1, "OVERTAKE ENABLED")])
    assert list(overtake_mode_by_lap(session)) == [True, True, False]


def test_empty_session():
    session = FakeSession({}, [])
    assert overtake_mode_by_lap(session).empty


@pytest.mark.parametrize("missing", [None, pd.DataFrame(columns=["Lap", "Message"])])
def test_missing_race_control(missing):
    session = FakeSession(green(1, 2), [])
    session.race_control_messages = missing
    assert not overtake_mode_by_lap(session).any()
