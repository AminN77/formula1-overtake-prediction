"""Feature builder: tier declarations and the signed direction of key features."""

from __future__ import annotations

import pandas as pd
import pytest

from era2026.episodes import hazard_rows_from_session
from era2026.features import (
    FEATURE_TIERS,
    KNOWN_COLLINEAR,
    TIER_1,
    TIER_2,
    build_features,
    features_by_tier,
)
from tests.test_episodes import FakeSession, lap


def session_with(laps, messages=None):
    return FakeSession(laps, messages)


def built(laps, messages=None) -> pd.DataFrame:
    session = session_with(laps, messages)
    return build_features(session, hazard_rows_from_session(session, 2026, 1))


def straight_fight(n_laps: int, gaps: list[float], **overrides):
    return [
        r
        for i, n in enumerate(range(1, n_laps + 1))
        for r in lap(n, ["VER", "HAM"], [0.0, gaps[i]], **overrides)
    ]


def test_every_declared_feature_is_produced():
    """build_features raises if a declared feature is missing, so this also
    guards against the registry drifting from the implementation."""
    out = built(straight_fight(3, [0.5, 0.5, 0.5]))
    assert not out.empty
    for name in FEATURE_TIERS:
        assert name in out.columns


def test_no_tier_3_features_leak_in():
    """DRS, absolute speeds, absolute lap times and team identity must be absent."""
    out = built(straight_fight(3, [0.5, 0.5, 0.5]))
    forbidden = ("drs", "attacker_team", "defender_team", "attacker_speed", "defender_speed",
                 "attacker_lap_time", "defender_lap_time")
    leaked = [c for c in out.columns if any(c.startswith(f) or c == f for f in forbidden)]
    assert leaked == [], f"tier 3 features leaked: {leaked}"


def test_tiers_are_partitioned():
    assert set(FEATURE_TIERS.values()) == {TIER_1, TIER_2}
    assert set(features_by_tier(TIER_1)) | set(features_by_tier(TIER_2)) == set(FEATURE_TIERS)
    assert not set(features_by_tier(TIER_1)) & set(features_by_tier(TIER_2))


def test_gap_dynamics_are_signed_correctly():
    """A closing battle has negative gap_delta and positive closing_rate."""
    out = built(straight_fight(4, [3.0, 2.0, 1.0, 0.5]))
    later = out[out["lap_number"] >= 2]
    assert (later["gap_delta_1"] < 0).all()
    assert (later["closing_rate"] > 0).all()
    assert (later["is_closing"] == 1).all()


def test_gap_dynamics_reset_per_episode():
    """Lag features must not bleed across two separate battles of the same pair."""
    laps = (
        straight_fight(2, [0.5, 0.5])
        + [r for n in (8, 9) for r in lap(n, ["VER", "HAM"], [0.0, 2.5])]
    )
    out = built(laps)
    first_lap_of_each = out[out["battle_lap"] == 1]
    assert len(first_lap_of_each) == 2
    assert (first_lap_of_each["gap_delta_1"] == 0.0).all()


def test_battle_lap_counts_within_episode():
    out = built(straight_fight(4, [0.5, 0.5, 0.5, 0.5]))
    assert sorted(out["battle_lap"]) == [1, 2, 3, 4]


def test_tyre_features_use_differences_not_absolutes():
    laps = straight_fight(2, [0.5, 0.5], VER={"TyreLife": 30, "Stint": 1},
                          HAM={"TyreLife": 5, "Stint": 2})
    out = built(laps)
    row = out.iloc[0]
    assert row["tyre_age_difference"] == pytest.approx(5 - 30)
    assert row["attacker_on_newer_stint"] == 1


def test_compound_advantage_favours_the_softer_tyre():
    laps = straight_fight(2, [0.5, 0.5], VER={"Compound": "HARD"}, HAM={"Compound": "SOFT"})
    out = built(laps)
    assert out.iloc[0]["compound_advantage"] > 0
    assert out.iloc[0]["same_compound"] == 0


def test_race_progress_and_phases():
    out = built(straight_fight(4, [0.5] * 4))
    assert out["race_progress"].max() == pytest.approx(1.0)
    assert out[out["lap_number"] == 1].iloc[0]["race_phase_opening"] == 1
    assert out[out["lap_number"] == 4].iloc[0]["race_phase_closing"] == 1


def test_queue_counts_only_consecutive_cars_within_range():
    """A car 5s ahead breaks the queue even if the one beyond it is close."""
    laps = lap(1, ["VER", "HAM", "LEC"], [0.0, 0.4, 0.8])
    out = built(laps)
    trailing = out[out["attacker"] == "LEC"].iloc[0]
    assert trailing["queue_ahead"] >= 1


def test_no_nan_in_declared_features():
    out = built(straight_fight(5, [3.0, 2.0, 1.0, 0.6, 0.4]))
    assert out[list(FEATURE_TIERS)].isna().sum().sum() == 0


def test_known_collinear_pairs_are_declared():
    """The overtake-mode / neutralised near-duplicate is documented, not silent."""
    assert KNOWN_COLLINEAR
    for left, right, correlation in KNOWN_COLLINEAR:
        assert left in FEATURE_TIERS
        assert right in FEATURE_TIERS
        assert abs(correlation) > 0.9
