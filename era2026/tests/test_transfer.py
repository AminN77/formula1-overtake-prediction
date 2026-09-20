"""Layer 1: the frozen pre-2026 scalar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from era2026.features import FEATURE_TIERS
from era2026.transfer import LEGACY_TARGET, TRANSFER_FEATURES, shared_features


def test_shared_features_excludes_era_specific_columns():
    """Nothing that describes the old car may cross the boundary."""
    shared = shared_features(list(FEATURE_TIERS))
    assert shared, "the two eras must share some features"
    forbidden = ("drs", "attacker_team", "defender_team", "attacker_speed_i1",
                 "defender_speed_i1", "attacker_straight_speed", "overtake_mode_enabled")
    assert not [c for c in shared if c in forbidden]


def test_shared_features_are_a_subset_of_ours():
    shared = shared_features(list(FEATURE_TIERS))
    assert set(shared).issubset(set(FEATURE_TIERS))


def test_shared_features_ignores_unknown_columns():
    assert shared_features(["definitely_not_a_legacy_column"]) == []


def test_transfer_contributes_exactly_one_column():
    """The whole point: eight seasons reduced to a single scalar, so the
    comparison with and without transfer is exact."""
    assert list(TRANSFER_FEATURES) == ["old_era_score"]


def test_legacy_target_matches_the_in_era_question():
    """A next-lap pass, not the legacy default of a pass within three laps."""
    assert LEGACY_TARGET == "overtake_next_lap"


def test_old_era_model_never_sees_in_era_rows():
    """OldEraModel.fit takes no argument: it can only read legacy files, so no
    2026 row can reach it and no backtest fold can leak into it."""
    import inspect

    from era2026.transfer import OldEraModel

    signature = inspect.signature(OldEraModel.fit)
    assert list(signature.parameters) == ["self"]
