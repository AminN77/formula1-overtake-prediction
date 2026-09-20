"""Registry, drift, and the promotion gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from era2026.drift import (
    MIN_RACES_FOR_RACE_LEVEL_DRIFT,
    MONOTONE_BY_CONSTRUCTION,
    drift_report,
    population_stability_index,
    race_constant_features,
)
from era2026.pipeline import GATE_WINDOW, PROMOTION_MARGIN


def frame(n_rounds: int, per_round: int = 100, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for r in range(1, n_rounds + 1):
        air = 20.0 + r  # constant within a race, varies between races
        for _ in range(per_round):
            records.append({
                "round_number": r,
                "gap_ahead": float(rng.uniform(0, 3) + shift),
                "air_temp": air,
                "attacker_prior_rounds": float(r - 1),
                "label": int(rng.random() < 0.07),
            })
    return pd.DataFrame(records)


def test_psi_is_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(size=2000)
    assert population_stability_index(x, x) < 0.01


def test_psi_grows_with_a_real_shift():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=3000), rng.normal(loc=2.0, size=3000)
    assert population_stability_index(a, b) > 0.25


def test_race_constant_features_are_detected():
    data = frame(8)
    constant = race_constant_features(data, ["gap_ahead", "air_temp", "attacker_prior_rounds"])
    assert "air_temp" in constant
    assert "gap_ahead" not in constant


def test_counters_are_never_reported_as_drift():
    """A career round count rises every race by construction. Reporting it is a
    permanent false positive, which is how a monitor gets ignored."""
    data = frame(12)
    report = drift_report(data[data["round_number"] <= 6], data[data["round_number"] > 6],
                          ["gap_ahead", "attacker_prior_rounds"])
    counter = report[report["feature"] == "attacker_prior_rounds"].iloc[0]
    assert counter["severity"] == "by_construction"
    assert counter["feature"] in MONOTONE_BY_CONSTRUCTION


def test_race_level_drift_stays_silent_without_enough_races():
    """A distribution shift cannot be measured from a handful of observations.
    The monitor must say so rather than emit a large meaningless number."""
    data = frame(6)
    report = drift_report(data[data["round_number"] <= 4], data[data["round_number"] > 4],
                          ["air_temp"])
    assert report.iloc[0]["severity"] == "insufficient_data"
    assert np.isnan(report.iloc[0]["psi"])


def test_race_level_drift_reports_once_there_are_enough_races():
    data = frame(MIN_RACES_FOR_RACE_LEVEL_DRIFT * 3)
    half = MIN_RACES_FOR_RACE_LEVEL_DRIFT * 3 // 2
    report = drift_report(data[data["round_number"] <= half], data[data["round_number"] > half],
                          ["air_temp"])
    assert report.iloc[0]["severity"] != "insufficient_data"


def test_row_level_drift_is_measurable_immediately():
    """Features varying within a race have thousands of observations, so they
    are testable from the first round."""
    stable = frame(6)
    shifted = frame(6, shift=2.0, seed=1)
    report = drift_report(stable, shifted, ["gap_ahead"])
    assert report.iloc[0]["scope"] == "per_row"
    assert report.iloc[0]["psi"] > 0.25


def test_gate_window_is_not_a_single_race():
    """Per-round PR-AUC has a standard deviation of 0.22, so one race cannot
    separate a better model from a luckier one."""
    assert GATE_WINDOW >= 3


def test_promotion_requires_a_margin():
    """A zero margin promotes on noise roughly every other week."""
    assert PROMOTION_MARGIN > 0


def test_registry_round_trip(tmp_path, monkeypatch):
    from era2026 import registry
    from era2026.registry import ModelVersion, incumbent, promote, register

    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(registry, "MANIFEST", tmp_path / "manifest.json")

    assert incumbent() is None
    first = ModelVersion(version="r01", created_at="now", trained_on_rounds=[1],
                         train_rows=10, train_events=1, data_fingerprint="abc",
                         params={}, features=[], metrics={})
    register(first)
    promote("r01")
    assert incumbent()["version"] == "r01"

    second = ModelVersion(version="r02", created_at="now", trained_on_rounds=[1, 2],
                          train_rows=20, train_events=2, data_fingerprint="def",
                          params={}, features=[], metrics={})
    register(second)
    promote("r02", over="r01")
    assert incumbent()["version"] == "r02"

    archived = [v for v in registry.history().to_dict("records") if v["version"] == "r01"]
    assert archived[0]["status"] == "archived"
