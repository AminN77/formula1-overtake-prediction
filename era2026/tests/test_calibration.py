"""Calibration and bootstrap intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from era2026.calibration import (
    BootstrapEnsemble,
    Calibrated,
    expected_calibration_error,
    reliability,
)


class Overconfident:
    """Pushes probabilities toward the extremes, so calibration has work to do."""

    name = "overconfident"

    def __init__(self):
        self.rate_ = 0.1

    def fit(self, X, y):
        self.rate_ = float(np.mean(y))
        return self

    def predict_proba(self, X):
        base = np.clip(X["signal"].to_numpy(), 0, 1)
        return np.clip(base ** 0.35, 0.001, 0.999)


def synthetic(n_rounds: int = 8, per_round: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for r in range(1, n_rounds + 1):
        for _ in range(per_round):
            signal = float(rng.uniform(0, 1))
            records.append({
                "round_number": r,
                "signal": signal,
                "label": int(rng.random() < signal * 0.3),
            })
    return pd.DataFrame(records)


def test_perfect_predictions_have_no_calibration_error():
    y = np.array([0, 0, 1, 1] * 25)
    p = y.astype(float) * 0.999 + 0.0005
    assert expected_calibration_error(y, p) < 0.01


def test_calibration_error_detects_systematic_bias():
    y = np.zeros(400, dtype=int)
    y[:40] = 1
    p = np.full(400, 0.5)  # claims 50% where the truth is 10%
    assert expected_calibration_error(y, p) > 0.3


def test_reliability_bins_sum_to_the_sample():
    frame = synthetic(n_rounds=2)
    table = reliability(frame["label"].to_numpy(), frame["signal"].to_numpy(), bins=5)
    assert table["n"].sum() == len(frame)
    assert set(table.columns) >= {"predicted", "observed", "gap", "n"}


def test_calibration_improves_an_overconfident_model():
    data = synthetic()
    train = data[data["round_number"] <= 6]
    test = data[data["round_number"] > 6]

    raw = Overconfident().fit(train, train["label"])
    calibrated = Calibrated(name="c", factory=Overconfident, method="isotonic")
    calibrated.fit(train, train["label"])

    before = expected_calibration_error(test["label"].to_numpy(), raw.predict_proba(test))
    after = expected_calibration_error(test["label"].to_numpy(), calibrated.predict_proba(test))
    assert after < before


def test_calibration_falls_back_when_there_is_one_group():
    data = synthetic(n_rounds=1)
    model = Calibrated(name="c", factory=Overconfident).fit(data, data["label"])
    p = model.predict_proba(data)
    assert len(p) == len(data)
    assert np.all((p >= 0) & (p <= 1))


def test_bootstrap_interval_brackets_the_point_estimate():
    data = synthetic()
    ensemble = BootstrapEnsemble(name="b", factory=Overconfident, n_bootstraps=10)
    ensemble.fit(data, data["label"])
    interval = ensemble.predict_interval(data)
    assert (interval["lower"] <= interval["hazard"] + 1e-9).all()
    assert (interval["hazard"] <= interval["upper"] + 1e-9).all()
    assert (interval["width"] >= 0).all()


def test_bootstrap_resamples_whole_races_not_rows():
    """Rows inside a race are correlated, so row-level resampling would
    understate the spread. Every member must see a whole number of races."""
    data = synthetic(n_rounds=6, per_round=50)
    seen = []

    class Recorder(Overconfident):
        def fit(self, X, y):
            seen.append(sorted(X["round_number"].unique()))
            return super().fit(X, y)

    BootstrapEnsemble(name="b", factory=Recorder, n_bootstraps=5).fit(data, data["label"])
    assert seen
    for rounds in seen:
        assert set(rounds).issubset(set(range(1, 7)))


def test_bootstrap_members_differ():
    data = synthetic()
    ensemble = BootstrapEnsemble(name="b", factory=Overconfident, n_bootstraps=12)
    ensemble.fit(data, data["label"])
    assert len(ensemble.members_) > 1
    assert ensemble.predict_interval(data)["width"].max() >= 0
