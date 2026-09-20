"""The frozen hyperparameter protocol."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from era2026.models import HAND_PICKED, GradientBoosting
from era2026.tuning import (
    DEFAULT_TUNE_ROUNDS,
    SEARCH_SPACE,
    freeze,
    load_frozen,
    sample_params,
    tune,
)


def synthetic(n_rounds: int = 8, per_round: int = 150, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    for r in range(1, n_rounds + 1):
        for _ in range(per_round):
            gap = float(rng.uniform(0, 3))
            records.append({
                "round_number": r,
                "gap_ahead": gap,
                "closing_rate": float(rng.normal()),
                "label": int(rng.random() < max(0.02, 0.35 - 0.1 * gap)),
                "censored": False,
            })
    return pd.DataFrame(records)


def test_sampled_params_stay_inside_the_space():
    rng = np.random.default_rng(0)
    for _ in range(20):
        params = sample_params(rng)
        assert set(params) == set(SEARCH_SPACE)
        for key, value in params.items():
            assert value in SEARCH_SPACE[key]


def test_tuning_only_reads_the_tuning_block():
    """The guarantee: a config chosen by looking at round k makes round k no
    longer out of sample, so the search must never see later rounds."""
    data = synthetic()
    result = tune(data, ["gap_ahead", "closing_rate"], tune_rounds=(1, 2, 3, 4), n_trials=3)
    assert result.tune_rounds == (1, 2, 3, 4)
    assert result.n_rows == len(data[data["round_number"].isin([1, 2, 3, 4])])


def test_tuning_ignores_censored_rows():
    data = synthetic()
    data.loc[data.index[:100], "censored"] = True
    result = tune(data, ["gap_ahead"], tune_rounds=(1, 2, 3, 4), n_trials=2)
    block = data[data["round_number"].isin([1, 2, 3, 4])]
    assert result.n_rows == int((~block["censored"]).sum())


def test_freeze_round_trips(tmp_path):
    data = synthetic()
    result = tune(data, ["gap_ahead"], tune_rounds=(1, 2, 3, 4), n_trials=3)
    path = freeze(result, tmp_path / "frozen.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["tuned_on_rounds"] == [1, 2, 3, 4]
    assert load_frozen(path) == result.best_params


def test_missing_frozen_file_falls_back_to_the_prior(tmp_path):
    assert load_frozen(tmp_path / "absent.json") is None


def test_explicit_params_override_the_frozen_set():
    model = GradientBoosting(name="g", columns=["gap_ahead"], params=HAND_PICKED)
    assert model.resolved_params() == HAND_PICKED


def test_default_tune_rounds_precede_the_backtest_start():
    from era2026.backtest import DEFAULT_START_ROUND

    assert max(DEFAULT_TUNE_ROUNDS) < DEFAULT_START_ROUND


def test_empty_tuning_block_is_rejected():
    data = synthetic(n_rounds=2)
    with pytest.raises(ValueError):
        tune(data, ["gap_ahead"], tune_rounds=(90, 91), n_trials=2)
