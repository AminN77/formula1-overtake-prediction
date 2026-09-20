"""Backtest protocol: the guarantee that round k never trains on round k."""

from __future__ import annotations

import numpy as np
import pandas as pd

from era2026.backtest import run_backtest


class Spy:
    """Records the rounds it was trained on, so leakage is detectable."""

    def __init__(self):
        self.name = "spy"
        self.seen: list[set[int]] = []

    def fit(self, X, y):
        self.seen.append(set(X["round_number"].unique()))
        return self

    def predict_proba(self, X):
        return np.full(len(X), 0.5)


def synthetic(n_rounds: int = 8, per_round: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for r in range(1, n_rounds + 1):
        for i in range(per_round):
            rows.append({
                "round_number": r,
                "episode_id": f"{r}-{i // 4}",
                "lap_number": i,
                "gap_ahead": float(rng.uniform(0, 3)),
                "label": int(rng.random() < 0.15),
                "censored": False,
            })
    return pd.DataFrame(rows)


def test_training_never_includes_the_test_round():
    spy = Spy()
    rows = synthetic()
    result = run_backtest(rows, [spy], start_round=5)
    tested = sorted(result.per_round["round_number"])
    assert tested == [5, 6, 7, 8]
    for test_round, seen in zip(tested, spy.seen):
        assert test_round not in seen, f"round {test_round} leaked into its own training set"
        assert seen == set(range(1, test_round)), "training set must be every earlier round"


def test_training_set_grows_each_round():
    spy = Spy()
    result = run_backtest(synthetic(), [spy], start_round=5)
    sizes = result.per_round.sort_values("round_number")["train_rows"].tolist()
    assert sizes == sorted(sizes)
    assert len(set(sizes)) == len(sizes), "the training pool must expand every round"


def test_censored_rows_never_reach_the_model():
    rows = synthetic()
    rows.loc[rows.index[:50], "censored"] = True
    spy = Spy()
    result = run_backtest(rows, [spy], start_round=5)
    assert result.per_round["n"].sum() < len(rows)


def test_every_round_from_start_is_tested_once():
    spy = Spy()
    result = run_backtest(synthetic(n_rounds=10), [spy], start_round=5)
    counts = result.per_round["round_number"].value_counts()
    assert set(counts.index) == {5, 6, 7, 8, 9, 10}
    assert (counts == 1).all()
