"""
Expanding-origin backtest: train on rounds 1..k-1, predict round k, advance.

Every round becomes an out-of-sample test exactly once, so 14 rounds yield 10
test results rather than one. The backtest walks the same path the production
loop walks, which is what keeps the reported number and the deployed behaviour
the same measurement.

No round k information ever reaches the model that predicts round k.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

from era2026.episodes import trainable

#: First round used as a test set. Four races is the minimum that trains.
DEFAULT_START_ROUND = 5


def _metrics(y: np.ndarray, p: np.ndarray, base_rate: float) -> dict[str, float]:
    out: dict[str, float] = {
        "n": int(len(y)),
        "events": int(y.sum()),
        "positive_rate": float(y.mean()) if len(y) else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
    }
    # AUCs are undefined on a single-class round
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y, p))
        out["pr_auc"] = float(average_precision_score(y, p))
        # lift over always predicting the base rate
        out["pr_auc_lift"] = out["pr_auc"] / out["positive_rate"] if out["positive_rate"] else np.nan
    else:
        out["roc_auc"] = out["pr_auc"] = out["pr_auc_lift"] = np.nan
    out["log_loss"] = float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6), labels=[0, 1])) if len(y) else np.nan
    return out


@dataclass
class BacktestResult:
    per_round: pd.DataFrame
    summary: pd.DataFrame
    predictions: pd.DataFrame


def run_backtest(
    rows: pd.DataFrame,
    models: list,
    start_round: int = DEFAULT_START_ROUND,
    round_column: str = "round_number",
) -> BacktestResult:
    train_pool = trainable(rows)
    features = None
    rounds = sorted(train_pool[round_column].unique())
    test_rounds = [r for r in rounds if r >= start_round]

    per_round: list[dict] = []
    predictions: list[pd.DataFrame] = []

    for test_round in test_rounds:
        train = train_pool[train_pool[round_column] < test_round]
        test = train_pool[train_pool[round_column] == test_round]
        if train.empty or test.empty or train["label"].nunique() < 2:
            continue

        y_train = train["label"].to_numpy()
        y_test = test["label"].to_numpy()
        base_rate = float(y_train.mean())

        for model in models:
            fitted = model.fit(train, pd.Series(y_train, index=train.index))
            p = np.asarray(fitted.predict_proba(test), dtype=float)
            record = {"round_number": int(test_round), "model": model.name,
                      "train_rows": len(train), **_metrics(y_test, p, base_rate)}
            per_round.append(record)
            predictions.append(pd.DataFrame({
                "round_number": int(test_round),
                "model": model.name,
                "episode_id": test["episode_id"].to_numpy(),
                "lap_number": test["lap_number"].to_numpy(),
                "label": y_test,
                "hazard": p,
            }))

    per_round_df = pd.DataFrame(per_round)
    if per_round_df.empty:
        return BacktestResult(per_round_df, pd.DataFrame(), pd.DataFrame())

    summary = (
        per_round_df.groupby("model", sort=False)
        .agg(rounds=("round_number", "nunique"),
             rows=("n", "sum"),
             events=("events", "sum"),
             pr_auc=("pr_auc", "mean"),
             pr_auc_sd=("pr_auc", "std"),
             roc_auc=("roc_auc", "mean"),
             brier=("brier", "mean"),
             log_loss=("log_loss", "mean"))
        .reset_index()
    )
    return BacktestResult(per_round_df, summary, pd.concat(predictions, ignore_index=True))


def pooled_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Metrics over all backtest rows pooled, which is less noisy than the mean
    of per-round metrics but hides the per-round trajectory."""
    out = []
    for name, group in predictions.groupby("model", sort=False):
        y, p = group["label"].to_numpy(), group["hazard"].to_numpy()
        out.append({"model": name, **_metrics(y, p, float(y.mean()))})
    return pd.DataFrame(out)
