"""
Hyperparameter search, run once on an early block and then frozen.

The evaluation protocol forbids tuning against a backtest round, because a
config chosen by looking at round k makes round k no longer out of sample. So
the search sees only the tuning block (rounds 1 to 4 by default) and the result
is written to a JSON file that is committed and never regenerated casually.

Four races with race-grouped folds is a small and noisy basis for a search, and
this does not pretend otherwise. The point is not to find the optimum. It is
that whatever config ships was chosen without looking at the rounds it is
reported on. A noisy honest search beats a clean dishonest one.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

DEFAULT_TUNE_ROUNDS = (1, 2, 3, 4)
DEFAULT_TRIALS = 40


def _project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


FROZEN_PARAMS_PATH = _project_root() / "frozen_params.json"

#: Deliberately narrow, and skewed small. At a few thousand rows with a 7%
#: positive rate, the useful axis is regularisation, not capacity.
SEARCH_SPACE = {
    "n_estimators": [150, 250, 300, 400, 600],
    "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
    "num_leaves": [7, 11, 15, 23, 31],
    "max_depth": [3, 4, 5, 6],
    "min_child_samples": [20, 30, 40, 60, 80],
    "subsample": [0.7, 0.8, 0.85, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.85],
    "reg_alpha": [0.0, 0.1, 0.5, 1.0],
    "reg_lambda": [0.5, 1.0, 3.0, 10.0],
}


def sample_params(rng: np.random.Generator) -> dict:
    return {key: rng.choice(values).item() for key, values in SEARCH_SPACE.items()}


@dataclass
class TuningResult:
    best_params: dict
    trials: pd.DataFrame
    tune_rounds: tuple[int, ...]
    n_rows: int
    n_events: int


def tune(
    rows: pd.DataFrame,
    columns: list[str],
    tune_rounds: tuple[int, ...] = DEFAULT_TUNE_ROUNDS,
    n_trials: int = DEFAULT_TRIALS,
    seed: int = 42,
    monotone: dict[str, int] | None = None,
) -> TuningResult:
    import lightgbm as lgb

    block = rows[rows["round_number"].isin(tune_rounds)]
    block = block[~block["censored"]] if "censored" in block.columns else block
    if block.empty or block["label"].nunique() < 2:
        raise ValueError("tuning block has no usable rows")

    X = block[columns].astype(float).fillna(0.0)
    y = block["label"].to_numpy()
    groups = block["round_number"].to_numpy()
    n_splits = min(len(np.unique(groups)), 4)
    constraints = [(monotone or {}).get(c, 0) for c in columns]

    rng = np.random.default_rng(seed)
    records = []
    for trial in range(n_trials):
        params = sample_params(rng)
        scores = []
        for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(X, y, groups):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            model = lgb.LGBMClassifier(
                **params, subsample_freq=1, monotone_constraints=constraints,
                random_state=seed, verbosity=-1,
            )
            model.fit(X.iloc[train_idx], y[train_idx])
            p = model.predict_proba(X.iloc[test_idx])[:, 1]
            scores.append(average_precision_score(y[test_idx], p))
        if scores:
            records.append({"trial": trial, "mean_pr_auc": float(np.mean(scores)),
                            "sd_pr_auc": float(np.std(scores)), "folds": len(scores), **params})

    trials = pd.DataFrame(records).sort_values("mean_pr_auc", ascending=False)
    if trials.empty:
        raise ValueError("no trial completed")
    best = trials.iloc[0]
    best_params = {key: best[key] for key in SEARCH_SPACE}
    for key in ("n_estimators", "num_leaves", "max_depth", "min_child_samples"):
        best_params[key] = int(best_params[key])
    return TuningResult(best_params, trials, tuple(tune_rounds), len(block), int(y.sum()))


def freeze(result: TuningResult, path: Path = FROZEN_PARAMS_PATH) -> Path:
    payload = {
        "params": result.best_params,
        "tuned_on_rounds": list(result.tune_rounds),
        "tuning_rows": result.n_rows,
        "tuning_events": result.n_events,
        "cv_pr_auc": float(result.trials.iloc[0]["mean_pr_auc"]),
        "trials": int(len(result.trials)),
        "note": (
            "Chosen on the tuning block only and frozen. Never re-tune against a "
            "backtest round: a config picked by looking at round k makes round k "
            "no longer out of sample."
        ),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_frozen(path: Path = FROZEN_PARAMS_PATH) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["params"]


def cli() -> None:
    import warnings

    warnings.filterwarnings("ignore")
    from era2026.dataset import load_or_build
    from era2026.features import FEATURE_TIERS
    from era2026.models import MONOTONE

    rows = load_or_build(2026)
    columns = list(FEATURE_TIERS)
    result = tune(rows, columns, monotone=MONOTONE)
    path = freeze(result)

    print(f"tuned on rounds {result.tune_rounds}: {result.n_rows} rows, {result.n_events} events")
    print(f"\ntop 5 of {len(result.trials)} trials\n")
    print(result.trials.head(5)[["mean_pr_auc", "sd_pr_auc", "n_estimators", "learning_rate",
                                 "num_leaves", "max_depth", "min_child_samples"]]
          .round(4).to_string(index=False))
    print(f"\nfrozen to {path}")
