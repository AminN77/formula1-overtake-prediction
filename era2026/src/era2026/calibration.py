"""
Layers 3 and 4: calibrated probabilities, and an interval around them.

A ranking metric says whether the model orders battles correctly. It says
nothing about whether a displayed "18%" means eighteen percent. Since the UI
shows probabilities to a reader, calibration is a correctness requirement here,
not a polish step.

Two separate things live in this module and they answer different questions.

**Calibration** fixes systematic bias in the probability scale, fitted inside
the training pool with a race-grouped CV so no row calibrates on a model that
saw it.

**The interval** is epistemic: how much would this prediction move if we had
drawn a different fourteen races? It comes from a bootstrap ensemble resampled
at race level, because rows within a race are not independent. It should narrow
as the era accumulates data, which is the project's premise made measurable.

A note on what this is *not*. The design originally called for conformal
prediction. Conformal for binary classification yields label *sets* drawn from
{0}, {1}, {0,1}, so at a 6% base rate almost every row returns {0} or {0,1} and
the output says nothing a reader can use. Bootstrap spread answers the question
a probability display actually raises.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold

DEFAULT_BOOTSTRAPS = 40
DEFAULT_INTERVAL = 0.80


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Observed frequency against predicted probability, in equal-count bins."""
    frame = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    try:
        frame["bin"] = pd.qcut(frame["p"], bins, labels=False, duplicates="drop")
    except ValueError:
        frame["bin"] = 0
    # A constant or near-constant prediction makes qcut emit NaN bins rather
    # than raising. Left alone, the groupby drops every row and the error comes
    # back NaN, which reads as "no calibration problem" for the single most
    # miscalibrated model possible.
    frame["bin"] = frame["bin"].fillna(0)
    grouped = frame.groupby("bin").agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")
    )
    grouped["gap"] = grouped["observed"] - grouped["predicted"]
    return grouped.reset_index()


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Mean absolute gap between predicted and observed, weighted by bin size."""
    table = reliability(y, p, bins)
    if table.empty:
        return float("nan")
    weights = table["n"] / table["n"].sum()
    return float((weights * table["gap"].abs()).sum())


class _SklearnAdapter(ClassifierMixin, BaseEstimator):
    """Presents one of our models to sklearn as an estimator it can clone.

    Inherits the sklearn base classes rather than duck-typing: since 1.6
    sklearn dispatches on ``__sklearn_tags__``, and a plain object is rejected
    before ``predict_proba`` is ever reached.
    """

    def __init__(self, factory):
        self._factory = factory
        self._fitted = None
        self.classes_ = np.array([0, 1])

    def get_params(self, deep: bool = True) -> dict:
        return {"factory": self._factory}

    def set_params(self, **params):
        if "factory" in params:
            self._factory = params["factory"]
        return self

    def fit(self, X, y):
        self._fitted = self._factory().fit(X, pd.Series(y, index=X.index))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(self._fitted.predict_proba(X), dtype=float).reshape(-1)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


@dataclass
class Calibrated:
    """Wraps a model with a probability calibrator fitted on grouped folds."""

    name: str
    factory: object
    method: str = "isotonic"
    folds: int = 4
    group_column: str = "round_number"
    model_: object | None = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "Calibrated":
        groups = X[self.group_column].to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            self.model_ = self.factory().fit(X, y)
            self._calibrated = False
            return self

        folds = min(self.folds, n_groups)
        self.model_ = CalibratedClassifierCV(
            estimator=_SklearnAdapter(self.factory),
            method=self.method,
            cv=GroupKFold(n_splits=folds).split(X, y, groups=groups),
        )
        self.model_.fit(X, y)
        self._calibrated = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.model_.predict_proba(X)
        return p[:, 1] if getattr(self, "_calibrated", False) else np.asarray(p, dtype=float)


@dataclass
class BootstrapEnsemble:
    """Race-level bootstrap. Rows inside a race are correlated, so resampling
    rows would understate the spread badly; whole races are resampled instead."""

    name: str
    factory: object
    n_bootstraps: int = DEFAULT_BOOTSTRAPS
    interval: float = DEFAULT_INTERVAL
    group_column: str = "round_number"
    seed: int = 42
    members_: list = field(default_factory=list, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BootstrapEnsemble":
        rng = np.random.default_rng(self.seed)
        groups = X[self.group_column].to_numpy()
        unique = np.unique(groups)
        self.members_ = []
        for _ in range(self.n_bootstraps):
            drawn = rng.choice(unique, size=len(unique), replace=True)
            mask = np.concatenate([np.flatnonzero(groups == g) for g in drawn])
            Xb, yb = X.iloc[mask], y.iloc[mask]
            if yb.nunique() < 2:
                continue
            self.members_.append(self.factory().fit(Xb, yb))
        if not self.members_:
            self.members_ = [self.factory().fit(X, y)]
        return self

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.asarray(m.predict_proba(X), dtype=float).reshape(-1)
                                for m in self.members_])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._matrix(X).mean(axis=1)

    def predict_interval(self, X: pd.DataFrame) -> pd.DataFrame:
        matrix = self._matrix(X)
        tail = (1 - self.interval) / 2
        lower = np.quantile(matrix, tail, axis=1)
        upper = np.quantile(matrix, 1 - tail, axis=1)
        return pd.DataFrame({
            "hazard": matrix.mean(axis=1),
            "lower": lower,
            "upper": upper,
            "width": upper - lower,
        }, index=X.index)
