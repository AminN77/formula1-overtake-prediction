"""The baseline ladder. Anything more complex reports its gain over this."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from era2026.features import FEATURE_TIERS

#: Physics we already know, encoded as monotone constraints so the model cannot
#: learn an absurd direction from a small sample. +1 raises the hazard, -1 lowers
#: it, 0 leaves it free.
MONOTONE: dict[str, int] = {
    "gap_ahead": -1,          # further back is harder
    "gap_mean_3": -1,
    "gap_min_3": -1,
    "closing_rate": +1,       # catching up helps
    "is_closing": +1,
    "closing_laps": +1,
    "pace_delta": -1,         # attacker slower per lap is worse
    "pace_delta_mean_3": -1,
    "speed_st_delta": +1,     # faster on the straight helps
    "speed_fl_delta": +1,
    "tyre_age_difference": -1,  # older tyres than the defender is worse
    "attacker_on_newer_stint": +1,
    "compound_advantage": +1,
    "queue_ahead": -1,        # stuck in a train is harder
}

#: The five features the simple baseline is allowed.
FIVE = ["gap_ahead", "closing_rate", "pace_delta", "tyre_age_difference", "attacker_on_newer_stint"]


@dataclass
class BaseRate:
    """Predict the training positive rate for every row."""

    name: str = "base_rate"
    rate_: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseRate":
        self.rate_ = float(np.mean(y)) if len(y) else 0.0
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self.rate_)


@dataclass
class LogisticBaseline:
    """Logistic regression on a named subset of features."""

    name: str
    columns: list[str]
    C: float = 1.0
    model_: Pipeline | None = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticBaseline":
        self.model_ = Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=self.C, max_iter=2000, class_weight=None)),
        ])
        self.model_.fit(X[self.columns], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.columns])[:, 1]


@dataclass
class GradientBoosting:
    """LightGBM with monotone constraints, shallow and regularised for a small n.

    Hyperparameters are fixed, not tuned. The evaluation protocol requires them
    to be chosen on an early block and then frozen for the whole backtest, so
    tuning happens once, separately, and never against a backtest round.
    """

    name: str = "lightgbm"
    columns: list[str] = field(default_factory=lambda: list(FEATURE_TIERS))
    monotone: bool = True
    model_: object | None = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoosting":
        import lightgbm as lgb

        constraints = [MONOTONE.get(c, 0) for c in self.columns] if self.monotone else None
        self.model_ = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=40,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            monotone_constraints=constraints,
            random_state=42,
            verbosity=-1,
        )
        self.model_.fit(X[self.columns], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.columns])[:, 1]


def ladder() -> list[object]:
    """The ladder, in increasing complexity. Order matters for reporting."""
    return [
        BaseRate(),
        LogisticBaseline(name="gap_only", columns=["gap_ahead"]),
        LogisticBaseline(name="logistic_5", columns=FIVE),
        GradientBoosting(name="lightgbm_mono", monotone=True),
        GradientBoosting(name="lightgbm_free", monotone=False),
    ]
