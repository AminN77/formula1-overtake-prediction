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

#: The original hand-picked configuration, kept as the comparison point the
#: frozen search has to justify itself against.
HAND_PICKED = {
    "n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "max_depth": 4,
    "min_child_samples": 40, "subsample": 0.85, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
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

    Hyperparameters come from ``frozen_params.json`` when it exists, chosen once
    on the tuning block and frozen. The protocol requires this: a config picked
    by looking at round k makes round k no longer out of sample. Passing
    ``params`` explicitly overrides the frozen set, which is how the hand-picked
    baseline stays comparable.
    """

    name: str = "lightgbm"
    columns: list[str] = field(default_factory=lambda: list(FEATURE_TIERS))
    monotone: bool = True
    params: dict | None = None
    model_: object | None = field(default=None, repr=False)

    def resolved_params(self) -> dict:
        if self.params is not None:
            return dict(self.params)
        from era2026.tuning import load_frozen

        return load_frozen() or dict(HAND_PICKED)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoosting":
        import lightgbm as lgb

        constraints = [MONOTONE.get(c, 0) for c in self.columns] if self.monotone else None
        self.model_ = lgb.LGBMClassifier(
            **self.resolved_params(),
            subsample_freq=1,
            monotone_constraints=constraints,
            random_state=42,
            verbosity=-1,
        )
        self.model_.fit(X[self.columns], y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.columns])[:, 1]




@dataclass
class RandomForest:
    """A bagged ensemble, for contrast with boosting. Same trees, opposite
    variance-reduction mechanism: boosting fits residuals sequentially, bagging
    averages independent fits. Worth comparing rather than assuming."""

    name: str = "random_forest"
    columns: list[str] = field(default_factory=lambda: list(FEATURE_TIERS))
    model_: object | None = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForest":
        from sklearn.ensemble import RandomForestClassifier

        self.model_ = RandomForestClassifier(
            n_estimators=400, max_depth=8, min_samples_leaf=20,
            max_features="sqrt", class_weight=None, n_jobs=-1, random_state=42,
        )
        self.model_.fit(X[self.columns].astype(float).fillna(0.0), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.columns].astype(float).fillna(0.0))[:, 1]


@dataclass
class SupportVector:
    """RBF-kernel SVM. Needs scaling, and its decision function is not a
    probability, so Platt scaling is applied inside the classifier.

    Included because the syllabus asks for it and because a max-margin method is
    a genuinely different inductive bias from trees, not because it is expected
    to win: kernel methods scale poorly with rows and struggle with the mixed
    numeric ranges of a tabular feature set.
    """

    name: str = "svm_rbf"
    columns: list[str] = field(default_factory=lambda: list(FEATURE_TIERS))
    C: float = 1.0
    model_: Pipeline | None = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SupportVector":
        from sklearn.svm import SVC

        self.model_ = Pipeline([
            ("scale", StandardScaler()),
            ("clf", SVC(C=self.C, kernel="rbf", gamma="scale", probability=True,
                        cache_size=500, random_state=42)),
        ])
        self.model_.fit(X[self.columns].astype(float).fillna(0.0), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.columns].astype(float).fillna(0.0))[:, 1]


@dataclass
class NeuralNet:
    """A small feed-forward network on the same tabular features.

    Two hidden layers, early stopping, scaled inputs. Deliberately small: with
    roughly 300 positives in an early training fold, capacity is the enemy. This
    is the fair tabular comparison for a neural approach; the sequence model in
    ``sequence.py`` is the version that gets to use structure trees cannot.
    """

    name: str = "mlp"
    columns: list[str] = field(default_factory=lambda: list(FEATURE_TIERS))
    hidden: tuple[int, ...] = (64, 32)
    model_: Pipeline | None = field(default=None, repr=False)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NeuralNet":
        from sklearn.neural_network import MLPClassifier

        self.model_ = Pipeline([
            ("scale", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=self.hidden, activation="relu", alpha=1e-2,
                learning_rate_init=3e-3, max_iter=400, early_stopping=True,
                n_iter_no_change=20, validation_fraction=0.15, random_state=42)),
        ])
        self.model_.fit(X[self.columns].astype(float).fillna(0.0), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X[self.columns].astype(float).fillna(0.0))[:, 1]


def ladder(full: bool = False) -> list[object]:
    """The ladder, in increasing complexity. Order matters for reporting.

    ``full`` adds the slower model classes the syllabus asks to see compared:
    a bagged ensemble, a kernel method and a feed-forward network.
    """
    core = [
        BaseRate(),
        LogisticBaseline(name="gap_only", columns=["gap_ahead"]),
        LogisticBaseline(name="logistic_5", columns=FIVE),
        GradientBoosting(name="lightgbm", monotone=True),
    ]
    if not full:
        return core
    return core + [
        LogisticBaseline(name="logistic_all", columns=list(FEATURE_TIERS), C=0.1),
        RandomForest(),
        SupportVector(),
        NeuralNet(),
    ]
