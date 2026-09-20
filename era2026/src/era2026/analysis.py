"""
Explainability, error analysis and learning curves.

Three questions a metric table cannot answer: what is the model actually using,
where does it fail, and is it limited by data or by capacity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from era2026.episodes import trainable
from era2026.features import FEATURE_TIERS, model_features


# ---------------------------------------------------------------- explainability
def permutation_importance(
    model, test: pd.DataFrame, columns: list[str], repeats: int = 5, seed: int = 42
) -> pd.DataFrame:
    """Drop in PR-AUC when one feature is shuffled.

    Preferred over a tree's built-in importance, which counts splits rather than
    contribution and inflates high-cardinality features. This measures what the
    model would lose if the feature carried no information, on data it did not
    train on.
    """
    rng = np.random.default_rng(seed)
    y = test["label"].to_numpy()
    if len(np.unique(y)) < 2:
        return pd.DataFrame()
    baseline = average_precision_score(y, model.predict_proba(test))

    records = []
    for column in columns:
        drops = []
        for _ in range(repeats):
            shuffled = test.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            drops.append(baseline - average_precision_score(y, model.predict_proba(shuffled)))
        records.append({
            "feature": column, "tier": FEATURE_TIERS.get(column, "unknown"),
            "importance": float(np.mean(drops)), "sd": float(np.std(drops)),
        })
    return (pd.DataFrame(records)
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
            .assign(baseline_pr_auc=baseline))


def shap_summary(fitted_tree_model, X: pd.DataFrame, columns: list[str],
                 sample: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Mean absolute SHAP value per feature, plus the sign of its correlation
    with the feature, which says whether more of it raises or lowers the hazard."""
    import shap

    frame = X[columns].astype(float).fillna(0.0)
    if len(frame) > sample:
        frame = frame.sample(sample, random_state=seed)

    explainer = shap.TreeExplainer(fitted_tree_model)
    values = explainer.shap_values(frame)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:  # (rows, features, classes)
        values = values[:, :, -1]

    records = []
    for i, column in enumerate(columns):
        column_values = frame[column].to_numpy()
        shap_column = values[:, i]
        direction = 0.0
        if np.std(column_values) > 0 and np.std(shap_column) > 0:
            direction = float(np.corrcoef(column_values, shap_column)[0, 1])
        records.append({
            "feature": column, "tier": FEATURE_TIERS.get(column, "unknown"),
            "mean_abs_shap": float(np.mean(np.abs(shap_column))),
            "direction": direction,
        })
    return pd.DataFrame(records).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- error analysis
KEY = ["episode_id", "lap_number"]


def attach(predictions: pd.DataFrame, rows: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Join row attributes onto predictions.

    A merge rather than a per-row lookup: an index ``.get`` silently returns a
    Series wherever a key repeats, which poisons every downstream groupby with
    unhashable values instead of failing loudly.
    """
    wanted = [c for c in columns if c in rows.columns]
    side = rows[KEY + wanted].drop_duplicates(subset=KEY)
    return predictions.merge(side, on=KEY, how="left")


def error_by_group(predictions: pd.DataFrame, rows: pd.DataFrame, by: str) -> pd.DataFrame:
    """Where the model does well and badly, sliced by any column of the rows."""
    merged = attach(predictions, rows, [by])

    records = []
    for value, group in merged.groupby(by, dropna=True):
        y, p = group["label"].to_numpy(), group["hazard"].to_numpy()
        if len(y) < 25:
            continue
        records.append({
            by: value, "rows": len(y), "events": int(y.sum()),
            "positive_rate": float(y.mean()),
            "mean_hazard": float(p.mean()),
            "bias": float(p.mean() - y.mean()),
            "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        })
    return pd.DataFrame(records).sort_values("pr_auc").reset_index(drop=True)


def error_by_band(predictions: pd.DataFrame, rows: pd.DataFrame,
                  column: str = "gap_ahead", bands: int = 5) -> pd.DataFrame:
    merged = attach(predictions, rows, [column]).dropna(subset=[column])
    merged["band"] = pd.qcut(merged[column], bands, duplicates="drop")

    records = []
    for band, group in merged.groupby("band", observed=True):
        y, p = group["label"].to_numpy(), group["hazard"].to_numpy()
        records.append({
            "band": str(band), "rows": len(y), "events": int(y.sum()),
            "positive_rate": float(y.mean()), "mean_hazard": float(p.mean()),
            "bias": float(p.mean() - y.mean()),
            "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        })
    return pd.DataFrame(records)


def worst_misses(predictions: pd.DataFrame, rows: pd.DataFrame, n: int = 10) -> dict:
    """The passes the model was most confident would not happen, and the
    non-passes it was most confident would."""
    merged = attach(predictions, rows, ["event_name", "attacker", "defender", "gap_ahead"])
    cols = ["event_name", "attacker", "defender", "lap_number", "gap_ahead", "hazard"]
    return {
        "missed_passes": merged[merged["label"] == 1].nsmallest(n, "hazard")[cols],
        "false_alarms": merged[merged["label"] == 0].nlargest(n, "hazard")[cols],
    }


# ---------------------------------------------------------------- learning curve
def learning_curve(rows: pd.DataFrame, factory, columns: list[str],
                   test_round: int = 14, min_rounds: int = 2) -> pd.DataFrame:
    """Train on a growing prefix of rounds, always score the same held-out race.

    Isolates the effect of training size from the effect of which race is being
    predicted, which the walk-forward backtest deliberately confounds. Train
    score is reported alongside so the generalisation gap is visible.
    """
    pool = trainable(rows)
    test = pool[pool["round_number"] == test_round]
    records = []
    for k in range(min_rounds, test_round):
        train = pool[pool["round_number"] <= k]
        if train["label"].nunique() < 2 or test.empty:
            continue
        model = factory().fit(train, train["label"])
        p_train = model.predict_proba(train)
        p_test = model.predict_proba(test)
        records.append({
            "train_rounds": k,
            "train_rows": len(train),
            "train_events": int(train["label"].sum()),
            "train_pr_auc": float(average_precision_score(train["label"], p_train)),
            "test_pr_auc": float(average_precision_score(test["label"], p_test))
            if test["label"].nunique() > 1 else np.nan,
        })
    frame = pd.DataFrame(records)
    if len(frame):
        frame["gap"] = frame["train_pr_auc"] - frame["test_pr_auc"]
    return frame
