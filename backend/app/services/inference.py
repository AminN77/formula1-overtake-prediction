"""Prediction and sensitivity helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .feature_builder import build_single_row, clean_raw_inputs, dataframe_for_model, engineer_batch_features


def derive_engineered_row(raw: dict[str, Any]) -> dict[str, Any]:
    """Full logical battle row from UI inputs (same as inference path, no model)."""
    return build_single_row(raw)


def predict_proba_df(pipeline: Any, df: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(df)[:, 1]


def _feature_frame(raw: dict[str, Any], meta: dict[str, Any]) -> pd.DataFrame:
    """If `raw` contains all model feature keys, use it directly; else engineer from battle UI."""
    raw = clean_raw_inputs(dict(raw))
    feats: list[str] = list(meta.get("features") or [])
    if feats and all(k in raw for k in feats):
        return dataframe_for_model({k: raw[k] for k in feats}, feats)
    row = build_single_row(raw)
    return dataframe_for_model(row, feats)


def predict_single(pipeline: Any, meta: dict[str, Any], raw: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    feats: list[str] = list(meta.get("features") or [])
    X = _feature_frame(raw, meta)
    p = float(predict_proba_df(pipeline, X)[0])
    if feats and all(k in raw for k in feats):
        row = {k: raw[k] for k in feats}
    else:
        row = build_single_row(raw)
    return p, row


def predict_batch(
    pipeline: Any,
    meta: dict[str, Any],
    df: pd.DataFrame,
    filter_pits: bool,
) -> pd.DataFrame:
    if filter_pits and "pit_stop_involved" in df.columns:
        df = df[~df["pit_stop_involved"]].reset_index(drop=True)
    df = engineer_batch_features(df)
    feats = meta["features"]
    for c in feats:
        if c not in df.columns:
            df[c] = 0
    probas = predict_proba_df(pipeline, df[feats])
    df = df.copy()
    df["overtake_probability"] = np.round(probas, 4)
    return df


def sensitivity_curve(
    pipeline: Any,
    meta: dict[str, Any],
    base_raw: dict[str, Any],
    feature: str,
    values: list[float] | None = None,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    steps: int = 24,
) -> tuple[float, list[dict[str, Any]]]:
    """Vary one *model input column* across a numeric range in feature space."""
    cat_cols = set(meta.get("cat_cols") or [])
    if feature in cat_cols:
        raise ValueError(f"Feature '{feature}' is categorical — not supported in this curve.")

    feats = meta["features"]
    if feature not in feats:
        raise ValueError(f"Unknown feature '{feature}' for active model.")

    X = _feature_frame(base_raw, meta)
    base_p = float(predict_proba_df(pipeline, X)[0])

    if values is None:
        if vmin is None or vmax is None:
            cur = float(X[feature].iloc[0])
            span = max(abs(cur) * 0.5, 1.0)
            vmin = cur - span
            vmax = cur + span
        values = list(np.linspace(float(vmin), float(vmax), max(2, steps)))

    out: list[dict[str, Any]] = []
    for v in values:
        X2 = X.copy()
        X2[feature] = float(v)
        p = float(predict_proba_df(pipeline, X2)[0])
        out.append({"value": float(v), "probability": p})
    return base_p, out


def local_feature_impacts(
    pipeline: Any,
    meta: dict[str, Any],
    base_raw: dict[str, Any],
    *,
    max_features: int = 12,
    relative_bump: float = 0.05,
) -> list[dict[str, Any]]:
    """Approximate |Δp| by perturbing each numeric *model input* column in X."""
    feats = meta["features"]
    X = _feature_frame(base_raw, meta)
    base_p = float(predict_proba_df(pipeline, X)[0])
    nums = set(meta.get("num_cols") or [])
    by_feat: dict[str, float] = {}
    for feat in nums:
        if feat not in X.columns:
            continue
        cur = float(X[feat].iloc[0])
        eps = max(abs(cur) * relative_bump, 1e-6)
        X2 = X.copy()
        X2[feat] = cur + eps
        try:
            p2 = float(predict_proba_df(pipeline, X2)[0])
        except Exception:
            continue
        by_feat[feat] = abs(p2 - base_p)
    ranked = sorted(by_feat.items(), key=lambda x: -x[1])[:max_features]
    return [{"feature": f, "max_abs_delta_probability": float(d)} for f, d in ranked]
