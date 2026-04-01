"""Model-level (global) feature importance from the trained estimator — not local ΔP."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def _get_final_estimator(pipeline: Any) -> Any | None:
    steps = getattr(pipeline, "named_steps", None) or {}
    clf = steps.get("classifier") or steps.get("clf")
    if clf is None and hasattr(pipeline, "steps"):
        clf = pipeline.steps[-1][1]
    if clf is None:
        return None
    if hasattr(clf, "calibrated_classifiers_"):
        return clf.calibrated_classifiers_[0].estimator
    return clf


def _linear_importance_ranking(est: Any, feats: list[str]) -> list[dict[str, Any]]:
    coef = getattr(est, "coef_", None)
    if coef is None:
        return []
    arr = np.abs(np.asarray(coef, dtype=float)).ravel()
    if len(arr) != len(feats):
        return []
    total = float(arr.sum()) or 1.0
    pairs = sorted(zip(feats, arr / total), key=lambda x: -x[1])
    return [{"feature": f, "importance": float(v)} for f, v in pairs]


def global_feature_importance_ranking(pipeline: Any, meta: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Training-data global signal: XGBoost MDI split across one-hot columns is summed per
    original feature; numeric columns map 1:1. Normalized to sum ≈ 1.
    """
    feats = list(meta.get("features") or [])
    if not feats:
        return []

    base = _get_final_estimator(pipeline)
    if base is None:
        return []

    pre = getattr(pipeline, "named_steps", {}).get("preprocess")
    if pre is None or not hasattr(pre, "get_feature_names_out"):
        return _linear_importance_ranking(base, feats)

    names = pre.get_feature_names_out()
    if not hasattr(base, "feature_importances_"):
        return _linear_importance_ranking(base, feats)

    fi = np.asarray(base.feature_importances_, dtype=float)
    if len(names) != len(fi):
        return []

    cat_cols = sorted(list(meta.get("cat_cols") or []), key=len, reverse=True)
    agg: dict[str, float] = defaultdict(float)

    for n, w in zip(names, fi):
        s = str(n)
        if s.startswith("num__"):
            agg[s[5:]] += float(w)
        elif s.startswith("cat__"):
            rest = s[5:]
            matched = False
            for col in cat_cols:
                prefix = col + "_"
                if rest.startswith(prefix):
                    agg[col] += float(w)
                    matched = True
                    break
            if not matched:
                agg[f"__{rest}"] += float(w)
        else:
            agg[s] += float(w)

    total = sum(agg.values()) or 1.0
    ranked = sorted(((k, v / total) for k, v in agg.items()), key=lambda x: -x[1])
    return [{"feature": k, "importance": float(v)} for k, v in ranked]
