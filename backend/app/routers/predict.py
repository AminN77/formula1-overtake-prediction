from __future__ import annotations

import io
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ..schemas.battle import DeriveRowRequest, DeriveRowResponse, PredictSingleRequest, PredictSingleResponse
from ..services.inference import derive_engineered_row, local_feature_impacts, predict_batch, predict_single
from ..services.model_registry import ModelRegistry

router = APIRouter(prefix="/predict", tags=["predict"])


def get_registry(request: Request) -> ModelRegistry:
    reg = getattr(request.app.state, "registry", None)
    if reg is None:
        raise HTTPException(500, "Model registry not initialized")
    return reg


def _jsonable_row(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): _jsonable_row(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable_row(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _verdict_label(p: float, threshold: float) -> tuple[str, str]:
    """Verdict uses trained threshold; labels below threshold describe score bands without implying pass/fail."""
    if p >= threshold:
        return "overtake", "Predicted overtake"
    if p < 0.05:
        return "hold", "Below threshold · score band: very low"
    if p < 0.15:
        return "hold", "Below threshold · score band: low"
    if p < 0.30:
        return "hold", "Below threshold · score band: moderate-low"
    if p < 0.50:
        return "hold", "Below threshold · score band: moderate"
    if p < threshold:
        return "hold", "Below threshold · score band: high"
    return "hold", "Below threshold · score band: very high"


@router.post("/single", response_model=PredictSingleResponse)
def predict_single_endpoint(
    body: PredictSingleRequest,
    reg: Annotated[ModelRegistry, Depends(get_registry)],
) -> PredictSingleResponse:
    m = reg.active
    meta = m.meta
    th = float(meta.get("threshold", 0.5))
    try:
        p, row = predict_single(m.pipeline, meta, body.inputs)
    except Exception as e:
        raise HTTPException(400, f"Prediction failed: {e}") from e

    verdict, label = _verdict_label(p, th)
    impacts = None
    if body.include_impacts:
        try:
            impacts = local_feature_impacts(m.pipeline, meta, body.inputs)
        except Exception:
            impacts = []
    row_out = row if body.include_row else None
    return PredictSingleResponse(
        probability=p,
        threshold=th,
        verdict=verdict,
        label=label,
        model_version=m.version,
        impacts=impacts,
        row=row_out,
    )


@router.post("/derive", response_model=DeriveRowResponse)
def derive_row_endpoint(body: DeriveRowRequest) -> DeriveRowResponse:
    """Return `build_single_row` output for UI readonly fields (no scoring)."""
    try:
        row = derive_engineered_row(dict(body.inputs))
    except Exception as e:
        raise HTTPException(400, f"Derive failed: {e}") from e
    return DeriveRowResponse(row=_jsonable_row(row))  # type: ignore[arg-type]


@router.post("/batch")
def predict_batch_endpoint(
    reg: Annotated[ModelRegistry, Depends(get_registry)],
    file: UploadFile = File(...),
    threshold: float = 0.5,
    filter_pits: bool = True,
) -> dict:
    m = reg.active
    meta = m.meta
    th = float(threshold)
    try:
        raw = file.file.read()
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8")
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}") from e

    n0 = len(df)
    try:
        scored = predict_batch(m.pipeline, meta, df, filter_pits=filter_pits)
    except Exception as e:
        raise HTTPException(400, f"Batch prediction failed: {e}") from e

    probas = scored["overtake_probability"].values
    scored["overtake_predicted"] = (probas >= th).astype(int)

    summary: dict = {
        "rows": len(scored),
        "rows_input": n0,
        "threshold": th,
        "predicted_positive_rate": float(scored["overtake_predicted"].mean())
        if len(scored)
        else 0.0,
        "model_version": m.version,
    }
    evaluation: dict | None = None
    if "overtake" in scored.columns:
        y = scored["overtake"].astype(int).values
        pred = scored["overtake_predicted"].astype(int).values
        summary["actual_positive_rate"] = float(y.mean()) if len(y) else 0.0
        if 0 < int(y.sum()) < len(y):
            summary["roc_auc"] = float(roc_auc_score(y, probas))
            summary["pr_auc"] = float(average_precision_score(y, probas))

        tp = int(((y == 1) & (pred == 1)).sum())
        fp = int(((y == 0) & (pred == 1)).sum())
        tn = int(((y == 0) & (pred == 0)).sum())
        fn = int(((y == 1) & (pred == 0)).sum())
        n_lab = len(y)
        acc = float((tp + tn) / n_lab) if n_lab else 0.0
        prec = float(tp / (tp + fp)) if (tp + fp) else None
        rec = float(tp / (tp + fn)) if (tp + fn) else None
        f1 = float(f1_score(y, pred, zero_division=0)) if n_lab else None
        evaluation = {
            "has_labels": True,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "confusion_labels": {"rows": ["actual 0", "actual 1"], "cols": ["pred 0", "pred 1"]},
        }
    else:
        evaluation = {"has_labels": False}

    scored_display = scored.copy()
    if "overtake" in scored_display.columns:
        yv = scored_display["overtake"].astype(int).values
        pv = scored_display["overtake_predicted"].astype(int).values
        outcomes: list[str | None] = []
        for yi, pi in zip(yv, pv):
            if yi == 1 and pi == 1:
                outcomes.append("TP")
            elif yi == 0 and pi == 1:
                outcomes.append("FP")
            elif yi == 0 and pi == 0:
                outcomes.append("TN")
            else:
                outcomes.append("FN")
        scored_display["eval_outcome"] = outcomes

    rows_json = scored_display.replace({np.nan: None}).to_dict(orient="records")

    csv_buf = io.StringIO()
    scored_display.to_csv(csv_buf, index=False)
    import base64

    b64 = base64.b64encode(csv_buf.getvalue().encode("utf-8")).decode("ascii")
    return {
        "summary": summary,
        "evaluation": evaluation,
        "columns": list(scored_display.columns),
        "rows": rows_json,
        "row_count": len(rows_json),
        "csv_base64": b64,
    }
