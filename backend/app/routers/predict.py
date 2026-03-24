from __future__ import annotations

import io
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sklearn.metrics import average_precision_score, roc_auc_score

from ..schemas.battle import PredictSingleRequest, PredictSingleResponse
from ..services.inference import local_feature_impacts, predict_batch, predict_single
from ..services.model_registry import ModelRegistry

router = APIRouter(prefix="/predict", tags=["predict"])


def get_registry(request: Request) -> ModelRegistry:
    reg = getattr(request.app.state, "registry", None)
    if reg is None:
        raise HTTPException(500, "Model registry not initialized")
    return reg


def _verdict_label(p: float, threshold: float) -> tuple[str, str]:
    if p >= threshold:
        return "overtake", "Predicted overtake"
    if p < 0.05:
        return "hold", "Very unlikely"
    if p < 0.15:
        return "hold", "Possible"
    if p < 0.30:
        return "hold", "Likely"
    if p < 0.50:
        return "hold", "Very likely"
    return "hold", "Highly likely"


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
    }
    if "overtake" in scored.columns:
        y = scored["overtake"].astype(int).values
        summary["actual_positive_rate"] = float(y.mean()) if len(y) else 0.0
        if 0 < int(y.sum()) < len(y):
            summary["roc_auc"] = float(roc_auc_score(y, probas))
            summary["pr_auc"] = float(average_precision_score(y, probas))

    csv_buf = io.StringIO()
    scored.to_csv(csv_buf, index=False)
    import base64

    b64 = base64.b64encode(csv_buf.getvalue().encode("utf-8")).decode("ascii")
    return {
        "summary": summary,
        "preview_columns": list(scored.columns),
        "preview_rows": scored.head(50).replace({np.nan: None}).to_dict(orient="records"),
        "csv_base64": b64,
    }
