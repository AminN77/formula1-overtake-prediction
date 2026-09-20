from __future__ import annotations

import io
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, Response, UploadFile
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from ..schemas.batch import BatchQueryRequest
from ..schemas.battle import DeriveRowRequest, DeriveRowResponse, PredictSingleRequest, PredictSingleResponse
from ..services.inference import derive_engineered_row, local_feature_impacts, predict_batch, predict_single
from ..services.batch_result_store import BatchResultStore, StoredBatchResult
from ..services.model_registry import ModelRegistry

router = APIRouter(prefix="/predict", tags=["predict"])


def get_registry(request: Request) -> ModelRegistry:
    reg = getattr(request.app.state, "registry", None)
    if reg is None:
        raise HTTPException(500, "Model registry not initialized")
    return reg


def get_batch_store(request: Request) -> BatchResultStore:
    store = getattr(request.app.state, "batch_result_store", None)
    if store is None:
        raise HTTPException(500, "Batch result store not initialized")
    return store


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


def _resolve_label_column(meta: dict, df: pd.DataFrame) -> str | None:
    target = meta.get("target")
    candidates: list[str] = []
    if isinstance(target, str) and target:
        candidates.append(target)
    for name in ("overtake", "label"):
        if name not in candidates:
            candidates.append(name)
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _horizon_breakdown(scored: pd.DataFrame) -> list[dict]:
    out: list[dict] = []
    if "overtake_predicted" not in scored.columns:
        return out
    pred = scored["overtake_predicted"].astype(int)
    for col, label in (
        ("overtake_next_lap", "Next lap"),
        ("overtake_within_2", "Within 2 laps"),
        ("overtake_within_3", "Within 3 laps"),
    ):
        if col not in scored.columns:
            continue
        y = scored[col].astype(int)
        positives = int(y.sum())
        predicted_true = int(((y == 1) & (pred == 1)).sum())
        out.append(
            {
                "column": col,
                "label": label,
                "positive_rows": positives,
                "predicted_true": predicted_true,
                "predicted_true_rate": float(predicted_true / positives) if positives else None,
            }
        )
    return out


def _filter_options(scored: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for key in ("attacker", "defender", "race_name", "track"):
        if key not in scored.columns:
            continue
        vals = sorted({str(v) for v in scored[key].dropna().astype(str) if str(v).strip()})
        out[key] = vals
    return out


def _apply_batch_filters(scored: pd.DataFrame, query: BatchQueryRequest) -> pd.DataFrame:
    out = scored
    if query.outcome != "ALL" and "eval_outcome" in out.columns:
        out = out[out["eval_outcome"].astype(str) == query.outcome]
    if query.prediction == "Predicted positive" and "overtake_predicted" in out.columns:
        out = out[out["overtake_predicted"].astype(int) == 1]
    elif query.prediction == "Predicted negative" and "overtake_predicted" in out.columns:
        out = out[out["overtake_predicted"].astype(int) == 0]
    for field, value in (
        ("attacker", query.attacker),
        ("defender", query.defender),
        ("race_name", query.race_name),
        ("track", query.track),
    ):
        if value != "ALL" and field in out.columns:
            out = out[out[field].astype(str) == value]
    if query.lap_min is not None and "lap_number" in out.columns:
        out = out[pd.to_numeric(out["lap_number"], errors="coerce") >= query.lap_min]
    if query.lap_max is not None and "lap_number" in out.columns:
        out = out[pd.to_numeric(out["lap_number"], errors="coerce") <= query.lap_max]
    if query.probability_min is not None and "overtake_probability" in out.columns:
        out = out[pd.to_numeric(out["overtake_probability"], errors="coerce") >= query.probability_min]
    if query.search.strip():
        needle = query.search.strip().lower()
        hay_cols = [c for c in ("attacker", "defender", "race_name", "track", "eval_outcome", "attacker_team", "defender_team") if c in out.columns]
        if hay_cols:
            mask = out[hay_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower().str.contains(needle, regex=False)
            out = out[mask]
    return out.reset_index(drop=True)


def _serialize_batch_rows(item: StoredBatchResult, query: BatchQueryRequest) -> dict:
    filtered = _apply_batch_filters(item.scored, query)
    total = len(item.scored)
    filtered_total = len(filtered)
    start = (query.page - 1) * query.page_size
    end = start + query.page_size
    page_rows = filtered.iloc[start:end].replace({np.nan: None}).to_dict(orient="records")
    page_count = max(1, int(np.ceil(filtered_total / query.page_size))) if filtered_total else 1
    return {
        "result_id": item.result_id,
        "summary": item.summary,
        "evaluation": item.evaluation,
        "columns": item.columns,
        "filter_options": item.filter_options,
        "rows": page_rows,
        "row_count": total,
        "filtered_row_count": filtered_total,
        "page": query.page,
        "page_size": query.page_size,
        "page_count": page_count,
        "has_more": end < filtered_total,
    }


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
    request: Request,
    reg: Annotated[ModelRegistry, Depends(get_registry)],
    file: UploadFile = File(...),
    threshold: float = 0.5,
    filter_pits: bool = True,
    preview_rows: int | None = Query(
        None,
        ge=1,
        le=500,
        description="Backward-compatible alias for the initial batch page size.",
    ),
    page_size: int = Query(
        25,
        ge=1,
        le=500,
        description="Rows per page for the initial batch result payload.",
    ),
) -> dict:
    m = reg.active
    meta = m.meta
    th = float(threshold)
    store = get_batch_store(request)
    initial_page_size = int(preview_rows or page_size)
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

    n_scored = len(scored)
    summary: dict = {
        "rows": n_scored,
        "rows_input": n0,
        "threshold": th,
        "predicted_positive_rate": float(scored["overtake_predicted"].mean())
        if n_scored
        else 0.0,
        "model_version": m.version,
    }
    evaluation: dict | None = None
    label_col = _resolve_label_column(meta, scored)
    if label_col is not None:
        y = scored[label_col].astype(int).values
        pred = scored["overtake_predicted"].astype(int).values
        summary["label_column"] = label_col
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
            "label_column": label_col,
            "confusion_matrix": [[tn, fp], [fn, tp]],
            "confusion_labels": {"rows": ["actual 0", "actual 1"], "cols": ["pred 0", "pred 1"]},
            "horizon_breakdown": _horizon_breakdown(scored),
        }
    else:
        evaluation = {"has_labels": False}

    scored_display = scored.copy()
    if label_col is not None:
        if label_col != "overtake" and "overtake" not in scored_display.columns:
            scored_display["overtake"] = scored_display[label_col]
        yv = scored_display[label_col].astype(int).values
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

    item = store.save(
        scored=scored_display,
        summary=summary,
        evaluation=evaluation,
        columns=list(scored_display.columns),
        filter_options=_filter_options(scored_display),
    )
    payload = _serialize_batch_rows(
        item,
        BatchQueryRequest(result_id=item.result_id, page=1, page_size=initial_page_size),
    )
    payload["summary"] = {
        **summary,
        "rows_in_response": len(payload["rows"]),
        "rows_truncated": len(payload["rows"]) < n_scored,
    }
    return payload


@router.post("/batch/query")
def query_batch_result(
    body: BatchQueryRequest,
    store: Annotated[BatchResultStore, Depends(get_batch_store)],
) -> dict:
    item = store.get(body.result_id)
    if item is None:
        raise HTTPException(404, "Batch result not found. Re-run the batch score to create a new result set.")
    return _serialize_batch_rows(item, body)


@router.get("/batch/download/{result_id}")
def download_batch_result(
    result_id: str,
    store: Annotated[BatchResultStore, Depends(get_batch_store)],
) -> Response:
    item = store.get(result_id)
    if item is None:
        raise HTTPException(404, "Batch result not found. Re-run the batch score to download it again.")
    csv_buf = io.StringIO()
    item.scored.to_csv(csv_buf, index=False)
    return Response(
        content=csv_buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="predictions.csv"'},
    )
