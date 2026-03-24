from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas.battle import SwitchModelRequest
from ..schemas.model_info import ModelCurrentResponse, ModelsSchemaResponse
from ..services.model_registry import LoadedModel, ModelRegistry
from ..services.schema_builder import build_feature_schema

router = APIRouter(prefix="/models", tags=["models"])


def get_registry(request: Request) -> ModelRegistry:
    reg = getattr(request.app.state, "registry", None)
    if reg is None:
        raise HTTPException(500, "Model registry not initialized")
    return reg


@router.get("/current", response_model=ModelCurrentResponse)
def models_current(
    reg: Annotated[ModelRegistry, Depends(get_registry)],
) -> ModelCurrentResponse:
    m = reg.active
    meta = m.meta
    entry_data = reg.data_version_for(m.version)
    return ModelCurrentResponse(
        version=m.version,
        data_version=entry_data,
        threshold=float(meta.get("threshold", 0.5)),
        train_years=list(meta.get("train_years") or []),
        train_rows=meta.get("train_rows"),
        features_count=len(meta.get("features") or []),
        cv_metrics=meta.get("cv_metrics"),
        meta={k: v for k, v in meta.items() if k != "best_params"},
    )


@router.get("/schema", response_model=ModelsSchemaResponse)
def models_schema(
    reg: Annotated[ModelRegistry, Depends(get_registry)],
) -> ModelsSchemaResponse:
    m = reg.active
    items = build_feature_schema(m.meta)
    return ModelsSchemaResponse(model_version=m.version, features=items)


@router.get("/versions")
def model_versions(reg: Annotated[ModelRegistry, Depends(get_registry)]) -> dict[str, list[str]]:
    return {"versions": reg.available_versions()}


@router.post("/switch")
def switch_model(
    body: SwitchModelRequest,
    reg: Annotated[ModelRegistry, Depends(get_registry)],
) -> dict[str, str]:
    try:
        reg.load(body.version)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    return {"active": reg.active.version}


def _loaded(reg: ModelRegistry) -> LoadedModel:
    return reg.active
