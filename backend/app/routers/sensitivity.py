from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from ..schemas.battle import SensitivityRequest, SensitivityResponse
from ..services.inference import sensitivity_curve
from ..services.model_registry import ModelRegistry

router = APIRouter(prefix="/sensitivity", tags=["sensitivity"])


def get_registry(request: Request) -> ModelRegistry:
    reg = getattr(request.app.state, "registry", None)
    if reg is None:
        raise HTTPException(500, "Model registry not initialized")
    return reg


@router.post("", response_model=SensitivityResponse)
def sensitivity_endpoint(
    body: SensitivityRequest,
    reg: Annotated[ModelRegistry, Depends(get_registry)],
) -> SensitivityResponse:
    m = reg.active
    try:
        base_p, curve = sensitivity_curve(
            m.pipeline,
            m.meta,
            body.inputs,
            body.feature,
            values=body.values,
            vmin=body.min,
            vmax=body.max,
            steps=body.steps,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(400, f"Sensitivity failed: {e}") from e

    return SensitivityResponse(
        baseline_probability=base_p,
        curve=curve,
        feature=body.feature,
    )
