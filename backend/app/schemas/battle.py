from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictSingleRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    include_impacts: bool = True
    include_row: bool = False


class PredictSingleResponse(BaseModel):
    probability: float
    threshold: float
    verdict: str
    label: str
    model_version: str
    impacts: list[dict[str, Any]] | None = None
    row: dict[str, Any] | None = None


class SensitivityRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)
    feature: str
    values: list[float] | None = None
    min: float | None = Field(None, alias="min")
    max: float | None = Field(None, alias="max")
    steps: int = 24

    model_config = {"populate_by_name": True}


class SensitivityResponse(BaseModel):
    baseline_probability: float
    curve: list[dict[str, Any]]
    feature: str


class SwitchModelRequest(BaseModel):
    version: str


class DeriveRowRequest(BaseModel):
    inputs: dict[str, Any] = Field(default_factory=dict)


class DeriveRowResponse(BaseModel):
    row: dict[str, Any]
