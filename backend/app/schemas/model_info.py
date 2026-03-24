from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class FeatureSchemaItem(BaseModel):
    name: str
    kind: str  # "number" | "boolean" | "string" | "category"
    default: Any | None = None
    min: float | None = None
    max: float | None = None
    options: list[Any] | None = None
    group: str | None = None


class ModelCurrentResponse(BaseModel):
    version: str
    data_version: str | None = None
    threshold: float
    train_years: list[int] | None = None
    train_rows: int | None = None
    features_count: int
    cv_metrics: dict[str, float] | None = None
    meta: dict[str, Any] | None = None


class ModelsSchemaResponse(BaseModel):
    model_version: str
    features: list[FeatureSchemaItem]
