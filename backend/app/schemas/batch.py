from __future__ import annotations

from pydantic import BaseModel, Field


class BatchQueryRequest(BaseModel):
    result_id: str
    page: int = Field(1, ge=1)
    page_size: int = Field(25, ge=1, le=500)
    outcome: str = "ALL"
    prediction: str = "ALL"
    attacker: str = "ALL"
    defender: str = "ALL"
    race_name: str = "ALL"
    track: str = "ALL"
    search: str = ""
    lap_min: float | None = None
    lap_max: float | None = None
    probability_min: float | None = None
