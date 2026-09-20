from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd


@dataclass
class StoredBatchResult:
    result_id: str
    scored: pd.DataFrame
    summary: dict
    evaluation: dict | None
    columns: list[str]
    filter_options: dict[str, list[str]]
    created_at: str


class BatchResultStore:
    def __init__(self, max_items: int = 8) -> None:
        self.max_items = max(1, int(max_items))
        self._items: OrderedDict[str, StoredBatchResult] = OrderedDict()

    def save(
        self,
        *,
        scored: pd.DataFrame,
        summary: dict,
        evaluation: dict | None,
        columns: list[str],
        filter_options: dict[str, list[str]],
    ) -> StoredBatchResult:
        result_id = uuid4().hex
        item = StoredBatchResult(
            result_id=result_id,
            scored=scored.copy(),
            summary=dict(summary),
            evaluation=dict(evaluation) if isinstance(evaluation, dict) else evaluation,
            columns=list(columns),
            filter_options={k: list(v) for k, v in filter_options.items()},
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._items[result_id] = item
        self._items.move_to_end(result_id)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return item

    def get(self, result_id: str) -> StoredBatchResult | None:
        item = self._items.get(result_id)
        if item is not None:
            self._items.move_to_end(result_id)
        return item
