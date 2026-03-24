from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.model_registry import ModelRegistry

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "artifacts"


@pytest.fixture
def test_registry() -> ModelRegistry:
    reg = ModelRegistry(FIXTURES, "test")
    reg.load()
    return reg


@pytest.fixture
def client(test_registry: ModelRegistry) -> TestClient:
    app = create_app(registry=test_registry)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def raw_vector() -> dict:
    return {f"f{i}": float(i) for i in range(5)}
