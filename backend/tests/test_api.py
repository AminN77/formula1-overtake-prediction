from __future__ import annotations

from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_models_current(client: TestClient) -> None:
    r = client.get("/api/models/current")
    assert r.status_code == 200
    j = r.json()
    assert j["version"] == "test"
    assert j["features_count"] == 5


def test_models_schema(client: TestClient) -> None:
    r = client.get("/api/models/schema")
    assert r.status_code == 200
    j = r.json()
    assert j["model_version"] == "test"
    assert len(j["features"]) == 5
    assert j.get("ui_year") == 2025


def test_circuits(client: TestClient) -> None:
    r = client.get("/api/circuits")
    assert r.status_code == 200
    j = r.json()
    assert j["season"] == 2025
    assert len(j["circuits"]) >= 20


def test_predict_single_raw_vector(client: TestClient, raw_vector: dict) -> None:
    r = client.post("/api/predict/single", json={"inputs": raw_vector, "include_impacts": True})
    assert r.status_code == 200
    j = r.json()
    assert 0.0 <= j["probability"] <= 1.0
    assert "impacts" in j


def test_sensitivity(client: TestClient, raw_vector: dict) -> None:
    r = client.post(
        "/api/sensitivity",
        json={"inputs": raw_vector, "feature": "f0", "min": -2.0, "max": 2.0, "steps": 5},
    )
    assert r.status_code == 200
    j = r.json()
    assert "curve" in j
    assert len(j["curve"]) >= 2
