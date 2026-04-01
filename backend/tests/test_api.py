from __future__ import annotations

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from pipeline.constructor_standings import clear_standings_cache


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


def test_models_global_importance(client: TestClient) -> None:
    r = client.get("/api/models/importance")
    assert r.status_code == 200
    j = r.json()
    assert j["model_version"] == "test"
    assert isinstance(j["importance"], list)
    assert len(j["importance"]) >= 5
    assert all("feature" in x and "importance" in x for x in j["importance"])


def test_models_schema(client: TestClient) -> None:
    r = client.get("/api/models/schema")
    assert r.status_code == 200
    j = r.json()
    assert j["model_version"] == "test"
    assert len(j["features"]) >= 5
    names = [f["name"] for f in j["features"]]
    assert "f0" in names
    assert j.get("trained_feature_names") == ["f0", "f1", "f2", "f3", "f4"]
    assert j.get("ui_year") == 2025


def test_models_versions(client: TestClient) -> None:
    r = client.get("/api/models/versions")
    assert r.status_code == 200
    assert set(r.json()["versions"]) == {"test", "test2"}


def test_models_switch(client: TestClient) -> None:
    r = client.post("/api/models/switch", json={"version": "test2"})
    assert r.status_code == 200
    assert r.json()["active"] == "test2"
    cur = client.get("/api/models/current")
    assert cur.json()["version"] == "test2"
    client.post("/api/models/switch", json={"version": "test"})


def test_standings_mocked(client: TestClient) -> None:
    clear_standings_cache()

    class _FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "season": 2025,
                    "constructors_championship": [
                        {
                            "teamId": "mclaren",
                            "position": 1,
                            "points": 100.0,
                            "wins": 5,
                            "team": {"teamName": "McLaren"},
                        }
                    ],
                }
            ).encode()

    with patch("urllib.request.urlopen", return_value=_FakeResp()):
        r = client.get("/api/standings?year=2025")
    assert r.status_code == 200
    body = r.json()
    assert body["season"] == 2025
    assert len(body["entries"]) == 1
    assert body["entries"][0]["app_team"] == "McLaren"


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


def test_predict_derive_row(client: TestClient) -> None:
    r = client.post(
        "/api/predict/derive",
        json={"inputs": {"race_name": "Italian Grand Prix", "lap_number": 10, "total_laps": 53}},
    )
    assert r.status_code == 200
    body = r.json()
    assert "row" in body
    assert "race_progress" in body["row"]


def test_predict_derive_blank_string_dropped(client: TestClient) -> None:
    """Empty strings must not reach int() — treated as missing so defaults apply."""
    r = client.post(
        "/api/predict/derive",
        json={
            "inputs": {
                "race_name": "Italian Grand Prix",
                "lap_number": "",
                "total_laps": 53,
            }
        },
    )
    assert r.status_code == 200
    assert r.json()["row"]["lap_number"] == 35


def test_predict_single_blank_extra_keys(client: TestClient, raw_vector: dict) -> None:
    body = {**raw_vector, "sector": "  "}
    r = client.post("/api/predict/single", json={"inputs": body, "include_impacts": False})
    assert r.status_code == 200
    assert 0.0 <= r.json()["probability"] <= 1.0


def test_sensitivity(client: TestClient, raw_vector: dict) -> None:
    r = client.post(
        "/api/sensitivity",
        json={"inputs": raw_vector, "feature": "f0", "min": -2.0, "max": 2.0, "steps": 5},
    )
    assert r.status_code == 200
    j = r.json()
    assert "curve" in j
    assert len(j["curve"]) >= 2
