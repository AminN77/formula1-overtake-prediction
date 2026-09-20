from __future__ import annotations

import io
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


def test_predict_batch_supports_label_column(client: TestClient) -> None:
    csv_text = "\n".join(
        [
            "f0,f1,f2,f3,f4,label",
            "0.0,1.0,2.0,3.0,4.0,0",
            "1.0,2.0,3.0,4.0,5.0,1",
            "2.0,3.0,4.0,5.0,6.0,1",
        ]
    )
    r = client.post(
        "/api/predict/batch?threshold=0.5&filter_pits=false",
        files={"file": ("batch.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["label_column"] == "label"
    assert body["evaluation"]["has_labels"] is True
    assert body["evaluation"]["label_column"] == "label"
    assert "eval_outcome" in body["columns"]
    assert "overtake_predicted" in body["columns"]


def test_predict_batch_preview_rows_limits_json_payload(client: TestClient) -> None:
    """Initial batch payload should honor the preview_rows alias for page size."""
    lines = ["f0,f1,f2,f3,f4,label"]
    for i in range(10):
        lines.append(f"{i}.0,{i + 1}.0,{i + 2}.0,{i + 3}.0,{i + 4}.0,{i % 2}")
    csv_text = "\n".join(lines)
    r = client.post(
        "/api/predict/batch?threshold=0.5&filter_pits=false&preview_rows=3",
        files={"file": ("batch.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["row_count"] == 10
    assert len(body["rows"]) == 3
    assert body["summary"]["rows_truncated"] is True
    assert body["summary"]["rows_in_response"] == 3
    assert body["page_size"] == 3
    assert "result_id" in body


def test_predict_batch_query_pages_filtered_rows(client: TestClient) -> None:
    csv_text = "\n".join(
        [
            "f0,f1,f2,f3,f4,label,attacker,defender,race_name,track",
            "0.0,1.0,2.0,3.0,4.0,0,NOR,HAM,Race A,MELBOURNE",
            "1.0,2.0,3.0,4.0,5.0,1,PIA,HAM,Race A,MELBOURNE",
            "2.0,3.0,4.0,5.0,6.0,1,PIA,LEC,Race B,MONZA",
            "3.0,4.0,5.0,6.0,7.0,0,NOR,LEC,Race B,MONZA",
        ]
    )
    created = client.post(
        "/api/predict/batch?threshold=0.5&filter_pits=false&page_size=2",
        files={"file": ("batch.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")},
    )
    assert created.status_code == 200
    result_id = created.json()["result_id"]
    queried = client.post(
        "/api/predict/batch/query",
        json={
            "result_id": result_id,
            "page": 1,
            "page_size": 10,
            "outcome": "ALL",
            "prediction": "ALL",
            "attacker": "PIA",
            "defender": "ALL",
            "race_name": "ALL",
            "track": "ALL",
            "search": "",
            "lap_min": None,
            "lap_max": None,
            "probability_min": None,
        },
    )
    assert queried.status_code == 200
    body = queried.json()
    assert body["filtered_row_count"] == 2
    assert all(row["attacker"] == "PIA" for row in body["rows"])


def test_predict_batch_download_returns_csv(client: TestClient) -> None:
    csv_text = "\n".join(
        [
            "f0,f1,f2,f3,f4,label",
            "0.0,1.0,2.0,3.0,4.0,0",
            "1.0,2.0,3.0,4.0,5.0,1",
        ]
    )
    created = client.post(
        "/api/predict/batch?threshold=0.5&filter_pits=false&page_size=1",
        files={"file": ("batch.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")},
    )
    assert created.status_code == 200
    result_id = created.json()["result_id"]
    downloaded = client.get(f"/api/predict/batch/download/{result_id}")
    assert downloaded.status_code == 200
    assert downloaded.headers["content-type"].startswith("text/csv")
    assert "overtake_probability" in downloaded.text


def test_predict_batch_reports_horizon_breakdown(client: TestClient) -> None:
    csv_text = "\n".join(
        [
            "f0,f1,f2,f3,f4,label,overtake_next_lap,overtake_within_2,overtake_within_3",
            "0.0,1.0,2.0,3.0,4.0,0,0,0,0",
            "1.0,2.0,3.0,4.0,5.0,1,0,1,1",
            "2.0,3.0,4.0,5.0,6.0,1,1,1,1",
        ]
    )
    r = client.post(
        "/api/predict/batch?threshold=0.5&filter_pits=false",
        files={"file": ("batch.csv", io.BytesIO(csv_text.encode("utf-8")), "text/csv")},
    )
    assert r.status_code == 200
    body = r.json()
    breakdown = body["evaluation"]["horizon_breakdown"]
    assert [item["column"] for item in breakdown] == [
        "overtake_next_lap",
        "overtake_within_2",
        "overtake_within_3",
    ]
    assert breakdown[0]["positive_rows"] == 1
    assert "predicted_true" in breakdown[1]


def test_sensitivity(client: TestClient, raw_vector: dict) -> None:
    r = client.post(
        "/api/sensitivity",
        json={"inputs": raw_vector, "feature": "f0", "min": -2.0, "max": 2.0, "steps": 5},
    )
    assert r.status_code == 200
    j = r.json()
    assert "curve" in j
    assert len(j["curve"]) >= 2
