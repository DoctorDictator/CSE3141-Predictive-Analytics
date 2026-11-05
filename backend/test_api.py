import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend import main


@pytest.fixture
def client():
    with TestClient(main.app) as current:
        yield current


@pytest.fixture
def example():
    info = json.loads((main.ARTIFACTS / "info.json").read_text(encoding="utf-8"))
    row = {}
    for key in info["features"]:
        low, high = info["ranges"][key]
        if key in {"has_leader", "is_truck"} or key.endswith("_present"):
            row[key] = 0
        elif key.endswith("_gap"):
            row[key] = 150 if low <= 150 <= high else low
        elif key == "closing_speed" or key.endswith("_closing_speed"):
            row[key] = 0
        else:
            row[key] = (low + high) / 2
    return row


def test_readiness_artifacts_and_task(client):
    health = client.get("/api/health").json()
    assert health["model_loaded"] is True
    assert health["task"] == "lane-change-behavior"
    for path in ["/api/model/info", "/api/model/metrics", "/api/analytics"]:
        response = client.get(path)
        assert response.status_code == 200
        assert response.json()["task"] == main.TASK
    assert client.get("/api/replay").status_code == 404


def test_prediction_matches_fitted_model(client, example):
    info = client.get("/api/model/info").json()
    model = joblib.load(main.ARTIFACTS / "behavior.joblib")
    response = client.post("/api/predict", json=example)
    assert response.status_code == 200
    class_index = list(model.classes_).index(1)
    expected = model.predict_proba(pd.DataFrame([example], columns=info["features"]))[0, class_index]
    assert response.json()["probability"] == pytest.approx(expected)
    assert response.json()["prediction"] == int(expected >= info["decision_threshold"])
    assert response.json()["label"] in ["Stay in lane", "Lane change"]


def test_batch_ids_and_training_support(client, example):
    payload = {"observations": [
        {"id": "v1", "inputs": example},
        {"id": "v2", "inputs": {**example, "speed": 75}},
    ]}
    response = client.post("/api/predict-batch", json=payload)
    assert response.status_code == 200
    predictions = {item["id"]: item for item in response.json()["predictions"]}
    assert predictions["v1"]["eligible"] is True
    assert predictions["v2"]["eligible"] is False


@pytest.mark.parametrize("field,value", [
    ("speed", -1), ("speed", 81), ("acceleration", 100), ("lateral_speed", -1),
    ("front_gap", 151), ("has_leader", 2), ("is_truck", 2), ("unknown", 1),
])
def test_invalid_features_are_rejected(client, example, field, value):
    assert client.post("/api/predict", json={**example, field: value}).status_code == 422


def test_absent_neighbor_encoding_and_batch_limits(client, example):
    invalid = {**example, "left_front_present": 0, "left_front_gap": 10}
    assert client.post("/api/predict", json=invalid).status_code == 422
    item = {"id": "same", "inputs": example}
    for rows in [[], [item] * 31, [item, item]]:
        assert client.post("/api/predict-batch", json={"observations": rows}).status_code == 422
    assert client.post("/api/predict-batch", json={"pairs": [item]}).status_code == 422


def test_unavailable_model_has_no_substitute_predictions(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "ARTIFACTS", tmp_path)
    with TestClient(main.app) as client:
        assert client.get("/api/health").json()["model_loaded"] is False
        assert client.get("/api/model/info").status_code == 503


def test_exported_artifacts_match_course_task():
    info = json.loads((main.ARTIFACTS / "info.json").read_text(encoding="utf-8"))
    metrics = json.loads((main.ARTIFACTS / "metrics.json").read_text(encoding="utf-8"))
    assert info["version"] == "behavior-3"
    assert len(info["features"]) == 19
    assert not set(info["features"]) & {"target", "vehicle_id", "recording", "lane_id"}
    assert {item["name"] for item in metrics["comparison"]} == {"Logistic Regression", "LDA", "QDA"}
    assert sum(map(sum, metrics["test"]["confusion_matrix"])) == metrics["split_sizes"]["test"]
    assert metrics["baseline"]["f1"] == 0 and metrics["baseline"]["recall"] == 0


def test_cors(client):
    response = client.options("/api/predict", headers={
        "Origin": "http://localhost:5173",
        "Access-Control-Request-Method": "POST",
    })
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
