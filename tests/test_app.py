"""Smoke tests for the FastAPI service: it starts, loads the committed model with the pinned
libraries, protects its endpoints, and can make a prediction."""

import pytest
from fastapi.testclient import TestClient

import app as service

PREDICT_KEY = "test-predict-key"
TRAIN_KEY = "test-train-key"

ONE_EVENT = [{
    "userId": "user-1",
    "eventId": 1,
    "startDate": "2026-10-01T10:00:00Z",
    "endDate": "2026-10-01T11:00:00Z",
    "durationMinutes": 60,
    "hourOfDay": 10,
    "dayOfWeek": 3,
    "importance": 1,
    "isRecurring": False,
}]


@pytest.fixture
def client(monkeypatch):
    # The keys are read from the environment at import time; set them explicitly so the tests
    # don't depend on the machine they run on.
    monkeypatch.setattr(service, "PREDICT_API_KEY", PREDICT_KEY)
    monkeypatch.setattr(service, "TRAIN_API_KEY", TRAIN_KEY)
    # As a context manager so the startup hook (which loads the model) runs.
    with TestClient(service.app) as test_client:
        yield test_client


def test_health_reports_the_model_as_loaded(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


@pytest.mark.parametrize("path, body", [("/predict", ONE_EVENT), ("/train", {"events": ONE_EVENT, "labels": [1]})])
@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong"}])
def test_protected_endpoints_reject_a_missing_or_wrong_key(client, path, body, headers):
    assert client.post(path, json=body, headers=headers).status_code == 401


def test_protected_endpoints_refuse_everything_when_no_key_is_configured(client, monkeypatch):
    monkeypatch.setattr(service, "PREDICT_API_KEY", None)

    response = client.post("/predict", json=ONE_EVENT, headers={"X-API-Key": ""})

    assert response.status_code == 401


def test_predict_returns_one_probability_per_event(client):
    response = client.post("/predict", json=ONE_EVENT, headers={"X-API-Key": PREDICT_KEY})

    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 1
    assert 0.0 <= predictions[0] <= 1.0
