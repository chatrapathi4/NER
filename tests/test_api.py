"""
test_api.py — Tests for FastAPI Endpoints

Uses FastAPI's TestClient (which wraps httpx) to test the API
without starting a real server.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from src.api.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_type" in data
        assert "version" in data


class TestPredictEndpoint:

    def test_valid_prediction(self, client):
        response = client.post(
            "/predict",
            json={"text": "Take Paracetamol 500 mg twice daily."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "entities" in data
        assert isinstance(data["entities"], list)
        assert "model_used" in data

    def test_entities_have_required_fields(self, client):
        response = client.post(
            "/predict",
            json={"text": "Amoxicillin 250 mg orally twice daily for 7 days."},
        )
        data = response.json()
        for entity in data["entities"]:
            assert "text" in entity
            assert "label" in entity
            assert entity["label"] in [
                "DRUG", "DOSAGE", "FREQUENCY", "DURATION", "ROUTE", "FORM"
            ]

    def test_empty_text_rejected(self, client):
        response = client.post(
            "/predict",
            json={"text": ""},
        )
        assert response.status_code == 422  # Validation error

    def test_missing_text_field(self, client):
        response = client.post(
            "/predict",
            json={},
        )
        assert response.status_code == 422

    def test_very_long_text(self, client):
        long_text = "Take Paracetamol 500 mg. " * 200  # 5000+ chars
        response = client.post(
            "/predict",
            json={"text": long_text},
        )
        # Should either succeed or return a validation error, not crash
        assert response.status_code in (200, 422)

    def test_special_characters(self, client):
        response = client.post(
            "/predict",
            json={"text": "Take medication — 500mg daily."},
        )
        assert response.status_code == 200

    def test_non_medical_text(self, client):
        response = client.post(
            "/predict",
            json={"text": "The weather is nice today."},
        )
        assert response.status_code == 200
        data = response.json()
        # Should return empty or few entities for non-medical text
        assert isinstance(data["entities"], list)
