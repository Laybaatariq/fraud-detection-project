import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_probability():
    sample_transaction = {
        "transaction_type": "TRANSFER",
        "amount": 5000,
        "oldbalanceOrg": 10000,
        "newbalanceOrig": 5000,
        "oldbalanceDest": 2000,
        "newbalanceDest": 7000,
        "hour_of_day": 14,
        "step": 1
    }
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200
    data = response.json()
    assert "fraud_probability" in data
    assert 0 <= data["fraud_probability"] <= 1
    assert data["prediction"] in ["Fraud", "Legit"]


def test_predict_missing_field_returns_error():
    incomplete_transaction = {
        "transaction_type": "TRANSFER",
        "amount": 5000
    }
    response = client.post("/predict", json=incomplete_transaction)
    assert response.status_code == 422


def test_explain_returns_top_features():
    sample_transaction = {
        "transaction_type": "CASH_OUT",
        "amount": 9500,
        "oldbalanceOrg": 9500,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 9500,
        "hour_of_day": 3,
        "step": 1
    }
    response = client.post("/explain", json=sample_transaction)
    assert response.status_code == 200
    data = response.json()
    assert "explanation" in data
    assert "top_features" in data
    assert len(data["top_features"]) > 0
