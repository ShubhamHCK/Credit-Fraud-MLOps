from fastapi.testclient import TestClient
import numpy as np
import pytest
from backend.main import app, models

class FakeModel:
    def __init__(self, probability=0.2):
        self.probability = probability
    def predict_proba(self, features):
        return np.array([[1-self.probability, self.probability]])

@pytest.fixture(autouse=True)
def inject_models():
    old = models.copy()
    models["credit"] = FakeModel(0.2)
    models["fraud"] = FakeModel(0.8)
    yield
    models.update(old)

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_credit_prediction(client):
    response = client.post("/api/credit/predict", json={
        "annual_income": 80000, "loan_amount": 20000,
        "credit_score": 720, "debt_to_income": 0.25,
        "employment_years": 6, "previous_defaults": 0
    })
    assert response.status_code == 200
    assert response.json()["risk_band"] == "Low Risk"

def test_fraud_prediction(client):
    response = client.post("/api/fraud/predict", json={
        "amount": 2500, "transaction_hour": 2,
        "transactions_last_24h": 12, "distance_from_usual_km": 500,
        "is_international": 1, "card_not_present": 1,
        "merchant_risk_score": 0.9, "account_age_days": 20
    })
    assert response.status_code == 200
    assert response.json()["decision"] == "Flag for investigation"

def test_invalid_credit_input(client):
    response = client.post("/api/credit/predict", json={
        "annual_income": -10, "loan_amount": 20000,
        "credit_score": 720, "debt_to_income": 0.25,
        "employment_years": 6, "previous_defaults": 0
    })
    assert response.status_code == 422
