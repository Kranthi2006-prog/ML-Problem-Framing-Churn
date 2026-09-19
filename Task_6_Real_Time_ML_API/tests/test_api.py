
import sys
from pathlib import Path

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent)
)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_prediction_endpoint():
    payload = {
        "features": {
            "tenure_months": 12,
            "support_tickets": 3,
            "monthly_spend_inr": 1499,
            "last_login_days": 5,
            "plan_type": "Premium"
        }
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "probabilities" in data


def test_invalid_input():
    payload = {
        "tenure_months": -5,
        "support_tickets": 3,
        "monthly_spend_inr": 1499,
        "last_login_days": 5,
        "plan_type": "Premium"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
