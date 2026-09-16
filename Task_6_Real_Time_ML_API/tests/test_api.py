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
        "tenure_months": 12,
        "support_tickets": 3,
        "monthly_spend_inr": 1499,
        "last_login_days": 5,
        "plan_type": "Premium"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "prediction_label" in data
    assert "probability_churn" in data
    assert "probability_not_churn" in data

    assert data["prediction"] in [0, 1]

    assert 0 <= data["probability_churn"] <= 1
    assert 0 <= data["probability_not_churn"] <= 1

    assert abs(
        data["probability_churn"]
        + data["probability_not_churn"]
        - 1
    ) < 0.001


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
