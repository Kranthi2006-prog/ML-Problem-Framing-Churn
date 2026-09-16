from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import CustomerInput


BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "model" / "champion_model.joblib"

model = joblib.load(MODEL_PATH)


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Real-time customer churn prediction using the Task 4 champion model.",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "Customer Churn Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/predict")
def predict(customer: CustomerInput):

    input_data = pd.DataFrame([{
        "tenure_months": customer.tenure_months,
        "support_tickets": customer.support_tickets,
        "monthly_spend_inr": customer.monthly_spend_inr,
        "last_login_days": customer.last_login_days,
        "plan_type": customer.plan_type
    }])

    prediction = model.predict(input_data)[0]

    probabilities = model.predict_proba(input_data)[0]

    probability_not_churn = float(probabilities[0])
    probability_churn = float(probabilities[1])

    prediction_label = (
        "Churned"
        if prediction == 1
        else "Not Churned"
    )

    return {
        "prediction": int(prediction),
        "prediction_label": prediction_label,
        "probability_churn": probability_churn,
        "probability_not_churn": probability_not_churn
    }
