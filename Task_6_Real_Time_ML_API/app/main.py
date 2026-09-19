from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import PredictionRequest


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

@app.get("/model-info")
def model_info():
    feature_names = getattr(model, "feature_names_in_", None)

    if feature_names is not None:
        features = list(feature_names)
    else:
        features = []

    return {
        "model_type": type(model).__name__,
        "expected_features": features,
        "feature_count": len(features)
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_data = pd.DataFrame([request.features])

        expected_features = getattr(model, "feature_names_in_", None)

        if expected_features is not None:
            expected_features = list(expected_features)

            received_features = list(request.features.keys())

            missing_features = [
                feature
                for feature in expected_features
                if feature not in received_features
            ]

            unexpected_features = [
                feature
                for feature in received_features
                if feature not in expected_features
            ]

            if missing_features:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Missing required features",
                        "missing_features": missing_features
                    }
                )

            if unexpected_features:
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "Unexpected features",
                        "unexpected_features": unexpected_features
                    }
                )

            input_data = input_data[expected_features]

        prediction = model.predict(input_data)[0]

        probabilities = {}

        if hasattr(model, "predict_proba"):
            probability_values = model.predict_proba(input_data)[0]
            classes = getattr(model, "classes_", None)

            if classes is not None:
                probabilities = {
                    str(cls): float(probability)
                    for cls, probability in zip(
                        classes,
                        probability_values
                    )
                }

        return {
            "prediction": prediction.item()
            if hasattr(prediction, "item")
            else prediction,
            "probabilities": probabilities
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(exc)}"
        )