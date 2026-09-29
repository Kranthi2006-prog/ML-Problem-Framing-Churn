# Task 6 – Real-Time ML API

## Customer Churn Prediction API

This project deploys the champion machine learning model developed in Task 4 as a real-time REST API using FastAPI.

The API accepts customer information as JSON input and returns the predicted churn class along with prediction probabilities.

## Project Architecture

```text
Client
   |
   v
FastAPI REST API
   |
   v
Input Validation - Pydantic
   |
   v
Champion ML Pipeline
   |
   +-- Preprocessing
   |     +-- Missing Value Handling
   |     +-- Standard Scaling
   |     +-- One-Hot Encoding
   |
   +-- Trained Classification Model
   |
   v
Prediction + Probabilities
```

## Project Structure

```text
Task_6_Real_Time_ML_API/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── model/
│   └── champion_model.joblib
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── requirements.txt
└── README.md
```

## Technologies Used

* Python 3.10
* FastAPI
* Pydantic
* Uvicorn
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Joblib
* Pytest
* Docker

## API Endpoints

### GET `/`

Returns basic information about the API.

Example response:

```json
{
  "message": "Customer Churn Prediction API is running",
  "version": "1.0.0",
  "docs": "/docs"
}
```

### GET `/health`

Checks whether the API and machine learning model are available.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### GET `/model-info`

Returns information about the loaded model and its expected input features.

### POST `/predict`

Accepts customer features and returns the churn prediction and probabilities.

Example request:

```json
{
  "features": {
    "tenure_months": 12,
    "support_tickets": 3,
    "monthly_spend_inr": 1499,
    "last_login_days": 5,
    "plan_type": "Premium"
  }
}
```

Example response:

```json
{
  "prediction": 0,
  "probabilities": {
    "0": 0.85,
    "1": 0.15
  }
}
```

The exact probability values depend on the trained champion model.

## Running Locally

Open PowerShell inside the `Task_6_Real_Time_ML_API` directory.

Create and activate a virtual environment:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

## Running Tests

Run:

```powershell
pytest -v
```

The API test suite validates:

* Root endpoint
* Health endpoint
* Prediction endpoint
* Invalid input handling

## Docker

The project includes a Dockerfile for containerized deployment.

Build the Docker image:

```bash
docker build -t customer-churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 customer-churn-api
```

The API can then be accessed through:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## Machine Learning Model

The API uses the serialized champion model generated during Task 4:

```text
model/champion_model.joblib
```

The saved model contains the complete preprocessing and classification pipeline, allowing the API to receive raw customer features and perform preprocessing before prediction.

## Input Features

The model expects:

| Feature             | Description                                       |
| ------------------- | ------------------------------------------------- |
| `tenure_months`     | Number of months the customer has been subscribed |
| `support_tickets`   | Number of support tickets                         |
| `monthly_spend_inr` | Monthly customer spending in INR                  |
| `last_login_days`   | Days since the customer's last login              |
| `plan_type`         | Customer subscription plan                        |

## Error Handling

The API validates incoming features and returns HTTP `422` when:

* Required features are missing.
* Unexpected features are supplied.

Prediction failures are returned with HTTP `500`.

## Testing Result

The implemented API test suite contains four tests:

```text
test_root_endpoint
test_health_endpoint
test_prediction_endpoint
test_invalid_input
```

All four tests passed during local testing.

## Task 6 Outcome

This task demonstrates how a trained machine learning model can be converted into a real-time prediction service using FastAPI.

The solution includes:

* Serialized ML model
* REST API
* Input validation
* Prediction probabilities
* Health monitoring endpoint
* Model information endpoint
* Automated API tests
* Docker deployment configuration
* API documentation through Swagger UI
