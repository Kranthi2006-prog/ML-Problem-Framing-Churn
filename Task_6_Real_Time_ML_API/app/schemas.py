from pydantic import BaseModel, Field
from typing import Any, Dict

class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Feature names and values required by the trained model."
    )


class CustomerInput(BaseModel):
    tenure_months: float = Field(..., ge=0)
    support_tickets: float = Field(..., ge=0)
    monthly_spend_inr: float = Field(..., ge=0)
    last_login_days: float = Field(..., ge=0)
    plan_type: str
