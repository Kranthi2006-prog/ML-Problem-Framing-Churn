from pydantic import BaseModel, Field


class CustomerInput(BaseModel):
    tenure_months: float = Field(..., ge=0)
    support_tickets: float = Field(..., ge=0)
    monthly_spend_inr: float = Field(..., ge=0)
    last_login_days: float = Field(..., ge=0)
    plan_type: str
