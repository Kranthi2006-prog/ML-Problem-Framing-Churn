from typing import Any, Dict

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ...,
        description="Feature names and values required by the trained model."
    )
