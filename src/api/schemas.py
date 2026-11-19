"""
schemas.py — Pydantic Models for API Request/Response Validation

Pydantic models provide:
    1. Automatic request validation (reject malformed input)
    2. API documentation generation (Swagger/OpenAPI)
    3. Type safety (IDE autocompletion, static analysis)
    4. Serialization (Python objects → JSON response)
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """
    Request body for the /predict endpoint.

    Attributes:
        text: The medical/prescription text to analyze.
    """
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Medical or prescription text to extract entities from.",
        json_schema_extra={
            "examples": ["Take Paracetamol 500 mg twice daily for 5 days."]
        },
    )


class EntityResponse(BaseModel):
    """
    A single extracted entity.

    Attributes:
        text:  The entity text as it appears in the input.
        label: The entity type (DRUG, DOSAGE, FREQUENCY, DURATION, ROUTE, FORM).
        start: Character start offset in the original text.
        end:   Character end offset in the original text.
    """
    text: str = Field(..., description="Entity text")
    label: str = Field(..., description="Entity type (DRUG, DOSAGE, etc.)")
    start: int = Field(0, description="Start character offset")
    end: int = Field(0, description="End character offset")


class PredictionResponse(BaseModel):
    """
    Response body for the /predict endpoint.

    Attributes:
        text:       The original input text.
        entities:   List of extracted entities.
        model_used: Which model backend was used.
    """
    text: str = Field(..., description="Original input text")
    entities: List[EntityResponse] = Field(
        default_factory=list,
        description="List of extracted medical entities",
    )
    model_used: str = Field(
        "rule_based",
        description="Model backend used for prediction",
    )


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""
    status: str = Field("healthy", description="Service status")
    model_type: str = Field(..., description="Active model type")
    version: str = Field("1.0.0", description="API version")
