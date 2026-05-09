"""
API request/response schemas.

Pydantic models for API validation.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class OptimizeRequest(BaseModel):
    """Request for molecule optimization."""

    protein_sequence: str = Field(..., min_length=5)
    seeds: List[str] = Field(..., min_items=1)
    objectives: Dict[str, float] = Field(...)
    max_iterations: int = Field(default=50, ge=1, le=500)
    batch_size: int = Field(default=50, ge=1, le=100)


class OptimizeResponse(BaseModel):
    """Response from molecule optimization."""

    best_molecule: Optional[Dict] = None
    top_5: List[Dict] = []
    pareto_front: List[Dict] = []
    total_iterations: int = 0
    execution_time_seconds: float = 0.0