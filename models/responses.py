"""Response models for the automerge API."""

from pydantic import BaseModel, Field


class ConflictResolution(BaseModel):
    """Resolution result for a single conflict."""

    result: str = Field(
        ...,
        description="The resolved code without conflict markers",
    )
    summary: str = Field(
        ...,
        description="Explanation of why the conflict was resolved this way",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score of the resolution (0-1)",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., examples=["healthy", "unhealthy"])
    ollama_available: bool = Field(default=False)
    model_loaded: str | None = Field(default=None)


class ResolveResponse(BaseModel):
    """Response for single conflict resolution."""

    result: str
    summary: str
    confidence: float = 0.0


class BatchResolveResponse(BaseModel):
    """Response for batch conflict resolution."""

    results: list[ConflictResolution] = Field(
        ...,
        description="List of resolution results in the same order as input conflicts",
    )
