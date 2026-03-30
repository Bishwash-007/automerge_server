"""Request models for the automerge API."""

from pydantic import BaseModel, Field
from typing import Literal


class ConflictRequest(BaseModel):
    """Request model for single conflict resolution."""

    conflict_text: str = Field(
        ...,
        description="The merge conflict text including markers",
        examples=[
            "<<<<<<< HEAD\nconsole.log('hello');\n=======\nconsole.log('world');\n>>>>>>> feature-branch"
        ],
    )
    language: str = Field(
        ...,
        description="Programming language of the file",
        examples=["typescript", "python", "javascript"],
    )
    file_path: str | None = Field(
        default=None,
        description="Optional file path for RAG context",
        examples=["src/index.ts"],
    )
    provider: Literal["ollama", "huggingface"] = Field(
        default="ollama",
        description="LLM provider to use for resolution",
    )


class BatchConflictRequest(BaseModel):
    """Request model for batch conflict resolution."""

    conflicts: list[ConflictRequest] = Field(
        ...,
        description="List of conflicts to resolve",
        min_length=1,
    )
