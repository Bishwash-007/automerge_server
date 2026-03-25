"""FastAPI application for merge conflict resolution."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from models import (
    ConflictRequest,
    BatchConflictRequest,
    HealthResponse,
    ResolveResponse,
    BatchResolveResponse,
    ConflictResolution,
)
from services import OllamaClient, RagService
from vectorstore.retriever import VectorRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Automerge Server...")
    settings = get_settings()
    logger.info(f"Using model: {settings.ollama_model}")
    logger.info(f"ChromaDB persist dir: {settings.chroma_persist_dir}")
    yield
    # Shutdown
    logger.info("Shutting down Automerge Server...")


# Initialize FastAPI app
app = FastAPI(
    title="Automerge Server",
    description="AI-powered merge conflict resolution with RAG",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS for VS Code extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # VS Code extension runs locally
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
settings = get_settings()
ollama_client = OllamaClient()
retriever = VectorRetriever()
rag_service = RagService(ollama_client=ollama_client, retriever=retriever)


@app.get("/predictor/health/", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Check if the service is healthy."""
    ollama_status = await ollama_client.health_check()

    status = "healthy" if ollama_status.get("ollama_available") else "unhealthy"

    return HealthResponse(
        status=status,
        ollama_available=ollama_status.get("ollama_available", False),
        model_loaded=ollama_status.get("model_loaded"),
    )


@app.post("/predictor/resolve/", response_model=ResolveResponse)
async def resolve_conflict(request: ConflictRequest) -> ResolveResponse:
    """
    Resolve a single merge conflict.

    Args:
        request: ConflictRequest with conflict_text, language, and optional file_path

    Returns:
        ResolveResponse with resolved code and summary
    """
    try:
        result = await rag_service.resolve_conflict(
            conflict_text=request.conflict_text,
            language=request.language,
            file_path=request.file_path,
        )

        return ResolveResponse(
            result=result.resolved_code,
            summary=result.summary,
            confidence=result.confidence,
        )

    except Exception as e:
        logger.error(f"Conflict resolution failed: {e}")
        # Return fallback response
        return ResolveResponse(
            result=request.conflict_text,  # Return original on failure
            summary=f"Resolution failed: {str(e)}",
            confidence=0.0,
        )


@app.post("/predictor/resolve/batch/", response_model=BatchResolveResponse)
async def resolve_batch(request: BatchConflictRequest) -> BatchResolveResponse:
    """
    Resolve multiple merge conflicts.

    Args:
        request: BatchConflictRequest with list of conflicts

    Returns:
        BatchResolveResponse with list of resolutions
    """
    results = []

    for conflict in request.conflicts:
        try:
            result = await rag_service.resolve_conflict(
                conflict_text=conflict.conflict_text,
                language=conflict.language,
                file_path=conflict.file_path,
            )

            results.append(
                ConflictResolution(
                    result=result.resolved_code,
                    summary=result.summary,
                    confidence=result.confidence,
                )
            )

        except Exception as e:
            logger.error(f"Batch resolution failed for conflict: {e}")
            # Add fallback result for this conflict
            results.append(
                ConflictResolution(
                    result=conflict.conflict_text,
                    summary=f"Resolution failed: {str(e)}",
                    confidence=0.0,
                )
            )

    return BatchResolveResponse(results=results)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Automerge Server",
        "version": "0.1.0",
        "description": "AI-powered merge conflict resolution with RAG",
        "endpoints": {
            "health": "/predictor/health/",
            "resolve": "/predictor/resolve/",
            "batch_resolve": "/predictor/resolve/batch/",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
