"""Configuration module for the automerge server."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:235b-cloud"

    # RAG Configuration
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080

    # Git History Indexing
    git_history_depth: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Allow other env vars to exist without error
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
