"""Configuration module for the automerge server."""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Hugging Face Configuration
    hf_model: str = "ankit-ml11/automerge-codet5"
    hf_max_length: int = 512
    hf_num_beams: int = 5

    # RAG Configuration
    chroma_persist_dir: str = "./chroma_db"
    embedding_model: str = "all-MiniLM-L6-v2"

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080

    # Git History Indexing
    git_history_depth: int = 500

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()