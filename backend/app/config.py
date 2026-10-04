"""Central configuration."""
import os
from functools import lru_cache
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    # LLM providers
    google_api_key: str = ""
    groq_api_key: str = ""
    agent_models: str = "google_genai:gemini-3.5-flash-lite,google_genai:gemini-3.1-flash-lite,groq:qwen/qwen3.8-27b"
    fast_models: str = "google_genai:gemini-3.5-flash-lite,google_genai:gemini-3.1-flash-lite,groq:qwen/qwen3.8-27b"
    judge_model: str = "groq:openai/gpt-oss-120b"

    # Local models
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Storage
    database_url: str = f"sqlite:///{BACKEND_DIR / 'data' / 'novatech.db'}"
    qdrant_url: str = ""
    qdrant_path: str = str(BACKEND_DIR / "data" / "qdrant")
    qdrant_collection: str = "novatech_docs"
    checkpoint_db: str = str(BACKEND_DIR / "data" / "checkpoints.sqlite")
    docs_dir: Path = BACKEND_DIR / "data" / "docs"
    guard_model_dir: Path = BACKEND_DIR / "models" / "guard-lora"
    logs_dir: Path = BACKEND_DIR / "data" / "logs"

    # API: signs login tokens. Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
    app_secret: str = ""
    cors_origins: str = "http://localhost:5173"   # the Vite dev server (comma-separated list)

    telegram_bot_token: str = ""
    bot_secret: str = ""
    api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    """Cached so .env is read once per process."""
    return Settings()
