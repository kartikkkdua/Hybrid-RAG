"""Central configuration, loaded from environment / .env.

Every knob that affects retrieval quality or cost lives here so the A/B harness
can vary a single object rather than reaching into modules.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    gen_model: str = Field(default="claude-sonnet-5", alias="GEN_MODEL")
    judge_model: str = Field(default="claude-sonnet-5", alias="JUDGE_MODEL")

    # --- Embeddings ---
    embed_backend: str = Field(default="auto", alias="EMBED_BACKEND")  # auto|st|tfidf
    embed_model: str = Field(default="BAAI/bge-small-en-v1.5", alias="EMBED_MODEL")

    # --- Rerank ---
    rerank_backend: str = Field(default="auto", alias="RERANK_BACKEND")  # auto|cross_encoder|lexical|none
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL")

    # --- Chunking ---
    chunk_tokens: int = Field(default=220, alias="CHUNK_TOKENS")
    chunk_overlap: int = Field(default=40, alias="CHUNK_OVERLAP")

    # --- Retrieval ---
    rrf_k: int = Field(default=60, alias="RRF_K")
    top_k_bm25: int = Field(default=40, alias="TOP_K_BM25")
    top_k_dense: int = Field(default=40, alias="TOP_K_DENSE")
    top_k_rerank: int = Field(default=8, alias="TOP_K_RERANK")

    # --- Storage ---
    # sqlite  : zero-infra default (FTS5 BM25 + NumPy cosine)
    # postgres: production backend (tsvector BM25 + pgvector HNSW)
    db_backend: str = Field(default="sqlite", alias="DB_BACKEND")
    db_path: str = Field(default="storage/corpus.db", alias="DB_PATH")
    pg_dsn: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/rag", alias="PG_DSN"
    )

    def resolved_db_path(self) -> Path:
        p = Path(self.db_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
