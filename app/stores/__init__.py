"""Storage backends. `build_store` picks one from settings."""
from __future__ import annotations

from ..config import Settings
from .base import Store
from .sqlite_store import SQLiteStore, build_fts_query

__all__ = ["Store", "SQLiteStore", "build_fts_query", "build_store"]


def build_store(settings: Settings, embed_dim: int = 0) -> Store:
    backend = (settings.db_backend or "sqlite").lower()
    if backend in ("postgres", "pg", "pgvector", "postgresql"):
        from .pg_store import PostgresStore

        return PostgresStore(settings.pg_dsn, embed_dim=embed_dim)
    return SQLiteStore(settings.resolved_db_path(), embed_dim=embed_dim)
