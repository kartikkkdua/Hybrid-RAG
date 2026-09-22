import os
import tempfile

import pytest

# Force the lightweight, deterministic backends so tests run anywhere.
os.environ.setdefault("EMBED_BACKEND", "tfidf")  # -> hashing fallback
os.environ.setdefault("RERANK_BACKEND", "lexical")
os.environ.setdefault("ANTHROPIC_API_KEY", "")


@pytest.fixture()
def service(tmp_path):
    from app.config import Settings
    from app.service import RAGService

    db = tmp_path / "test.db"
    settings = Settings(DB_PATH=str(db), EMBED_BACKEND="tfidf", RERANK_BACKEND="lexical")
    svc = RAGService(settings=settings)
    yield svc
    svc.close()
