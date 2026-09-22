"""Pluggable embedding backends behind one interface.

- `SentenceTransformerEmbedder`: real semantic vectors (recommended; needs the
  optional `ml` extra). Used when EMBED_BACKEND=st or =auto and importable.
- `HashingEmbedder`: stateless sklearn HashingVectorizer, L2-normalised. Needs no
  model download and no corpus fit, so ingestion stays incremental and the whole
  pipeline / test-suite runs anywhere. Lexical, not semantic — the fallback.

All vectors are L2-normalised, so cosine similarity == dot product.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from .config import Settings


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _l2(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        norm = np.linalg.norm(mat) or 1.0
        return mat / norm
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import

        self._model = SentenceTransformer(model_name)
        self.name = f"st:{model_name}"
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                  show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class HashingEmbedder:
    """Deterministic lexical fallback — no model, no fit, incremental-safe."""

    def __init__(self, n_features: int = 1024):
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vec = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm=None, stop_words="english"
        )
        self.name = f"hashing:{n_features}"
        self.dim = n_features

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        mat = self._vec.transform(texts).toarray().astype(np.float32)
        return _l2(mat)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    backend = (settings.embed_backend or "auto").lower()
    if backend in ("st", "auto"):
        try:
            return SentenceTransformerEmbedder(settings.embed_model)
        except Exception:
            if backend == "st":
                raise
            # fall through to hashing
    return HashingEmbedder()
