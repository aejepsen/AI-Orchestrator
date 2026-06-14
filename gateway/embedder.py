"""Abstração de embeddings: SBERT (CPU) com fallback para Ollama."""

from __future__ import annotations

import logging
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class SBERTEmbedder:
    """Sentence-Transformers rodando em CPU."""

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None  # lazy
        self._dim = 384  # MiniLM-L12 default

    @property
    def dim(self) -> int:
        return self._dim

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name, cache_folder=self._cache_dir, device="cpu"
            )
            self._dim = self._model.get_embedding_dimension()
            logger.info(
                "SBERTEmbedder loaded: %s (dim=%d)", self._model_name, self._dim
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        self._ensure_model()
        embeddings = self._model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return [e.tolist() for e in embeddings]


class OllamaEmbedder:
    """Adapter: usa OllamaClient.embed() existente como fallback."""

    def __init__(self, llm: Any, model: str = "nomic-embed-text") -> None:
        self._llm = llm
        self._model = model
        self._dim = 768  # nomic-embed-text default

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._llm.embed(texts, model=self._model)
        if result and len(result[0]) != self._dim:
            self._dim = len(result[0])
        return result
