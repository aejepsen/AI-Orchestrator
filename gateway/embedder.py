"""Abstração de embeddings: SBERT (CPU) com fallback para Ollama."""

from __future__ import annotations

import logging
from typing import Any, Literal, Protocol

logger = logging.getLogger(__name__)

# Modelos E5 (intfloat/multilingual-e5-*) exigem prefixo no texto
# para diferenciar queries de documentos.
# Ref: https://huggingface.co/intfloat/multilingual-e5-large
E5_QUERY_PREFIX = "query: "
E5_PASSAGE_PREFIX = "passage: "


class Embedder(Protocol):
    @property
    def dim(self) -> int: ...
    def embed(self, texts: list[str], *, prefix_type: str = "document") -> list[list[float]]: ...


class SBERTEmbedder:
    """Sentence-Transformers rodando em CPU.
    
    Suporta modelos E5 (intfloat/multilingual-e5-*) com prefixo automático
    query:/passage: e detecção de dimensão no primeiro `embed()`.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        cache_dir: str | None = None,
    ) -> None:
        self._model_name = model_name
        self._cache_dir = cache_dir
        self._model: Any = None  # lazy
        self._dim: int = 0       # detectado do modelo no primeiro load

    @property
    def dim(self) -> int:
        if self._dim == 0:
            self._ensure_model()
        return self._dim

    def _ensure_model(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self._model_name, cache_folder=self._cache_dir, device="cpu"
            )
            self._dim = self._model.get_sentence_embedding_dimension()
            logger.info(
                "SBERTEmbedder loaded: %s (dim=%d)", self._model_name, self._dim
            )

    def _is_e5(self) -> bool:
        return "e5" in self._model_name.lower() and "cross-encoder" not in self._model_name.lower()

    def embed(self, texts: list[str], *, prefix_type: str = "document") -> list[list[float]]:
        self._ensure_model()
        if self._is_e5():
            prefix = E5_QUERY_PREFIX if prefix_type == "query" else E5_PASSAGE_PREFIX
            texts = [prefix + t for t in texts]
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

    def embed(self, texts: list[str], *, prefix_type: str = "document") -> list[list[float]]:
        result = self._llm.embed(texts, model=self._model)
        if result and len(result[0]) != self._dim:
            self._dim = len(result[0])
        return result
