"""Testes do módulo embedder: protocol, SBERT com mock, OllamaEmbedder adapter."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from gateway.embedder import Embedder, OllamaEmbedder, SBERTEmbedder


class FakeEmbedder:
    """Implementação fake do protocol Embedder para testes."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self.calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1] * self._dim for _ in texts]


def test_fake_embedder_satisfaz_protocol():
    emb: Embedder = FakeEmbedder()
    assert emb.dim == 384
    vecs = emb.embed(["hello"])
    assert len(vecs) == 1
    assert len(vecs[0]) == 384


def test_sbert_embedder_lazy_load():
    emb = SBERTEmbedder(model_name="test-model", cache_dir="/tmp/models")
    assert emb.dim == 384
    assert emb._model is None  # não carregou ainda


def test_sbert_embedder_embed_com_mock():
    """Testa SBERTEmbedder injetando modelo fake diretamente (sem importar sentence_transformers)."""

    class FakeRow:
        """Simula elemento do array retornado por encode() — precisa de .tolist()."""

        def __init__(self, values: list[float]) -> None:
            self._values = values

        def tolist(self) -> list[float]:
            return self._values

    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 384
    fake_model.encode.return_value = [FakeRow([0.5] * 384), FakeRow([0.3] * 384)]

    emb = SBERTEmbedder(model_name="test-model", cache_dir="/tmp/cache")
    # Injeta modelo diretamente, bypassing _ensure_model / import.
    emb._model = fake_model
    emb._dim = fake_model.get_sentence_embedding_dimension()

    result = emb.embed(["a", "b"])

    fake_model.encode.assert_called_once_with(
        ["a", "b"], convert_to_numpy=True, normalize_embeddings=True
    )
    assert len(result) == 2
    assert len(result[0]) == 384
    assert result[0][0] == 0.5
    assert emb.dim == 384


def test_ollama_embedder_adapter():
    fake_llm = MagicMock()
    fake_llm.embed.return_value = [[0.2] * 768, [0.4] * 768]

    emb = OllamaEmbedder(fake_llm, model="nomic-embed-text")
    assert emb.dim == 768

    result = emb.embed(["hello", "world"])
    fake_llm.embed.assert_called_once_with(["hello", "world"], model="nomic-embed-text")
    assert len(result) == 2
    assert len(result[0]) == 768


def test_ollama_embedder_atualiza_dim():
    fake_llm = MagicMock()
    fake_llm.embed.return_value = [[0.1] * 1024]

    emb = OllamaEmbedder(fake_llm, model="custom-model")
    assert emb.dim == 768  # default

    emb.embed(["test"])
    assert emb.dim == 1024  # atualizado após primeira chamada
