"""Testes do detector OOD por resíduo de subespaço (log-only)."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from gateway.graph import GatewayGraph
from gateway.subspace_guard import OODGuard, SubspaceGuard


def _cluster(n: int, dim: int = 32, seed: int = 7) -> np.ndarray:
    """Vetores concentrados num subespaço de 5 dims (distribuição 'legítima')."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(5, dim))
    return rng.normal(size=(n, 5)) @ base


class TestSubspaceGuard:
    def test_query_dentro_do_span_tem_residuo_baixo(self) -> None:
        vectors = _cluster(50)
        guard = SubspaceGuard()
        guard.fit(vectors)
        in_dist = vectors[0] * 3.0  # mesma direção, escala diferente
        assert guard.score(in_dist) < 0.15

    def test_query_ortogonal_tem_residuo_alto(self) -> None:
        vectors = _cluster(50, dim=32)
        guard = SubspaceGuard()
        guard.fit(vectors)
        # Vetor fora do subespaço de 5 dims: componente ortogonal via Gram-Schmidt.
        rng = np.random.default_rng(99)
        q = rng.normal(size=32)
        basis = guard._basis
        q_orth = q - basis @ (basis.T @ q)
        assert guard.score(q_orth) > 0.9

    def test_rank_respeita_energia_e_teto(self) -> None:
        guard = SubspaceGuard(energy=0.90, max_rank=3)
        guard.fit(_cluster(50))
        assert 1 <= guard.rank <= 3

    def test_poucos_vetores_fica_inativo(self) -> None:
        guard = SubspaceGuard()
        guard.fit(_cluster(4))
        assert not guard.ready
        with pytest.raises(RuntimeError):
            guard.score(np.ones(32))

    def test_vetor_zero_score_zero(self) -> None:
        guard = SubspaceGuard()
        guard.fit(_cluster(50))
        assert guard.score(np.zeros(32)) == 0.0


class _FakeEmbedder:
    def __init__(self, dim: int = 32) -> None:
        self._dim = dim
        self._rng = np.random.default_rng(3)
        self._base = self._rng.normal(size=(5, dim))

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str], *, prefix_type: str = "document") -> list[list[float]]:
        # Determinístico por hash do texto, dentro do subespaço de 5 dims.
        out = []
        for text in texts:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            out.append((rng.normal(size=5) @ self._base).tolist())
        return out


def _golden(tmp_path: Path, n: int = 20) -> Path:
    path = tmp_path / "golden.jsonl"
    lines = [json.dumps({"question": f"pergunta legítima {i}", "expect_domains": ["rh"]}) for i in range(n)]
    path.write_text("\n".join(lines))
    return path


class TestOODGuard:
    def test_score_lazy_fit_e_valor(self, tmp_path: Path) -> None:
        guard = OODGuard(_FakeEmbedder(), examples_path=str(_golden(tmp_path)), threshold=0.5)
        score = guard.score("pergunta legítima 3")
        assert score is not None and 0.0 <= score <= 1.0

    def test_golden_ausente_degrada_para_none(self, tmp_path: Path) -> None:
        guard = OODGuard(_FakeEmbedder(), examples_path=str(tmp_path / "nao_existe.jsonl"), threshold=0.5)
        assert guard.score("qualquer") is None
        assert guard.score("qualquer") is None  # segunda chamada não re-tenta

    def test_embedder_quebrado_degrada_para_none(self, tmp_path: Path) -> None:
        class _Broken:
            def embed(self, texts, *, prefix_type="document"):
                raise RuntimeError("embedder fora")

        guard = OODGuard(_Broken(), examples_path=str(_golden(tmp_path)), threshold=0.5)
        assert guard.score("qualquer") is None


class TestSanitizeIntegration:
    def _graph_stub(self, guard) -> GatewayGraph:
        graph = GatewayGraph.__new__(GatewayGraph)
        graph._local = threading.local()
        graph._local.trace = None
        graph._injection_detector = None
        graph._ood_guard = guard
        return graph

    def test_residual_entra_no_state(self) -> None:
        class _StubGuard:
            threshold = 0.85

            def score(self, question: str) -> float:
                return 0.42

        update = self._graph_stub(_StubGuard())._sanitize({"question": "quantas férias tenho?", "trace_id": "t"})
        assert update["_ood_residual"] == 0.42

    def test_guard_indisponivel_nao_seta_residual(self) -> None:
        class _NoneGuard:
            threshold = 0.85

            def score(self, question: str) -> None:
                return None

        update = self._graph_stub(_NoneGuard())._sanitize({"question": "oi", "trace_id": "t"})
        assert "_ood_residual" not in update

    def test_sem_guard_sanitize_intacto(self) -> None:
        update = self._graph_stub(None)._sanitize({"question": "oi", "trace_id": "t"})
        assert "_ood_residual" not in update
        assert update["sanitized"]
