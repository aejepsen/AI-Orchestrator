"""Detector OOD por resíduo de subespaço (SVD do golden de roteamento).

Terceiro sinal de segurança do gateway, LOG-ONLY: regex (`flag_injection`) pega
padrões conhecidos, BERTimbau pega injection semântica, e o resíduo de
subespaço pega o que os dois não veem — queries FORA da distribuição de uso
(assunto alheio, payloads bizarros, ruído). A query é projetada no subespaço
gerado pelas perguntas legítimas do golden (base via SVD truncada); a norma do
resíduo mede a distância a esse subespaço.

Limite honesto (parecer CliffordNet §2.3, verificado numericamente): injection
FRASEADA como query válida vive dentro do subespaço e passa com resíduo ≈ 0 —
este sinal NÃO substitui o BERTimbau nem o system prompt. Ele complementa.

Custo: uma SVD no primeiro uso (lazy) + um produto matriz-vetor por query
(O(k·d), sub-ms em CPU). numpy puro, zero dependência nova.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Rank truncado: base full-rank torna o span permissivo demais (quase toda
# query projeta dentro). Calibrado no golden 153 + 30 queries OOD sintéticas
# (evals/eval_ood_guard.py, 2026-07-02): energy 0.99 deu a melhor separação
# (AUC 0.937 vs 0.925 a 0.90); threshold operacional 0.60 ≈ P95 in-dist →
# ~24/30 OOD flagados com ~5% de flag em tráfego legítimo (ok: log-only).
DEFAULT_ENERGY = 0.99
DEFAULT_MAX_RANK = 120
_MIN_FIT_VECTORS = 8


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


class SubspaceGuard:
    """Base ortonormal do golden (SVD truncada) + resíduo de projeção."""

    def __init__(self, *, energy: float = DEFAULT_ENERGY, max_rank: int = DEFAULT_MAX_RANK) -> None:
        self._energy = energy
        self._max_rank = max_rank
        self._basis: np.ndarray | None = None  # d×k

    @property
    def ready(self) -> bool:
        return self._basis is not None

    @property
    def rank(self) -> int:
        return 0 if self._basis is None else int(self._basis.shape[1])

    def fit(self, vectors: np.ndarray) -> None:
        """Constrói a base do subespaço. Poucos vetores → guard fica inativo."""
        matrix = np.asarray(vectors, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] < _MIN_FIT_VECTORS:
            logger.warning(
                "SubspaceGuard: %s vetores é insuficiente para fit (mínimo %d) — inativo",
                0 if matrix.ndim != 2 else matrix.shape[0],
                _MIN_FIT_VECTORS,
            )
            return
        matrix = _normalize_rows(matrix)
        _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
        total = float(np.sum(singular**2))
        if total == 0.0:
            return
        cumulative = np.cumsum(singular**2) / total
        k = int(np.searchsorted(cumulative, self._energy) + 1)
        k = min(k, self._max_rank, len(singular))
        self._basis = vt[:k].T
        logger.info("SubspaceGuard: base ajustada (n=%d, rank=%d)", matrix.shape[0], k)

    def score(self, vector: np.ndarray) -> float:
        """Norma do resíduo da projeção (0 = dentro do subespaço; 1 = ortogonal)."""
        if self._basis is None:
            raise RuntimeError("SubspaceGuard.score() antes do fit()")
        q = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm == 0.0:
            return 0.0
        q = q / norm
        projection = self._basis @ (self._basis.T @ q)
        return float(np.linalg.norm(q - projection))


class OODGuard:
    """Wrapper operacional: embeda o golden no primeiro uso (lazy) e pontua queries.

    Falha de infraestrutura (embedder fora, golden ausente) nunca propaga:
    `score()` devolve None e o guard se desativa — degradação graceful, mesmo
    padrão do restante do gateway.
    """

    def __init__(
        self,
        embedder: Any,
        *,
        examples_path: str,
        threshold: float,
        energy: float = DEFAULT_ENERGY,
        max_rank: int = DEFAULT_MAX_RANK,
    ) -> None:
        self._embedder = embedder
        self._examples_path = examples_path
        self.threshold = threshold
        self._guard = SubspaceGuard(energy=energy, max_rank=max_rank)
        self._failed = False

    def _ensure_fitted(self) -> bool:
        if self._guard.ready:
            return True
        if self._failed:
            return False
        try:
            path = Path(self._examples_path)
            questions = [
                json.loads(line)["question"]
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            vectors = self._embedder.embed(questions, prefix_type="document")
            self._guard.fit(np.asarray(vectors))
        except Exception as exc:  # noqa: BLE001 — sinal opcional, nunca derruba request
            logger.warning("OODGuard indisponível (%s) — sinal desativado", exc)
        if not self._guard.ready:
            self._failed = True
        return self._guard.ready

    def score(self, question: str) -> float | None:
        """Resíduo OOD da pergunta, ou None se o guard estiver indisponível."""
        if not self._ensure_fitted():
            return None
        try:
            vector = self._embedder.embed([question], prefix_type="query")[0]
            return round(self._guard.score(np.asarray(vector)), 4)
        except Exception as exc:  # noqa: BLE001
            logger.warning("OODGuard.score falhou (%s) — sinal ignorado neste request", exc)
            return None
