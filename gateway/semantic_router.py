"""Camada de roteamento semântico: golden set indexado no Qdrant.

Primeira camada do pipeline de classificação (antes do LLM): a pergunta é
embedada localmente (Ollama, modelo dedicado) e comparada por cosseno com os
exemplos rotulados do golden de roteamento. Aceite exige consenso: top-1 acima
do threshold E unanimidade dos vizinhos confiantes no conjunto
de domínios. Qualquer dúvida → None → o LLM classifier decide.

Falha de infraestrutura (Qdrant fora, embedding indisponível) nunca derruba a
request: loga warning e devolve None (degradação graceful para o LLM).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import httpx

from gateway.embedder import Embedder
from gateway.llm import LLMError
from gateway.router import RoutePlan

logger = logging.getLogger(__name__)

COLLECTION = "routing_examples"


def _point_id(question: str) -> str:
    """ID determinístico (UUID derivado do hash) — upsert idempotente."""
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


class SemanticRouter:
    """Busca kNN no Qdrant sobre o golden de roteamento."""

    def __init__(
        self,
        qdrant_url: str,
        embedder: Embedder,
        *,
        examples_path: str,
        threshold: float = 0.80,
        top_k: int = 5,
        min_score_gap: float = 0.05,
        client: httpx.Client | None = None,
    ) -> None:
        self._qdrant_url = qdrant_url.rstrip("/")
        self._embedder = embedder
        self._examples_path = examples_path
        self._threshold = threshold
        self._top_k = top_k
        self._min_score_gap = min_score_gap
        self._client = client or httpx.Client(timeout=10.0)
        self._ready = False

    # -- infraestrutura (lazy, idempotente) -----------------------------------

    def ensure_ready(self) -> None:
        """Cria a collection e indexa o golden set. Lazy: roda uma vez por processo."""
        if self._ready:
            return
        self._ensure_collection()
        self._seed_from_golden()
        self._ready = True

    def _ensure_collection(self) -> None:
        response = self._client.get(f"{self._qdrant_url}/collections/{COLLECTION}")
        if response.status_code == 200:
            # Verifica se a dimensão da collection existente bate com o embedder.
            try:
                info = response.json().get("result", {})
                existing_dim = (
                    info.get("config", {})
                    .get("params", {})
                    .get("vectors", {})
                    .get("size")
                )
                if existing_dim is not None and existing_dim != self._embedder.dim:
                    logger.warning(
                        "Collection %s com dim=%d, embedder dim=%d — recriando",
                        COLLECTION, existing_dim, self._embedder.dim,
                    )
                    self._client.delete(f"{self._qdrant_url}/collections/{COLLECTION}")
                else:
                    return
            except (KeyError, ValueError):
                return
        response = self._client.put(
            f"{self._qdrant_url}/collections/{COLLECTION}",
            json={"vectors": {"size": self._embedder.dim, "distance": "Cosine"}},
        )
        response.raise_for_status()

    def _seed_from_golden(self) -> None:
        path = Path(self._examples_path)
        if not path.exists():
            logger.warning("semantic_router: golden ausente em %s — índice vazio", path)
            return
        records = [
            record
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for record in [json.loads(line)]
            if not record.get("expect_clarification") and record.get("expect_domains")
        ]
        if not records:
            return
        vectors = self._embedder.embed([r["question"] for r in records])
        points = [
            {
                "id": _point_id(record["question"]),
                "vector": vector,
                "payload": {"question": record["question"], "domains": record["expect_domains"]},
            }
            for record, vector in zip(records, vectors)
        ]
        response = self._client.put(
            f"{self._qdrant_url}/collections/{COLLECTION}/points?wait=true",
            json={"points": points},
        )
        response.raise_for_status()
        logger.info("semantic_router: %d exemplos indexados em %s", len(points), COLLECTION)

    # -- roteamento ------------------------------------------------------------

    def route(self, question: str, *, exclude_question: str | None = None) -> RoutePlan | None:
        """Rota por similaridade ou None (sem consenso/infra fora → LLM decide).

        `exclude_question` remove um exemplo do resultado — usado pelo eval
        em leave-one-out para não casar consigo mesmo.
        """
        try:
            self.ensure_ready()
            vector = self._embedder.embed([question])[0]
            body: dict = {"vector": vector, "limit": self._top_k + 1, "with_payload": True}
            response = self._client.post(
                f"{self._qdrant_url}/collections/{COLLECTION}/points/search", json=body
            )
            response.raise_for_status()
            hits = response.json().get("result", [])
        except (httpx.HTTPError, LLMError, KeyError, ValueError) as exc:
            logger.warning("semantic_router indisponível (%s) — fallback para LLM", exc)
            return None

        if exclude_question is not None:
            excluded = _point_id(exclude_question)
            hits = [h for h in hits if h.get("id") != excluded]
        hits = hits[: self._top_k]

        confident = [h for h in hits if h.get("score", 0.0) >= self._threshold]
        if not confident or confident[0] is not hits[0]:
            return None

        top_score = hits[0].get("score", 0.0)
        top_domains = set(hits[0]["payload"]["domains"])

        # Score gap: se o melhor vizinho com domínios diferentes está próximo demais,
        # a query é ambígua — LLM decide. Evita matches onde single-domain casa com
        # multi-domain por proximidade lexical (ex: "SKU CAD-001" em estoque vs vendas).
        for h in hits[1:]:
            h_domains = set(h["payload"]["domains"])
            if h_domains != top_domains and (top_score - h.get("score", 0.0)) < self._min_score_gap:
                logger.debug(
                    "semantic_router: score gap insuficiente (%.3f vs %.3f) — fallback LLM",
                    top_score, h.get("score", 0.0),
                )
                return None

        # Consenso: todos os vizinhos confiantes devem concordar nos domínios.
        # Vizinhos com domínios heterogêneos indicam zona de fronteira entre rotas.
        if any(set(h["payload"]["domains"]) != top_domains for h in confident):
            return None

        score = hits[0]["score"]
        nearest = hits[0]["payload"]["question"]
        return RoutePlan(
            domains=sorted(top_domains),  # type: ignore[arg-type]
            plan=f'Roteado por similaridade semântica (score {score:.2f}) com "{nearest}".',
            clarification=None,
        )
