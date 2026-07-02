"""Camada de roteamento semântico: golden set indexado no Qdrant.

Primeira camada do pipeline de classificação (antes do LLM): a pergunta é
embedada localmente (Ollama, modelo dedicado) e comparada por cosseno com os
exemplos rotulados do golden de roteamento. Aceite exige consenso: top-1 acima
do threshold E unanimidade dos vizinhos confiantes no conjunto
de domínios. Qualquer dúvida → None → o LLM classifier decide.

Semiose — Camada C (re-ranking contextual):
  Nível 1 (Harness): boost aditivo determinístico para hits cujo domínio
  coincide com o context_domain do turno anterior. Preserva _raw_score
  para tracing. Sem LLM, zero latência extra.

Falha de infraestrutura (Qdrant fora, embedding indisponível) nunca derruba a
request: loga warning e devolve None (degradação graceful para o LLM).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import httpx

from gateway.bm25 import BM25Index
from gateway.embedder import Embedder
from gateway.llm import LLMError
from gateway.router import RoutePlan

logger = logging.getLogger(__name__)

COLLECTION = "routing_examples"

# Semiose — S2: constante k da fusão RRF (Cormack et al., 2009). 60 é o valor
# canônico: amortece a diferença entre ranks altos sem anular o topo.
RRF_K = 60

# Semiose — Camada C: boost aditivo para re-ranking contextual.
# Valor calibrado para threshold 0.80-0.92: desempata matches bons,
# não resgata matches ruins. Cap em 1.0 preserva semântica de cosseno.
CONTEXT_BOOST = 0.05

# Semiose — Camada C Nível 2 (S3): modelo cross-encoder padrão (multilíngue).
DEFAULT_CROSS_ENCODER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


def _point_id(question: str) -> str:
    """ID determinístico (UUID derivado do hash) — upsert idempotente."""
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def _contextual_text(question: str, domains: list[str]) -> str:
    """Semiose — Camada A+ (S1): situa o exemplo no seu domínio antes de embedar.

    Espelha Contextual Embeddings (Anthropic, 2024): anexar o contexto que dá
    sentido ao trecho reduz falhas de recuperação. Aplicado só no corpus (índice),
    não na query — assimetria intencional, como na técnica original.
    """
    if not domains:
        return question
    return f"[domínio: {', '.join(domains)}] {question}"


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
        context_boost: float = CONTEXT_BOOST,
        contextual_embeddings: bool = False,
        rerank_cross_encoder: bool = False,
        cross_encoder_model: str = DEFAULT_CROSS_ENCODER,
        cross_encoder: Any = None,
        hybrid_retrieval: bool = False,
        rrf_k: int = RRF_K,
        client: httpx.Client | None = None,
        api_key: str | None = None,
    ) -> None:
        self._qdrant_url = qdrant_url.rstrip("/")
        self._embedder = embedder
        self._examples_path = examples_path
        self._threshold = threshold
        self._top_k = top_k
        self._min_score_gap = min_score_gap
        self._context_boost = context_boost
        self._contextual_embeddings = contextual_embeddings
        self._rerank_cross_encoder = rerank_cross_encoder
        self._cross_encoder_model = cross_encoder_model
        # Cross-encoder injetável (testes) ou lazy-load. _ce_loaded evita reimport em falha.
        self._cross_encoder = cross_encoder
        self._ce_loaded = cross_encoder is not None
        # S2 — retrieval híbrido: índice BM25 construído junto do seed do golden.
        self._hybrid_retrieval = hybrid_retrieval
        self._rrf_k = rrf_k
        self._bm25: BM25Index | None = None
        self._bm25_questions: list[str] = []
        headers = {"api-key": api_key} if api_key else {}
        self._client = client or httpx.Client(timeout=10.0, headers=headers)
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
        # S1 — Contextual Embeddings: embeda o exemplo prefixado com seu domínio
        # (quando habilitado), mas mantém a question original no payload e no id.
        if self._contextual_embeddings:
            texts = [_contextual_text(r["question"], r["expect_domains"]) for r in records]
        else:
            texts = [r["question"] for r in records]
        # S2 — Contextual BM25: mesmo corpus (e mesmo prefixo contextual, quando
        # habilitado) da metade densa. Índice em memória, nada vai ao Qdrant.
        if self._hybrid_retrieval:
            self._bm25 = BM25Index(texts)
            self._bm25_questions = [r["question"] for r in records]
        vectors = self._embedder.embed(texts, prefix_type="document")
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

    def route(
        self,
        question: str,
        *,
        exclude_question: str | None = None,
        context_domain: str | None = None,
    ) -> RoutePlan | None:
        """Rota por similaridade ou None (sem consenso/infra fora → LLM decide).

        `exclude_question` remove um exemplo do resultado — usado pelo eval
        em leave-one-out para não casar consigo mesmo.

        `context_domain` (Semiose — Camada C): domínio do turno anterior.
        Aplica boost aditivo determinístico nos hits cujo domínio coincide,
        desempatando matches bons sem resgatar matches ruins.
        """
        try:
            self.ensure_ready()
            vector = self._embedder.embed([question], prefix_type="query")[0]
            # S2 — híbrido: pool denso ampliado (2×top_k) para a fusão RRF ter
            # candidatos a promover/demover antes do corte final em top_k.
            limit = self._top_k * 2 + 1 if self._bm25 is not None else self._top_k + 1
            body: dict = {"vector": vector, "limit": limit, "with_payload": True}
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
        if self._bm25 is not None and len(hits) > 1:
            hits = self._rrf_fuse(question, hits, exclude_question)
        hits = hits[: self._top_k]

        # Semiose — Camada C: re-ranking contextual (Nível 1 — Harness).
        # Preserva _raw_score para tracing; boost aditivo com cap em 1.0.
        # Gap-gated: só aplica boost quando top-1 e top-2 estão próximos
        # (gap ≤ 0.03 — cenário genuinamente ambíguo onde contexto desempata).
        if context_domain:
            if len(hits) >= 2:
                gap = hits[0].get("score", 0.0) - hits[1].get("score", 0.0)
            else:
                gap = 0.0
            if gap <= 0.03:
                for h in hits:
                    h["_raw_score"] = h.get("score", 0.0)
                    if context_domain in h.get("payload", {}).get("domains", []):
                        h["score"] = min(h["_raw_score"] + self._context_boost, 1.0)
                hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
                if any(h.get("_raw_score") != h.get("score") for h in hits):
                    logger.debug(
                        "semantic_router: context rerank applied (context_domain=%s, gap=%.4f)",
                        context_domain, gap,
                    )

        # Semiose — Camada C Nível 2 (S3): cross-encoder como desempate.
        # Só atua no caso ambíguo (top-2 com domínios diferentes e gap pequeno)
        # que a lógica de consenso rejeitaria. Substitui o "Nível 2 LLM" planejado.
        # Opt-in e graceful (modelo/lib ausente → cai na lógica normal → None).
        if self._rerank_cross_encoder and len(hits) >= 2:
            decided = self._cross_encoder_decide(hits, question)
            if decided is not None:
                return decided

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

        raw_score = hits[0].get("_raw_score", top_score)
        nearest = hits[0]["payload"]["question"]
        plan_detail = f'Roteado por similaridade semântica (score {raw_score:.2f}'
        if context_domain and raw_score != top_score:
            plan_detail += f", boosted {top_score:.2f} via contexto {context_domain}"
        plan_detail += f') com "{nearest}".'
        return RoutePlan(
            domains=sorted(top_domains),  # type: ignore[arg-type]
            plan=plan_detail,
            clarification=None,
        )

    # -- retrieval híbrido (S2): fusão RRF denso + BM25 ------------------------

    def _rrf_fuse(self, question: str, hits: list[dict], exclude_question: str | None) -> list[dict]:
        """Reordena o pool denso por Reciprocal Rank Fusion (denso + BM25).

        Só REORDENA: o cosseno de cada hit é preservado para os gates de
        aceitação (threshold, score gap, consenso), como no S3 — RRF score é
        rank-based e não é comparável ao threshold de cosseno. Candidatos
        apenas-lexicais não entram: sem cosseno, não passariam no gate. O ganho
        vem da composição do top_k (demover um hit de domínio divergente sem
        apoio lexical evita o veto do consenso → menos fallback pro LLM).
        """
        assert self._bm25 is not None
        bm25_rank: dict[str, int] = {}
        for position, doc_index in enumerate(self._bm25.rank(question, limit=len(hits) * 2)):
            doc_question = self._bm25_questions[doc_index]
            if doc_question != exclude_question:
                bm25_rank[doc_question] = len(bm25_rank)
            if len(bm25_rank) >= len(hits):
                break

        def fused_score(dense_rank: int, hit: dict) -> float:
            score = 1.0 / (self._rrf_k + dense_rank + 1)
            lexical = bm25_rank.get(hit.get("payload", {}).get("question", ""))
            if lexical is not None:
                score += 1.0 / (self._rrf_k + lexical + 1)
            return score

        ranked = sorted(
            enumerate(hits), key=lambda pair: fused_score(pair[0], pair[1]), reverse=True
        )
        for dense_rank, hit in ranked:
            hit["_rrf_score"] = round(fused_score(dense_rank, hit), 6)
        if [hit for _, hit in ranked] != hits:
            logger.debug("semantic_router: RRF reordenou o pool denso (hybrid retrieval)")
        return [hit for _, hit in ranked]

    # -- re-ranking cross-encoder (S3, lazy + graceful) ------------------------

    def _ensure_cross_encoder(self) -> Any | None:
        """Lazy-load do CrossEncoder. Nunca bloqueia se a lib/modelo faltar."""
        if self._cross_encoder is not None or self._ce_loaded:
            return self._cross_encoder
        self._ce_loaded = True
        try:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(self._cross_encoder_model, device="cpu")
            logger.info("CrossEncoder carregado: %s", self._cross_encoder_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrossEncoder indisponível (%s) — rerank cross-encoder ignorado", exc)
            self._cross_encoder = None
        return self._cross_encoder

    def _cross_encoder_decide(self, hits: list[dict], question: str) -> RoutePlan | None:
        """Desempata o top-2 ambíguo com cross-encoder; senão devolve None.

        "Ambíguo" = os dois melhores hits têm domínios diferentes e diferença de
        cosseno < min_score_gap — exatamente o caso que o consenso rejeitaria. Se
        o cross-encoder resolver com confiança (vencedor acima do threshold),
        roteia para ele. Em qualquer outra situação devolve None e deixa a lógica
        normal seguir (clareza, indisponibilidade do modelo, baixa confiança).
        """
        a, b = hits[0], hits[1]
        da = set(a.get("payload", {}).get("domains", []))
        db = set(b.get("payload", {}).get("domains", []))
        if not da or da == db:
            return None
        if abs(a.get("score", 0.0) - b.get("score", 0.0)) >= self._min_score_gap:
            return None
        ce = self._ensure_cross_encoder()
        if ce is None:
            return None
        pairs = [
            [question, a.get("payload", {}).get("question", "")],
            [question, b.get("payload", {}).get("question", "")],
        ]
        try:
            sa, sb = ce.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CrossEncoder.predict falhou (%s) — desempate ignorado", exc)
            return None
        winner = a if float(sa) >= float(sb) else b
        if winner.get("score", 0.0) < self._threshold:
            return None
        w_domains = set(winner["payload"]["domains"])
        nearest = winner["payload"]["question"]
        logger.debug("semantic_router: cross-encoder desempatou para %s", sorted(w_domains))
        return RoutePlan(
            domains=sorted(w_domains),  # type: ignore[arg-type]
            plan=(
                f'Roteado por desempate cross-encoder (cosseno {winner.get("score", 0.0):.2f}) '
                f'com "{nearest}".'
            ),
            clarification=None,
        )
