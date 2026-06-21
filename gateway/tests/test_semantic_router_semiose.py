"""Testes das melhorias Semiose no semantic router: S1 (contextual embeddings)
e S3 (cross-encoder reranker). Nenhum teste carrega modelos reais."""

from __future__ import annotations

import json
from typing import Any

import httpx

from gateway.semantic_router import (
    COLLECTION,
    SemanticRouter,
    _contextual_text,
    _point_id,
)


class FakeEmbedder:
    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self.embed_calls: list[list[str]] = []

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(list(texts))
        return [[0.1] * self._dim for _ in texts]


class FakeCrossEncoder:
    """Cross-encoder fake: pontua por correspondência com um termo-alvo."""

    def __init__(self, prefer: str) -> None:
        self._prefer = prefer
        self.calls: list[list[list[str]]] = []

    def predict(self, pairs: list[list[str]]) -> list[float]:
        self.calls.append(pairs)
        return [1.0 if self._prefer in b else 0.0 for _q, b in pairs]


def _qdrant_transport(hits: list[dict[str, Any]]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/collections/{COLLECTION}":
            return httpx.Response(200, json={"result": {}})
        if request.url.path == f"/collections/{COLLECTION}/points/search":
            return httpx.Response(200, json={"result": hits})
        if request.url.path.startswith(f"/collections/{COLLECTION}/points"):
            return httpx.Response(200, json={"result": {}})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _hit(question: str, domains: list[str], score: float) -> dict[str, Any]:
    return {"id": _point_id(question), "score": score, "payload": {"question": question, "domains": domains}}


# ── S1 — Contextual Embeddings ───────────────────────────────────────────


def test_contextual_text_prefixa_dominio():
    assert _contextual_text("qual a comissão?", ["vendas"]) == "[domínio: vendas] qual a comissão?"


def test_contextual_text_sem_dominio_retorna_original():
    assert _contextual_text("oi", []) == "oi"


def _seed_router(golden_path: str, embedder: FakeEmbedder, *, contextual: bool) -> SemanticRouter:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        return httpx.Response(200, json={"result": {}})

    return SemanticRouter(
        "http://qdrant.test",
        embedder,
        examples_path=golden_path,
        contextual_embeddings=contextual,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _write_golden(tmp_path) -> str:
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        json.dumps({"question": "comissão?", "expect_domains": ["vendas"], "expect_clarification": False}),
        encoding="utf-8",
    )
    return str(golden)


def test_s1_off_embeda_question_crua(tmp_path):
    emb = FakeEmbedder()
    router = _seed_router(_write_golden(tmp_path), emb, contextual=False)
    router.ensure_ready()
    assert emb.embed_calls[-1] == ["comissão?"]


def test_s1_on_embeda_question_prefixada(tmp_path):
    emb = FakeEmbedder()
    router = _seed_router(_write_golden(tmp_path), emb, contextual=True)
    router.ensure_ready()
    assert emb.embed_calls[-1] == ["[domínio: vendas] comissão?"]


# ── S3 — Cross-encoder como desempate do top-2 ambíguo ───────────────────

# Cenário ambíguo: top-2 com domínios diferentes e gap pequeno (0.02 < 0.05).
# Sem desempate, o consenso rejeita (None). O cross-encoder escolhe o vencedor.
_AMBIGUOUS_HITS = [
    _hit("salário do funcionário", ["rh"], 0.95),
    _hit("comissão do vendedor", ["vendas"], 0.93),
]


def _ce_router(ce: Any, *, hits: list[dict[str, Any]] | None = None, **kw: Any) -> SemanticRouter:
    return SemanticRouter(
        "http://qdrant.test",
        FakeEmbedder(),
        examples_path="/inexistente/golden.jsonl",
        threshold=0.80,
        top_k=3,
        cross_encoder=ce,
        client=httpx.Client(transport=_qdrant_transport(hits or _AMBIGUOUS_HITS)),
        **kw,
    )


def test_s3_desempata_para_vendas():
    ce = FakeCrossEncoder(prefer="comissão")
    router = _ce_router(ce, rerank_cross_encoder=True)
    plan = router.route("quanto a Juliana recebeu de comissão?")
    assert plan is not None
    assert plan.domains == ["vendas"]
    assert "desempate" in plan.plan
    assert ce.calls, "cross-encoder deveria ter sido chamado"


def test_s3_desempata_para_rh():
    ce = FakeCrossEncoder(prefer="salário")
    router = _ce_router(ce, rerank_cross_encoder=True)
    plan = router.route("qual o salário do Rafael?")
    assert plan is not None
    assert plan.domains == ["rh"]


def test_s3_desligado_caso_ambiguo_retorna_none():
    router = _ce_router(None, rerank_cross_encoder=False)
    assert router.route("pergunta na fronteira rh/vendas") is None


def test_s3_graceful_quando_cross_encoder_indisponivel():
    # rerank ligado mas cross_encoder=None e modelo inexistente → cai na lógica
    # normal de consenso, que rejeita o caso ambíguo. Sem exceção.
    router = SemanticRouter(
        "http://qdrant.test",
        FakeEmbedder(),
        examples_path="/inexistente/golden.jsonl",
        threshold=0.80,
        top_k=3,
        rerank_cross_encoder=True,
        cross_encoder_model="modelo/inexistente-xyz",
        client=httpx.Client(transport=_qdrant_transport(_AMBIGUOUS_HITS)),
    )
    assert router.route("pergunta na fronteira") is None


def test_s3_nao_atua_quando_gap_grande():
    # Gap grande (0.95 vs 0.70): não é ambíguo. CE não deve ser chamado;
    # como o segundo está abaixo do threshold, roteia normal para rh.
    hits = [
        _hit("salário do funcionário", ["rh"], 0.95),
        _hit("comissão do vendedor", ["vendas"], 0.70),
    ]
    ce = FakeCrossEncoder(prefer="comissão")
    router = _ce_router(ce, hits=hits, rerank_cross_encoder=True)
    plan = router.route("qual o salário?")
    assert plan is not None
    assert plan.domains == ["rh"]
    assert not ce.calls, "cross-encoder não deveria ser chamado em caso não-ambíguo"
