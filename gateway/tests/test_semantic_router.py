"""Testes do semantic router: consenso, threshold, degradação e guards."""

from __future__ import annotations

import json
from typing import Any

import httpx

from gateway.llm import LLMError
from gateway.router import classify_intent
from gateway.semantic_router import COLLECTION, SemanticRouter, _point_id


class FakeEmbedLLM:
    """LLM fake só com embed(); chat() não deve ser chamado nestes testes."""

    def __init__(self, fail: bool = False) -> None:
        self._fail = fail
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if self._fail:
            raise LLMError("embedding fora")
        self.embed_calls.append(texts)
        return [[0.1] * 768 for _ in texts]

    def chat(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise AssertionError("chat() não deveria ser chamado quando o semantic resolve")


def _qdrant_transport(hits: list[dict[str, Any]]) -> httpx.MockTransport:
    """Mock do Qdrant: collection existe; search devolve `hits`."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/collections/{COLLECTION}":
            return httpx.Response(200, json={"result": {}})
        if request.url.path == f"/collections/{COLLECTION}/points/search":
            return httpx.Response(200, json={"result": hits})
        if request.url.path.startswith(f"/collections/{COLLECTION}/points"):
            return httpx.Response(200, json={"result": {}})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _router(hits: list[dict[str, Any]], *, llm: Any | None = None, threshold: float = 0.80) -> SemanticRouter:
    return SemanticRouter(
        "http://qdrant.test",
        llm or FakeEmbedLLM(),
        embed_model="fake-embed",
        examples_path="/inexistente/golden.jsonl",  # seed vira no-op com warning
        threshold=threshold,
        top_k=3,
        client=httpx.Client(transport=_qdrant_transport(hits)),
    )


def _hit(question: str, domains: list[str], score: float) -> dict[str, Any]:
    return {"id": _point_id(question), "score": score, "payload": {"question": question, "domains": domains}}


def test_aceita_top1_acima_do_threshold_com_consenso():
    router = _router(
        [
            _hit("qual a comissão do vendedor?", ["vendas"], 0.93),
            _hit("comissão deste mês?", ["vendas"], 0.88),
            _hit("desconto máximo?", ["vendas"], 0.82),
        ]
    )
    plan = router.route("me mostre a comissão da Juliana")
    assert plan is not None
    assert plan.domains == ["vendas"]
    assert "similaridade" in plan.plan


def test_rejeita_top1_abaixo_do_threshold():
    router = _router([_hit("qual a comissão?", ["vendas"], 0.61)])
    assert router.route("pergunta vaga qualquer") is None


def test_rejeita_vizinhos_divergentes():
    router = _router(
        [
            _hit("comissão do vendedor", ["vendas"], 0.86),
            _hit("salário do funcionário", ["rh"], 0.85),
            _hit("férias acumuladas", ["rh"], 0.84),
        ]
    )
    assert router.route("pergunta na fronteira rh/vendas") is None


def test_degrada_para_none_com_embedding_fora():
    router = _router([], llm=FakeEmbedLLM(fail=True))
    assert router.route("qualquer pergunta") is None


def test_leave_one_out_exclui_self_match():
    question = "qual a comissão do vendedor?"
    router = _router([_hit(question, ["vendas"], 1.0)])
    assert router.route(question, exclude_question=question) is None


def test_classify_intent_usa_semantic_sem_chamar_llm():
    class StubSemantic:
        def route(self, question: str, **kwargs: Any):
            from gateway.router import RoutePlan

            return RoutePlan(domains=["vendas"], plan="semântico", clarification=None)

    plan = classify_intent("comissão da Juliana", llm=FakeEmbedLLM(), semantic=StubSemantic())
    assert plan.domains == ["vendas"]


def test_guard_aplicado_na_saida_semantica():
    class StubSemantic:
        def route(self, question: str, **kwargs: Any):
            from gateway.router import RoutePlan

            return RoutePlan(domains=["rh", "vendas"], plan="semântico", clarification=None)

    plan = classify_intent(
        "qual a comissão do funcionário Rafael?", llm=FakeEmbedLLM(), semantic=StubSemantic()
    )
    assert plan.domains == ["vendas"]


def test_seed_indexa_golden(tmp_path):
    golden = tmp_path / "golden.jsonl"
    golden.write_text(
        "\n".join(
            [
                json.dumps({"question": "comissão?", "expect_domains": ["vendas"], "expect_clarification": False}),
                json.dumps({"question": "tempo amanhã?", "expect_domains": [], "expect_clarification": True}),
            ]
        ),
        encoding="utf-8",
    )
    upserts: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(404)
        if request.method == "PUT" and request.url.path.endswith("/points"):
            upserts.append(json.loads(request.content))
            return httpx.Response(200, json={"result": {}})
        return httpx.Response(200, json={"result": {}})

    router = SemanticRouter(
        "http://qdrant.test",
        FakeEmbedLLM(),
        embed_model="fake-embed",
        examples_path=str(golden),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    router.ensure_ready()
    assert len(upserts) == 1
    points = upserts[0]["points"]
    assert len(points) == 1  # clarification fica de fora
    assert points[0]["payload"]["domains"] == ["vendas"]
