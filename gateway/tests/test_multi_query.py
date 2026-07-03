"""Testes do S5 — multi-query expansion no SemanticRouter (opt-in)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from gateway.query_enricher import expand_query_llm
from gateway.semantic_router import COLLECTION, SemanticRouter
from gateway.tests.test_semantic_router import FakeEmbedder

HIT_OK = [
    {"id": "a", "score": 0.95, "payload": {"question": "Quantas férias tenho?", "domains": ["rh"]}},
]


def _router(responses: list[list[dict]], expander, calls: list[str]) -> SemanticRouter:
    """Mock Qdrant devolvendo `responses` em sequência (1 por search)."""
    state = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/collections/{COLLECTION}":
            return httpx.Response(200, json={"result": {}})
        if request.url.path == f"/collections/{COLLECTION}/points/search":
            calls.append("search")
            hits = responses[min(state["i"], len(responses) - 1)]
            state["i"] += 1
            return httpx.Response(200, json={"result": [dict(h) for h in hits]})
        if request.url.path.startswith(f"/collections/{COLLECTION}/points"):
            return httpx.Response(200, json={"result": {}})
        return httpx.Response(404)

    return SemanticRouter(
        "http://qdrant.test",
        FakeEmbedder(),
        examples_path="/nao/existe.jsonl",
        top_k=2,
        query_expander=expander,
        multi_query_n=2,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_hit_primario_nao_expande() -> None:
    calls: list[str] = []
    expander_calls: list[str] = []

    def expander(q: str) -> list[str]:  # pragma: no cover — não deve rodar
        expander_calls.append(q)
        return []

    plan = _router([HIT_OK], expander, calls).route("quantos dias de férias tenho?")
    assert plan is not None and plan.domains == ["rh"]
    assert expander_calls == []


def test_miss_expande_e_variante_resolve() -> None:
    calls: list[str] = []
    plan = _router(
        [[], HIT_OK],  # 1ª busca vazia (miss) → variante acha consenso
        lambda q: ["qual o saldo de férias do funcionário?"],
        calls,
    ).route("e as férias?")
    assert plan is not None
    assert plan.domains == ["rh"]
    assert "multi-query" in plan.plan
    assert calls.count("search") == 2


def test_variantes_sem_consenso_devolve_none() -> None:
    plan = _router([[], []], lambda q: ["variante 1", "variante 2"], []).route("pergunta vaga")
    assert plan is None


def test_expander_quebrado_degrada_para_none() -> None:
    def broken(q: str) -> list[str]:
        raise RuntimeError("LLM fora")

    assert _router([[]], broken, []).route("pergunta") is None


def test_sem_expander_comportamento_original() -> None:
    assert _router([[]], None, []).route("pergunta") is None


def test_variante_igual_a_pergunta_descartada() -> None:
    calls: list[str] = []
    plan = _router([[], HIT_OK], lambda q: [q, "variante diferente"], calls).route("pergunta X")
    # 1ª busca (miss) + 1 busca da única variante válida.
    assert calls.count("search") == 2
    assert plan is not None


class _FakeLLM:
    def __init__(self, content: str) -> None:
        self._content = content
        self.calls: list[Any] = []

    def chat(self, messages: list[dict], *, format: Any = None) -> Any:
        self.calls.append(messages)

        class R:
            content = self._content

        return R()


def test_expand_query_llm_parseia_variants() -> None:
    llm = _FakeLLM(json.dumps({"variants": ["v1", "v2", "v3"]}))
    assert expand_query_llm(llm, "pergunta", n=2) == ["v1", "v2"]


def test_expand_query_llm_sem_variants_lanca() -> None:
    with pytest.raises(ValueError):
        expand_query_llm(_FakeLLM('{"outro": []}'), "pergunta", n=2)
