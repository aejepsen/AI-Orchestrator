"""Testes do retrieval híbrido S2: fusão RRF (denso + BM25) no SemanticRouter.

O RRF só REORDENA o pool denso — o cosseno de cada hit é preservado para os
gates de aceitação. O ganho medível: demover do top_k um hit de domínio
divergente sem apoio lexical evita o veto do consenso (→ menos fallback LLM).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import httpx

from gateway.semantic_router import COLLECTION, SemanticRouter
from gateway.tests.test_semantic_router import FakeEmbedder

Q_RESERVA = "Reservar unidades de um produto"
Q_CONTAS = "Quais contas a pagar vencem hoje?"
Q_SALDO = "Qual o saldo do SKU ABC-123 no estoque?"

# Pool denso: o hit de finanças (domínio divergente) está colado no top-1 por
# cosseno; o exemplo lexicalmente idêntico à query vem em terceiro.
DENSE_HITS = [
    {"id": "x", "score": 0.95, "payload": {"question": Q_RESERVA, "domains": ["estoque"]}},
    {"id": "y", "score": 0.94, "payload": {"question": Q_CONTAS, "domains": ["financas"]}},
    {"id": "z", "score": 0.93, "payload": {"question": Q_SALDO, "domains": ["estoque"]}},
]

QUERY = "qual o saldo do sku ABC-123 no estoque?"


def _write_golden(tmp_path: Path) -> Path:
    records = [
        {"question": Q_RESERVA, "expect_domains": ["estoque"]},
        {"question": Q_CONTAS, "expect_domains": ["financas"]},
        {"question": Q_SALDO, "expect_domains": ["estoque"]},
    ]
    path = tmp_path / "golden_routing.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records))
    return path


def _transport(hits: list[dict[str, Any]], search_bodies: list[dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == f"/collections/{COLLECTION}":
            return httpx.Response(200, json={"result": {}})
        if request.url.path == f"/collections/{COLLECTION}/points/search":
            search_bodies.append(json.loads(request.content))
            return httpx.Response(200, json={"result": hits})
        if request.url.path.startswith(f"/collections/{COLLECTION}/points"):
            return httpx.Response(200, json={"result": {}})
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _router(tmp_path: Path, *, hybrid: bool, search_bodies: list[dict] | None = None) -> SemanticRouter:
    return SemanticRouter(
        "http://qdrant.test",
        FakeEmbedder(),
        examples_path=str(_write_golden(tmp_path)),
        top_k=2,
        hybrid_retrieval=hybrid,
        client=httpx.Client(
            transport=_transport(copy.deepcopy(DENSE_HITS), search_bodies if search_bodies is not None else [])
        ),
    )


def test_sem_hybrid_consenso_veta_e_cai_no_llm(tmp_path: Path) -> None:
    # top-2 denso = estoque + financas com gap 0.01 < min_score_gap → None.
    assert _router(tmp_path, hybrid=False).route(QUERY) is None


def test_hybrid_promove_match_lexical_e_roteia(tmp_path: Path) -> None:
    plan = _router(tmp_path, hybrid=True).route(QUERY)
    assert plan is not None
    assert plan.domains == ["estoque"]


def test_hybrid_preserva_cosseno_nos_gates_e_no_plan(tmp_path: Path) -> None:
    plan = _router(tmp_path, hybrid=True).route(QUERY)
    assert plan is not None
    assert "0.93" in plan.plan  # cosseno do hit promovido, não o RRF score


def test_hybrid_amplia_pool_denso(tmp_path: Path) -> None:
    bodies: list[dict] = []
    _router(tmp_path, hybrid=True, search_bodies=bodies).route(QUERY)
    assert bodies[0]["limit"] == 2 * 2 + 1

    bodies.clear()
    _router(tmp_path, hybrid=False, search_bodies=bodies).route(QUERY)
    assert bodies[0]["limit"] == 2 + 1


def test_golden_ausente_degrada_para_denso_puro(tmp_path: Path) -> None:
    router = SemanticRouter(
        "http://qdrant.test",
        FakeEmbedder(),
        examples_path=str(tmp_path / "inexistente.jsonl"),
        top_k=2,
        hybrid_retrieval=True,
        client=httpx.Client(transport=_transport([dict(h) for h in DENSE_HITS], [])),
    )
    # Sem índice BM25 o comportamento é o denso puro (aqui: veto do consenso).
    assert router.route(QUERY) is None
