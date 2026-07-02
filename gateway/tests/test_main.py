"""Testes do endpoint SSE (grafo mockado, transporte ASGI em memória)."""

from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest

from gateway.main import create_app


@pytest.fixture(autouse=True)
def _allow_open_access(monkeypatch):
    """Testes unitários rodam sem ACCESS_TOKEN; habilita modo dev."""
    monkeypatch.setenv("ALLOW_OPEN_ACCESS", "1")


class FakeGraph:
    """Reproduz a interface stream(question, trace_id=, on_agent=) do GatewayGraph."""

    def __init__(self, *, clarification: bool = False, boom: bool = False) -> None:
        self._clarification = clarification
        self._boom = boom

    def stream(self, question: str, *, history=None, trace_id: str | None = None, on_agent=None, thread_id: str | None = None, on_confirm=None, on_token=None):
        if self._boom:
            raise RuntimeError("grafo explodiu")
        yield {"sanitize": {"sanitized": question}}
        if self._clarification:
            yield {"classify": {"route": {"domains": [], "plan": "", "clarification": "Qual área?"}}}
            yield {"respond_clarification": {"final_answer": "Qual área?"}}
            return
        route = {"domains": ["vendas", "estoque"], "plan": "p", "clarification": None}
        yield {"classify": {"route": route}}
        yield {"confirm_dispatch": {}}
        if on_agent:
            on_agent("vendas", "Desconto ok.")
            on_agent("estoque", "Saldo ok.")
        yield {"dispatch": {"agent_results": {}}}
        yield {"synthesize": {"final_answer": "Sim, pode aceitar."}}


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


def _post_chat(graph: FakeGraph, question: str) -> tuple[int, str, list[tuple[str, dict]]]:
    app = create_app(graph_factory=lambda: graph)

    async def call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/chat", json={"question": question})
            return response.status_code, response.headers.get("content-type", ""), response.text

    status, content_type, body = asyncio.run(call())
    return status, content_type, _parse_sse(body)


def test_health():
    app = create_app(graph_factory=lambda: FakeGraph())

    async def call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.get("/health")

    response = asyncio.run(call())
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_sse_multi_dominio():
    status, content_type, events = _post_chat(FakeGraph(), "posso aceitar o pedido?")

    assert status == 200
    assert content_type.startswith("text/event-stream")
    names = [name for name, _ in events]
    assert names == ["route", "agent", "agent", "final"]

    route = events[0][1]
    assert route["domains"] == ["vendas", "estoque"]

    agents = {data["domain"]: data["answer"] for name, data in events if name == "agent"}
    assert agents == {"vendas": "Desconto ok.", "estoque": "Saldo ok."}

    final = events[-1][1]
    assert final["answer"] == "Sim, pode aceitar."
    assert final["trace_id"]


def test_chat_sse_clarification():
    _, _, events = _post_chat(FakeGraph(clarification=True), "me ajuda aí")

    names = [name for name, _ in events]
    assert names == ["route", "final"]
    assert events[0][1]["clarification"] == "Qual área?"
    assert events[1][1]["answer"] == "Qual área?"


def test_chat_erro_vira_evento_sse():
    _, _, events = _post_chat(FakeGraph(boom=True), "qualquer coisa")

    assert events[0][0] == "error"
    # Stack trace NÃO é exposto ao cliente; mensagem genérica.
    assert events[0][1]["detail"] == "Erro interno. Tente novamente."
    assert "trace_id" in events[0][1]


def test_chat_question_vazia_e_422():
    app = create_app(graph_factory=lambda: FakeGraph())

    async def call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/chat", json={"question": ""})

    response = asyncio.run(call())
    assert response.status_code == 422
