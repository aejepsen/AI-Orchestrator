"""Testes do streaming token-a-token: chat_stream (NDJSON) e nó synthesize."""

from __future__ import annotations

import json
import threading
from typing import Any

import httpx
import pytest

from gateway.graph import GatewayGraph
from gateway.llm import ChatResponse, LLMError, OllamaClient


def _client(lines: list[dict], status: int = 200) -> OllamaClient:
    body = "\n".join(json.dumps(line) for line in lines)

    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(status, text=body)

    return OllamaClient(
        "http://ollama.test",
        "qwen3.5-9b-orch",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class TestChatStream:
    def test_tokens_na_ordem_e_content_completo(self) -> None:
        lines = [
            {"message": {"content": "Olá"}, "done": False},
            {"message": {"content": ", mundo"}, "done": False},
            {"message": {"content": ""}, "done": True},
        ]
        tokens: list[str] = []
        response = _client(lines).chat_stream(
            [{"role": "user", "content": "oi"}], on_token=tokens.append
        )
        assert tokens == ["Olá", ", mundo"]
        assert response.content == "Olá, mundo"
        assert response.tool_calls == ()

    def test_thinking_descartado(self) -> None:
        lines = [
            {"message": {"thinking": "hmm", "content": ""}, "done": False},
            {"message": {"content": "resposta"}, "done": True},
        ]
        tokens: list[str] = []
        response = _client(lines).chat_stream(
            [{"role": "user", "content": "oi"}], on_token=tokens.append
        )
        assert tokens == ["resposta"]
        assert response.content == "resposta"

    def test_erro_http_vira_llmerror(self) -> None:
        with pytest.raises(LLMError):
            _client([], status=500).chat_stream(
                [{"role": "user", "content": "oi"}], on_token=lambda t: None
            )

    def test_chunk_invalido_vira_llmerror(self) -> None:
        client = OllamaClient(
            "http://o.test",
            "m",
            client=httpx.Client(
                transport=httpx.MockTransport(lambda r: httpx.Response(200, text="não é json\n"))
            ),
        )
        with pytest.raises(LLMError):
            client.chat_stream([{"role": "user", "content": "oi"}], on_token=lambda t: None)


class _StreamOnlyLLM:
    """chat_stream OK; chat() proibido — garante que o caminho streaming foi usado."""

    def chat_stream(self, messages: list[dict], *, trace: Any = None, on_token) -> ChatResponse:
        for token in ("Sim: ", "tudo certo."):
            on_token(token)
        return ChatResponse(content="Sim: tudo certo.")

    def chat(self, *args: Any, **kwargs: Any) -> ChatResponse:  # pragma: no cover
        raise AssertionError("chat() não deveria ser chamado com on_token registrado")


class _BlockingLLM:
    """chat() OK; chat_stream proibido — caminho sem on_token."""

    def chat(self, messages: list[dict], *, trace: Any = None) -> ChatResponse:
        return ChatResponse(content="Resposta síncrona.")

    def chat_stream(self, *args: Any, **kwargs: Any) -> ChatResponse:  # pragma: no cover
        raise AssertionError("chat_stream() não deveria ser chamado sem on_token")


def _graph_stub(llm: Any, on_token) -> GatewayGraph:
    graph = GatewayGraph.__new__(GatewayGraph)
    graph._local = threading.local()
    graph._local.on_token = on_token
    graph._local.trace = None
    graph._llm = llm
    return graph


def _multi_domain_state() -> dict:
    return {
        "question": "posso aceitar o pedido?",
        "sanitized": "posso aceitar o pedido?",
        "trace_id": "t-1",
        "route": {"domains": ["vendas", "estoque"], "plan": "p"},
        "agent_results": {
            "vendas": {"answer": "Desconto OK."},
            "estoque": {"answer": "Há saldo."},
        },
        "history": [],
    }


class TestSynthesizeStreaming:
    def test_multi_dominio_streama_e_final_igual_ao_acumulado(self) -> None:
        tokens: list[str] = []
        graph = _graph_stub(_StreamOnlyLLM(), on_token=tokens.append)
        update = graph._synthesize(_multi_domain_state())
        assert tokens == ["Sim: ", "tudo certo."]
        assert update["final_answer"] == "Sim: tudo certo."

    def test_sem_on_token_usa_chat_sincrono(self) -> None:
        graph = _graph_stub(_BlockingLLM(), on_token=None)
        update = graph._synthesize(_multi_domain_state())
        assert update["final_answer"] == "Resposta síncrona."

    def test_single_domain_nao_chama_llm(self) -> None:
        tokens: list[str] = []
        graph = _graph_stub(_BlockingLLM(), on_token=tokens.append)
        state = _multi_domain_state()
        state["route"] = {"domains": ["vendas"], "plan": "p"}
        update = graph._synthesize(state)
        assert update["final_answer"] == "Desconto OK."
        assert tokens == []
