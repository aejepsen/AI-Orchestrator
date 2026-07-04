"""Testes do loop de tool-calling — LLM e registry mockados (zero rede)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from gateway.agents import build_system_prompt, run_domain_agent
from gateway.llm import ChatResponse, ToolCall
from gateway.tools.registry import ToolNotFound


class FakeLLM:
    """Devolve respostas roteirizadas e grava as mensagens recebidas."""

    def __init__(self, responses: list[ChatResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]], *, tools: Any = None, temperature: float = 0.0, trace: Any = None) -> ChatResponse:
        self.calls.append([dict(m) for m in messages])
        return self._responses.pop(0)


class FakeRegistry:
    """Executor roteirizado por nome de tool; tools_for devolve um stub."""

    def __init__(self, results: dict[str, dict[str, Any]]) -> None:
        self._results = results
        self.executed: list[tuple[str, str, dict[str, Any]]] = []

    def tools_for(self, domain: str) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": name}} for name in self._results]

    def execute(self, domain: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.executed.append((domain, name, args))
        if name not in self._results:
            raise ToolNotFound(domain, name, sorted(self._results))
        return self._results[name]


def test_single_tool_call_then_final_answer() -> None:
    llm = FakeLLM(
        [
            ChatResponse(content="", tool_calls=(ToolCall("get_product", {"sku": "MON-27P-003"}),)),
            ChatResponse(content="O disponível do MON-27P-003 é 35 unidades."),
        ]
    )
    registry = FakeRegistry({"get_product": {"status": 200, "body": {"sku": "MON-27P-003", "available": 35}}})

    result = run_domain_agent("estoque", "Qual o disponível do MON-27P-003?", registry=registry, llm=llm)

    assert result.final_answer == "O disponível do MON-27P-003 é 35 unidades."
    assert result.stop_reason == "answer"
    assert result.iters == 2
    assert [(t.name, t.status) for t in result.tool_trace] == [("get_product", 200)]
    assert registry.executed == [("estoque", "get_product", {"sku": "MON-27P-003"})]
    # resultado da tool reinjetado como role=tool na segunda chamada ao LLM
    tool_messages = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"])["body"]["available"] == 35


def test_422_result_reinjected_as_observation() -> None:
    error_body = {"error": "insufficient_stock", "detail": "Disponível: 4.", "rule": "reserva limitada ao disponível"}
    llm = FakeLLM(
        [
            ChatResponse(content="", tool_calls=(ToolCall("create_reservation", {"sku": "NTB-DEV-004", "quantity": 50}),)),
            ChatResponse(content="Não foi possível reservar: disponível de apenas 4 unidades."),
        ]
    )
    registry = FakeRegistry({"create_reservation": {"status": 422, "body": error_body}})

    result = run_domain_agent("estoque", "Reserve 50 do NTB-DEV-004.", registry=registry, llm=llm)

    assert result.tool_trace[0].status == 422
    assert "4 unidades" in result.final_answer
    tool_messages = [m for m in llm.calls[1] if m["role"] == "tool"]
    observation = json.loads(tool_messages[0]["content"])
    assert observation["status"] == 422
    assert observation["body"]["rule"] == "reserva limitada ao disponível"


def test_hallucinated_tool_becomes_observation_not_crash() -> None:
    llm = FakeLLM(
        [
            ChatResponse(content="", tool_calls=(ToolCall("drop_database", {}),)),
            ChatResponse(content="Essa operação não está disponível."),
        ]
    )
    registry = FakeRegistry({"list_products": {"status": 200, "body": []}})

    result = run_domain_agent("estoque", "Apague tudo.", registry=registry, llm=llm)

    assert result.tool_trace[0].name == "drop_database"
    assert result.tool_trace[0].status == 0
    tool_messages = [m for m in llm.calls[1] if m["role"] == "tool"]
    assert "unknown_tool" in tool_messages[0]["content"]


def test_max_iters_stops_loop() -> None:
    looping = ChatResponse(content="", tool_calls=(ToolCall("list_products", {}),))
    llm = FakeLLM([looping, looping, looping])
    registry = FakeRegistry({"list_products": {"status": 200, "body": []}})

    result = run_domain_agent("estoque", "Liste em loop.", registry=registry, llm=llm, max_iters=3)

    assert result.stop_reason == "max_iters"
    assert result.iters == 3
    assert len(result.tool_trace) == 3
    assert "Não foi possível concluir" in result.final_answer


def test_multiple_tool_calls_in_one_iteration() -> None:
    llm = FakeLLM(
        [
            ChatResponse(
                content="",
                tool_calls=(ToolCall("get_product", {"sku": "A"}), ToolCall("get_product", {"sku": "B"})),
            ),
            ChatResponse(content="Comparados."),
        ]
    )
    registry = FakeRegistry({"get_product": {"status": 200, "body": {}}})

    result = run_domain_agent("estoque", "Compare A e B.", registry=registry, llm=llm)

    assert len(result.tool_trace) == 2
    assert result.iters == 2


def test_system_prompt_scoped_and_dated() -> None:
    prompt = build_system_prompt("rh")
    assert "RH" in prompt
    assert "Hoje é 20" in prompt  # data ISO presente
    assert "nunca invente" in prompt


@pytest.mark.parametrize("domain", ["financas", "rh", "estoque", "vendas"])
def test_system_prompt_has_label_for_every_domain(domain: str) -> None:
    prompt = build_system_prompt(domain)
    assert "agente especialista" in prompt
    assert "ferramentas fornecidas" in prompt
