"""Testes do classificador de intenção (LLM mockado, sem rede)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import ValidationError

from gateway.llm import ChatResponse
from gateway.router import RoutePlan, classify_intent, lexical_route


class FakeLLM:
    """Devolve respostas pré-programadas e grava as chamadas."""

    def __init__(self, *contents: str) -> None:
        self._contents = list(contents)
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages, *, tools=None, temperature=0.0, format=None) -> ChatResponse:
        self.calls.append({"messages": list(messages), "format": format})
        if not self._contents:
            raise AssertionError("FakeLLM sem respostas restantes")
        return ChatResponse(content=self._contents.pop(0))


def _plan(domains: list[str], plan: str = "ok", clarification: str | None = None) -> str:
    return json.dumps({"domains": domains, "plan": plan, "clarification": clarification})


# -- RoutePlan -----------------------------------------------------------------


def test_routeplan_dedup_preserva_ordem():
    route = RoutePlan(domains=["vendas", "estoque", "vendas"], plan="x")
    assert route.domains == ["vendas", "estoque"]


def test_routeplan_clarification_esvazia_domains():
    route = RoutePlan(domains=["rh"], plan="", clarification="qual o assunto?")
    assert route.domains == []


def test_routeplan_vazio_sem_clarification_invalido():
    with pytest.raises(ValidationError):
        RoutePlan(domains=[], plan="")


def test_routeplan_dominio_desconhecido_invalido():
    with pytest.raises(ValidationError):
        RoutePlan(domains=["juridico"], plan="x")


# -- classify_intent -----------------------------------------------------------


def test_json_valido_primeira_tentativa():
    llm = FakeLLM(_plan(["rh"]))
    route = classify_intent("férias do Carlos", llm)
    assert route.domains == ["rh"]
    assert route.clarification is None
    assert len(llm.calls) == 1
    assert llm.calls[0]["format"] == "json"


def test_json_com_fence_markdown_e_tolerado():
    llm = FakeLLM(f"```json\n{_plan(['estoque'])}\n```")
    assert classify_intent("saldo do SKU", llm).domains == ["estoque"]


def test_json_com_texto_ao_redor_e_tolerado():
    llm = FakeLLM(f"Claro! {_plan(['vendas'])} Espero ter ajudado.")
    assert classify_intent("desconto no pedido", llm).domains == ["vendas"]


def test_multi_dominio():
    llm = FakeLLM(_plan(["vendas", "estoque", "financas"]))
    route = classify_intent("posso aceitar pedido de 500 unidades com 15% de desconto?", llm)
    assert route.domains == ["vendas", "estoque", "financas"]


def test_clarification_e_rota_valida():
    llm = FakeLLM(_plan([], plan="", clarification="Sobre qual domínio?"))
    route = classify_intent("me ajuda aí", llm)
    assert route.domains == []
    assert route.clarification == "Sobre qual domínio?"


def test_invalido_gera_retry_com_erro_reinjetado():
    llm = FakeLLM("não sou JSON", _plan(["financas"]))
    route = classify_intent("contas a pagar", llm)
    assert route.domains == ["financas"]
    assert len(llm.calls) == 2
    retry_msgs = llm.calls[1]["messages"]
    assert any("inválida" in m["content"] for m in retry_msgs if m["role"] == "user")


def test_duas_falhas_caem_no_fallback_lexico():
    llm = FakeLLM("lixo", "mais lixo")
    route = classify_intent("quantos dias de férias o Pedro tem?", llm)
    assert route.domains == ["rh"]
    assert len(llm.calls) == 2


# -- fallback léxico -----------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("saldo de férias do funcionário", ["rh"]),
        ("reposição do sku ABC no estoque", ["estoque"]),
        ("comissão sobre a venda do pedido 9", ["vendas"]),
        ("fluxo de caixa e contas a pagar", ["financas"]),
    ],
)
def test_lexical_route_por_dominio(question, expected):
    assert lexical_route(question).domains == expected


def test_lexical_route_normaliza_acentos():
    assert lexical_route("FÉRIAS do colaborador").domains == ["rh"]


def test_lexical_route_multi_dominio():
    route = lexical_route("o pedido de venda cabe no estoque e no caixa?")
    assert set(route.domains) == {"vendas", "estoque", "financas"}


def test_lexical_route_sem_match_pede_esclarecimento():
    route = lexical_route("qual a previsão do tempo amanhã?")
    assert route.domains == []
    assert route.clarification
