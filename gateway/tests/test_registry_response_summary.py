"""Testes do resumo de resposta na description das tools (resíduo estoque-03).

O modelo julga a capacidade de uma tool apenas pela description — o schema
de resposta não entra no tool schema do Ollama. `_response_summary` anexa os
campos da resposta 2xx à description para o modelo saber o que a tool devolve.
"""

import json
from pathlib import Path

from gateway.tools.registry import _response_summary, parse_openapi

FIXTURE = Path(__file__).parent / "fixtures" / "financas_openapi.json"


def _spec() -> dict:
    return json.loads(FIXTURE.read_text())


class TestResponseSummary:
    def test_description_inclui_campos_de_resposta(self) -> None:
        tools = parse_openapi(_spec())
        description = tools["get_account"].description
        assert "Resposta:" in description
        assert "status" in description
        assert "amount" in description

    def test_lista_indica_array_de_itens(self) -> None:
        tools = parse_openapi(_spec())
        description = tools["list_accounts"].description
        assert "Resposta: lista de itens com" in description

    def test_enum_expande_valores(self) -> None:
        tools = parse_openapi(_spec())
        # Account.status é Literal["aberta", "paga", "recebida"] no serviço.
        assert "aberta" in tools["get_account"].description

    def test_operation_sem_schema_nao_anexa_resumo(self) -> None:
        operation = {"responses": {"200": {"description": "ok"}}}
        assert _response_summary(operation, {}) is None

    def test_summary_original_preservado(self) -> None:
        tools = parse_openapi(_spec())
        description = tools["create_account"].description
        assert description.startswith("Cria conta a pagar ou a receber")
