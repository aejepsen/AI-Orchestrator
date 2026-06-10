"""Testes do tool registry: parse de OpenAPI real (fixture) + executor HTTP mockado."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from gateway.tools.registry import ToolNotFound, ToolRegistry, parse_openapi

FIXTURE = Path(__file__).parent / "fixtures" / "financas_openapi.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def tools(spec: dict) -> dict:
    return parse_openapi(spec)


class TestParseOpenapi:
    def test_health_excluded(self, tools: dict) -> None:
        assert "health_check" not in tools

    def test_all_business_operations_present(self, tools: dict) -> None:
        assert set(tools) == {
            "list_accounts",
            "create_account",
            "get_account",
            "pay_account",
            "receive_account",
            "get_cashflow",
        }

    def test_query_params_flattened_from_anyof(self, tools: dict) -> None:
        params = tools["list_accounts"].parameters["properties"]
        assert params["type"]["enum"] == ["pagar", "receber"]
        assert "anyOf" not in params["type"]
        assert tools["list_accounts"].query_params == ("type", "status")

    def test_path_param_merge(self, tools: dict) -> None:
        spec = tools["get_account"]
        assert spec.path == "/accounts/{account_id}"
        assert spec.path_params == ("account_id",)
        assert spec.parameters["properties"]["account_id"]["type"] == "integer"
        assert "account_id" in spec.parameters["required"]

    def test_body_schema_merged_with_ref_resolved(self, tools: dict) -> None:
        spec = tools["create_account"]
        props = spec.parameters["properties"]
        assert {"type", "description", "counterparty", "amount", "due_date"} <= set(props)
        assert set(spec.body_params) == set(props)
        assert {"type", "description", "counterparty", "amount", "due_date"} <= set(spec.parameters["required"])
        # campo opcional (anyOf com null) achatado para o tipo real
        assert props["approver_role"]["enum"] == ["gerente", "diretor"]

    def test_required_query_params(self, tools: dict) -> None:
        spec = tools["get_cashflow"]
        assert set(spec.query_params) == {"start", "end"}
        assert set(spec.parameters["required"]) == {"start", "end"}

    def test_description_joins_summary_and_description(self, tools: dict) -> None:
        description = tools["create_account"].description
        assert description.startswith("Cria conta a pagar ou a receber")
        assert "alçada" in description

    def test_ollama_tool_format(self, tools: dict) -> None:
        tool = tools["list_accounts"].as_ollama_tool()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "list_accounts"
        assert tool["function"]["parameters"]["type"] == "object"


def _registry(spec: dict, handler) -> ToolRegistry:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    registry = ToolRegistry({"financas": "http://financas.test"}, client=client)
    registry.preload("financas", parse_openapi(spec))
    return registry


class TestExecutor:
    def test_path_params_substituted(self, spec: dict) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["method"] = request.method
            return httpx.Response(200, json={"id": 3})

        registry = _registry(spec, handler)
        result = registry.execute("financas", "get_account", {"account_id": 3})
        assert seen["url"] == "http://financas.test/accounts/3"
        assert seen["method"] == "GET"
        assert result == {"status": 200, "body": {"id": 3}}

    def test_query_params_sent(self, spec: dict) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["params"] = dict(request.url.params)
            return httpx.Response(200, json=[])

        registry = _registry(spec, handler)
        registry.execute("financas", "list_accounts", {"type": "pagar", "status": None})
        assert seen["params"] == {"type": "pagar"}  # None nunca vira query

    def test_body_sent_as_json(self, spec: dict) -> None:
        seen: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["json"] = json.loads(request.content)
            return httpx.Response(201, json={"id": 12})

        registry = _registry(spec, handler)
        args = {
            "type": "pagar",
            "description": "Suprimentos",
            "counterparty": "Papelaria Central",
            "amount": 1200.0,
            "due_date": "2026-07-10",
        }
        result = registry.execute("financas", "create_account", args)
        assert seen["json"] == args
        assert result["status"] == 201

    def test_business_error_422_returned_not_raised(self, spec: dict) -> None:
        body = {"error": "insufficient_approval_authority", "detail": "exige gerente", "rule": "alçada"}

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json=body)

        registry = _registry(spec, handler)
        result = registry.execute(
            "financas",
            "create_account",
            {"type": "pagar", "description": "x", "counterparty": "y", "amount": 30000, "due_date": "2026-07-15"},
        )
        assert result == {"status": 422, "body": body}

    def test_not_found_404_returned_not_raised(self, spec: dict) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"error": "account_not_found", "detail": "não existe"})

        registry = _registry(spec, handler)
        result = registry.execute("financas", "get_account", {"account_id": 999})
        assert result["status"] == 404

    def test_unknown_tool_raises_tool_not_found(self, spec: dict) -> None:
        registry = _registry(spec, lambda request: httpx.Response(200))
        with pytest.raises(ToolNotFound) as excinfo:
            registry.execute("financas", "delete_everything", {})
        assert "list_accounts" in excinfo.value.known

    def test_registry_fetches_openapi_lazily_and_caches(self, spec: dict) -> None:
        calls = {"openapi": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/openapi.json":
                calls["openapi"] += 1
                return httpx.Response(200, json=spec)
            return httpx.Response(200, json=[])

        client = httpx.Client(transport=httpx.MockTransport(handler))
        registry = ToolRegistry({"financas": "http://financas.test"}, client=client)
        assert len(registry.tools_for("financas")) == 6
        registry.execute("financas", "list_accounts", {})
        assert calls["openapi"] == 1
