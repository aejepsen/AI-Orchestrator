"""Testes do envio do header X-Internal-Key pelo registry (gateway→serviços)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from gateway.config import load_settings
from gateway.tools.registry import ToolRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "financas_openapi.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(FIXTURE.read_text())


def test_header_sent_on_openapi_fetch_and_execute(spec: dict) -> None:
    seen: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.headers.get("X-Internal-Key")))
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=spec)
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    registry = ToolRegistry(
        {"financas": "http://financas.test"}, client=client, internal_api_key="secret-key"
    )
    registry.execute("financas", "list_accounts", {})

    # Header em TODA chamada: fetch do openapi.json (isento no serviço, enviado
    # por simetria) e chamada de negócio.
    assert seen == [("/openapi.json", "secret-key"), ("/accounts", "secret-key")]


def test_no_header_when_key_not_configured(spec: dict) -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("X-Internal-Key"))
        if request.url.path == "/openapi.json":
            return httpx.Response(200, json=spec)
        return httpx.Response(200, json=[])

    client = httpx.Client(transport=httpx.MockTransport(handler))
    registry = ToolRegistry({"financas": "http://financas.test"}, client=client)
    registry.execute("financas", "list_accounts", {})
    assert seen == [None, None]


def test_settings_load_internal_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_API_KEY", "from-env")
    assert load_settings().internal_api_key == "from-env"
    monkeypatch.delenv("INTERNAL_API_KEY")
    assert load_settings().internal_api_key is None
