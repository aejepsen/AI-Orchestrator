"""Testes do circuit breaker por domínio (gateway/tools/circuit.py + integração no registry).

Sem rede: MockTransport para chamadas HTTP, clock injetado para o cooldown.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from gateway.tools.circuit import CLOSED, HALF_OPEN, OPEN, CircuitBreaker
from gateway.tools.registry import ToolRegistry, parse_openapi

FIXTURE = Path(__file__).parent / "fixtures" / "financas_openapi.json"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(FIXTURE.read_text())


def _registry(spec: dict, handler, clock: FakeClock) -> tuple[ToolRegistry, CircuitBreaker]:
    breaker = CircuitBreaker(threshold=3, cooldown_s=30.0, clock=clock)
    client = httpx.Client(transport=httpx.MockTransport(handler))
    registry = ToolRegistry({"financas": "http://financas.test"}, client=client, breaker=breaker)
    registry.preload("financas", parse_openapi(spec))
    return registry, breaker


def _connect_error_handler(calls: list) :
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        raise httpx.ConnectError("connection refused", request=request)

    return handler


class TestTransportFailuresOpenCircuit:
    def test_three_connect_errors_open(self, spec: dict) -> None:
        clock, calls = FakeClock(), []
        registry, breaker = _registry(spec, _connect_error_handler(calls), clock)
        for _ in range(3):
            result = registry.execute("financas", "list_accounts", {})
            assert result["status"] == 0
            assert result["body"]["rule"] == "transport_error"
        assert breaker.state("financas") == OPEN
        assert len(calls) == 3

    def test_three_5xx_open(self, spec: dict) -> None:
        clock = FakeClock()
        registry, breaker = _registry(spec, lambda r: httpx.Response(500, json={"detail": "boom"}), clock)
        for _ in range(3):
            result = registry.execute("financas", "list_accounts", {})
            assert result["status"] == 500  # 5xx real ainda volta como observação
        assert breaker.state("financas") == OPEN

    def test_422_does_not_open(self, spec: dict) -> None:
        clock, calls = FakeClock(), []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            return httpx.Response(422, json={"error": "x", "detail": "y", "rule": "z"})

        registry, breaker = _registry(spec, handler, clock)
        for _ in range(5):
            assert registry.execute("financas", "list_accounts", {})["status"] == 422
        assert breaker.state("financas") == CLOSED
        assert len(calls) == 5

    def test_404_does_not_open(self, spec: dict) -> None:
        clock = FakeClock()
        registry, breaker = _registry(spec, lambda r: httpx.Response(404, json={"error": "nf"}), clock)
        for _ in range(5):
            registry.execute("financas", "get_account", {"account_id": 99})
        assert breaker.state("financas") == CLOSED

    def test_success_resets_consecutive_window(self, spec: dict) -> None:
        clock = FakeClock()
        responses = ["fail", "fail", "ok", "fail", "fail"]

        def handler(request: httpx.Request) -> httpx.Response:
            kind = responses.pop(0)
            if kind == "fail":
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, json=[])

        registry, breaker = _registry(spec, handler, clock)
        for _ in range(5):
            registry.execute("financas", "list_accounts", {})
        assert breaker.state("financas") == CLOSED  # nunca 3 falhas CONSECUTIVAS


class TestOpenAndHalfOpen:
    def _opened(self, spec: dict, calls: list) -> tuple[ToolRegistry, CircuitBreaker, FakeClock]:
        clock = FakeClock()
        registry, breaker = _registry(spec, _connect_error_handler(calls), clock)
        for _ in range(3):
            registry.execute("financas", "list_accounts", {})
        assert breaker.state("financas") == OPEN
        return registry, breaker, clock

    def test_open_returns_degraded_without_http_call(self, spec: dict) -> None:
        calls: list = []
        registry, _, _ = self._opened(spec, calls)
        result = registry.execute("financas", "list_accounts", {})
        assert result == {
            "status": 0,
            "body": {
                "error": "service_unavailable",
                "detail": (
                    "O serviço de financas está temporariamente indisponível; "
                    "tente novamente em instantes."
                ),
                "rule": "circuit_breaker",
            },
        }
        assert len(calls) == 3  # nenhuma chamada nova durante OPEN

    def test_half_open_success_closes(self, spec: dict) -> None:
        clock, calls = FakeClock(), []
        state = {"fail": True}

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request.url.path)
            if state["fail"]:
                raise httpx.ConnectError("refused", request=request)
            return httpx.Response(200, json=[])

        registry, breaker = _registry(spec, handler, clock)
        for _ in range(3):
            registry.execute("financas", "list_accounts", {})
        assert breaker.state("financas") == OPEN

        state["fail"] = False
        clock.advance(30.0)  # cooldown expira → probe half-open
        result = registry.execute("financas", "list_accounts", {})
        assert result["status"] == 200
        assert breaker.state("financas") == CLOSED
        assert registry.execute("financas", "list_accounts", {})["status"] == 200

    def test_half_open_failure_reopens(self, spec: dict) -> None:
        calls: list = []
        registry, breaker, clock = self._opened(spec, calls)
        clock.advance(30.0)
        result = registry.execute("financas", "list_accounts", {})  # probe falha
        assert result["status"] == 0
        assert breaker.state("financas") == OPEN
        assert len(calls) == 4
        # reaberto: degradado imediato, sem HTTP
        assert registry.execute("financas", "list_accounts", {})["body"]["rule"] == "circuit_breaker"
        assert len(calls) == 4

    def test_before_cooldown_still_open(self, spec: dict) -> None:
        calls: list = []
        registry, breaker, clock = self._opened(spec, calls)
        clock.advance(29.9)
        assert registry.execute("financas", "list_accounts", {})["body"]["rule"] == "circuit_breaker"
        assert breaker.state("financas") == OPEN
        assert len(calls) == 3


class TestBreakerUnit:
    def test_half_open_allows_single_probe(self) -> None:
        clock = FakeClock()
        breaker = CircuitBreaker(threshold=3, cooldown_s=30.0, clock=clock)
        for _ in range(3):
            breaker.record_failure("rh")
        clock.advance(30.0)
        assert breaker.allow("rh") is True  # probe
        assert breaker.state("rh") == HALF_OPEN
        assert breaker.allow("rh") is False  # demais chamadas degradam

    def test_domains_are_isolated(self) -> None:
        breaker = CircuitBreaker(threshold=3, cooldown_s=30.0, clock=FakeClock())
        for _ in range(3):
            breaker.record_failure("estoque")
        assert breaker.state("estoque") == OPEN
        assert breaker.allow("vendas") is True

    def test_transition_logged_as_json(self, caplog) -> None:
        breaker = CircuitBreaker(threshold=3, cooldown_s=30.0, clock=FakeClock())
        with caplog.at_level("WARNING", logger="gateway"):
            for _ in range(3):
                breaker.record_failure("financas")
        events = [json.loads(r.message) for r in caplog.records]
        assert {"event": "circuit_breaker", "domain": "financas", "state": "open", "failures": 3} in events
