"""Testes do boundary público: access token + rate limit (sem LLM)."""

from __future__ import annotations

import asyncio

import httpx

from gateway.main import create_app
from gateway.security import AccessTokenGuard, RateLimiter, client_ip


class FakeGraph:
    def stream(self, question: str, *, trace_id: str | None = None, on_agent=None):
        yield {"classify": {"route": {"domains": ["vendas"], "plan": "p", "clarification": None}}}
        yield {"synthesize": {"final_answer": "ok"}}


def _post(app, headers: dict[str, str] | None = None) -> httpx.Response:
    async def call():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return await client.post("/chat", json={"question": "q"}, headers=headers or {})

    return asyncio.run(call())


# --- AccessTokenGuard ---------------------------------------------------------


def test_guard_sem_token_configurado_fail_closed(monkeypatch):
    monkeypatch.delenv("ALLOW_OPEN_ACCESS", raising=False)
    guard = AccessTokenGuard(expected=None)
    assert not guard.enabled
    assert not guard.allows(None)
    assert not guard.allows("qualquer-coisa")


def test_guard_sem_token_com_allow_open_access(monkeypatch):
    monkeypatch.setenv("ALLOW_OPEN_ACCESS", "1")
    guard = AccessTokenGuard(expected=None)
    assert not guard.enabled
    assert guard.allows(None)
    assert guard.allows("qualquer-coisa")


def test_guard_compara_token_em_tempo_constante():
    guard = AccessTokenGuard(expected="s3cret")
    assert guard.enabled
    assert guard.allows("s3cret")
    assert not guard.allows("errado")
    assert not guard.allows(None)
    assert not guard.allows("")


def test_chat_401_com_token_errado_e_200_com_correto():
    app = create_app(
        graph_factory=lambda: FakeGraph(),
        access_guard=AccessTokenGuard(expected="s3cret"),
        rate_limiter=RateLimiter(max_requests=100),
    )

    negado = _post(app, headers={"X-Access-Token": "errado"})
    assert negado.status_code == 401
    assert negado.json() == {"error": "unauthorized", "detail": "Token de acesso ausente ou inválido."}

    sem_token = _post(app)
    assert sem_token.status_code == 401

    ok = _post(app, headers={"X-Access-Token": "s3cret"})
    assert ok.status_code == 200
    assert ok.headers["content-type"].startswith("text/event-stream")


# --- RateLimiter --------------------------------------------------------------


def test_rate_limiter_bloqueia_apos_limite_e_libera_apos_janela():
    now = [0.0]
    limiter = RateLimiter(max_requests=2, window_s=3600.0, clock=lambda: now[0])

    assert limiter.allow("1.2.3.4")
    assert limiter.allow("1.2.3.4")
    assert not limiter.allow("1.2.3.4")
    # outro IP tem cota própria
    assert limiter.allow("5.6.7.8")
    # janela expira → libera de novo
    now[0] = 3601.0
    assert limiter.allow("1.2.3.4")


def test_rate_limiter_default_vem_do_env(monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_HOUR", "3")
    assert RateLimiter().max_requests == 3


def test_chat_429_apos_limite(monkeypatch):
    monkeypatch.setenv("ALLOW_OPEN_ACCESS", "1")
    app = create_app(
        graph_factory=lambda: FakeGraph(),
        access_guard=AccessTokenGuard(expected=None),
        rate_limiter=RateLimiter(max_requests=1),
    )

    assert _post(app).status_code == 200
    excedido = _post(app)
    assert excedido.status_code == 429
    assert excedido.json() == {"error": "rate_limited", "detail": "Limite de requisições por hora atingido."}


def test_rate_limit_usa_primeiro_hop_de_x_forwarded_for(monkeypatch):
    monkeypatch.setenv("ALLOW_OPEN_ACCESS", "1")
    app = create_app(
        graph_factory=lambda: FakeGraph(),
        access_guard=AccessTokenGuard(expected=None),
        rate_limiter=RateLimiter(max_requests=1),
    )

    assert _post(app, headers={"X-Forwarded-For": "203.0.113.7, 172.16.0.1"}).status_code == 200
    assert _post(app, headers={"X-Forwarded-For": "203.0.113.7, 10.0.0.9"}).status_code == 429
    # IP diferente no primeiro hop → cota própria
    assert _post(app, headers={"X-Forwarded-For": "198.51.100.2"}).status_code == 200


# --- client_ip ----------------------------------------------------------------


def test_client_ip_prefere_x_forwarded_for():
    assert client_ip({"x-forwarded-for": "203.0.113.7, 172.16.0.1"}, "10.0.0.1") == "203.0.113.7"
    assert client_ip({"x-forwarded-for": " 203.0.113.7 "}, "10.0.0.1") == "203.0.113.7"
    assert client_ip({}, "10.0.0.1") == "10.0.0.1"
    assert client_ip({"x-forwarded-for": ""}, "10.0.0.1") == "10.0.0.1"
