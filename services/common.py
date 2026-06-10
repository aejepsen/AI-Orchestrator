"""Envelope de erro compartilhado pelos microsserviços + autenticação interna.

Todo erro de negócio/recurso sai com o mesmo corpo estruturado:
{"error": "<codigo>", "detail": "<PT-BR acionável>", "rule": "<regra avaliada>"}
— esse payload é reinjetado ao agente LLM na Fase 2, então `detail`
precisa ser autoexplicativo.

Autenticação interna (Fase 4): o gateway envia `X-Internal-Key` em toda
chamada; os serviços recusam chamadas diretas sem a key (401 com o mesmo
envelope). Sem `INTERNAL_API_KEY` no ambiente o serviço roda ABERTO com
warning — dev-friendly; em produção o compose define a key.
"""

import hmac
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("service")

# Rotas isentas: probe de saúde e contrato OpenAPI (fonte das tools do gateway).
_PUBLIC_PATHS = frozenset({"/health", "/openapi.json", "/docs"})


class RuleViolation(Exception):
    """Regra de negócio reprovada (HTTP 422)."""

    def __init__(self, error: str, detail: str, rule: str) -> None:
        super().__init__(detail)
        self.error = error
        self.detail = detail
        self.rule = rule


class NotFound(Exception):
    """Recurso inexistente (HTTP 404)."""

    def __init__(self, error: str, detail: str) -> None:
        super().__init__(detail)
        self.error = error
        self.detail = detail


def register_internal_auth(app: FastAPI) -> None:
    """Exige `X-Internal-Key` == env `INTERNAL_API_KEY` em todas as rotas de negócio.

    A env é lida por request (testável via monkeypatch); ausente → serviço
    aberto com warning único no log. Comparação em tempo constante.
    """
    warned = {"open": False}

    @app.middleware("http")
    async def internal_auth(request: Request, call_next):  # type: ignore[no-untyped-def]
        expected = os.environ.get("INTERNAL_API_KEY")
        if not expected:
            if not warned["open"]:
                warned["open"] = True
                logger.warning(
                    "INTERNAL_API_KEY não definida — serviço rodando ABERTO (aceitável apenas em dev)"
                )
            return await call_next(request)
        if request.url.path in _PUBLIC_PATHS:
            return await call_next(request)
        provided = request.headers.get("X-Internal-Key", "")
        if not hmac.compare_digest(provided.encode(), expected.encode()):
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "detail": "Chamada interna exige o header X-Internal-Key com a chave correta.",
                    "rule": "internal_api_key",
                },
            )
        return await call_next(request)


def register_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(RuleViolation)
    async def rule_violation_handler(request: Request, exc: RuleViolation) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"error": exc.error, "detail": exc.detail, "rule": exc.rule},
        )

    @app.exception_handler(NotFound)
    async def not_found_handler(request: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content={"error": exc.error, "detail": exc.detail, "rule": "recurso deve existir"},
        )
