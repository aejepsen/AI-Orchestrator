"""Experience Layer: endpoint único SSE do orquestrador.

`POST /chat {question}` → text/event-stream com eventos:
- `route`  — JSON do RoutePlan assim que o classificador conclui;
- `agent`  — um por domínio concluído: {domain, answer};
- `confirm` — pendente de confirmação humana: {domains, plan, thread_id};
- `final`  — {answer, trace_id};
- `error`  — {detail, trace_id} em falha não tratada do grafo.

`POST /chat/{thread_id}/resume {approved}` → retoma grafo após HITL.

O grafo é síncrono (LLM + microsserviços via httpx sync); roda em
run_in_executor e os eventos fluem por asyncio.Queue conforme cada nó
conclui. O grafo é construído lazy no primeiro /chat para que /health
não dependa dos microsserviços (o registry busca os OpenAPI no boot).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gateway.agents import DomainAgentRunner
from gateway.config import Settings, load_settings
from gateway.graph import GatewayGraph
from gateway.security import AccessTokenGuard, RateLimiter, client_ip
from gateway.tracing import Tracer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

_SENTINEL = ("__done__", None)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=64)


class ResumeRequest(BaseModel):
    approved: bool = True


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_FRONTEND_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


def create_app(
    graph_factory: Callable[[], GatewayGraph] | None = None,
    access_guard: AccessTokenGuard | None = None,
    rate_limiter: RateLimiter | None = None,
) -> FastAPI:
    app = FastAPI(title="AI-Orchestrator Gateway", version="0.5.0", docs_url=None, redoc_url=None, openapi_url=None)
    factory = graph_factory or (lambda: _default_graph_factory())
    guard = access_guard or AccessTokenGuard()
    limiter = rate_limiter or RateLimiter()

    def get_graph() -> GatewayGraph:
        # Lazy + cacheado: registry (OpenAPI dos serviços) só no primeiro /chat.
        if not hasattr(app.state, "graph"):
            app.state.graph = factory()
        return app.state.graph

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat", response_model=None)
    async def chat(body: ChatRequest, http_request: Request) -> StreamingResponse | JSONResponse:
        if not guard.allows(http_request.headers.get("x-access-token")):
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "detail": "Token de acesso ausente ou inválido."},
            )
        fallback_ip = http_request.client.host if http_request.client else "unknown"
        if not limiter.allow(client_ip(http_request.headers, fallback_ip)):
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "detail": "Limite de requisições por hora atingido."},
            )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict[str, Any] | None]] = asyncio.Queue()
        trace_id = str(uuid.uuid4())
        tid = body.thread_id or str(uuid.uuid4())

        def emit(event: str, data: dict[str, Any] | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        def worker() -> None:
            try:
                graph = get_graph()
                stream = graph.stream(
                    body.question,
                    trace_id=trace_id,
                    thread_id=tid,
                    on_agent=lambda domain, answer: emit("agent", {"domain": domain, "answer": answer}),
                    # HITL desabilitado até detecção de intent write vs read.
                    # on_confirm=lambda domains, plan: emit("confirm", {"domains": domains, "plan": plan, "thread_id": tid}),
                )
                for update in stream:
                    for node, payload in update.items():
                        if not payload:
                            continue
                        if node == "classify":
                            emit("route", payload.get("route", {}))
                        elif node == "confirm_dispatch":
                            if "final_answer" in payload:
                                emit("final", {"answer": payload["final_answer"], "trace_id": trace_id})
                        elif node in ("synthesize", "respond_clarification"):
                            emit("final", {"answer": payload["final_answer"], "trace_id": trace_id})
            except Exception as exc:  # noqa: BLE001 — erro vira evento SSE, não 500 mudo
                logger.exception("Erro no grafo: %s", exc)
                emit("error", {"detail": "Erro interno. Tente novamente.", "trace_id": trace_id})
            finally:
                emit(*_SENTINEL)

        future = loop.run_in_executor(None, worker)

        async def event_stream():
            # Heartbeat: Cloudflare encerra com 524 após ~100 s sem bytes; os
            # agentes podem ficar minutos entre eventos. Comentário SSE (":")
            # mantém a conexão viva e é ignorado pelos parsers.
            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if (event, data) == _SENTINEL:
                    break
                yield _sse(event, data or {})
            await future

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.post("/chat/{thread_id}/resume", response_model=None)
    async def resume_chat(thread_id: str, body: ResumeRequest, http_request: Request) -> StreamingResponse | JSONResponse:
        if not guard.allows(http_request.headers.get("x-access-token")):
            return JSONResponse(
                status_code=401,
                content={"error": "unauthorized", "detail": "Token de acesso ausente ou inválido."},
            )
        fallback_ip = http_request.client.host if http_request.client else "unknown"
        if not limiter.allow(client_ip(http_request.headers, fallback_ip)):
            return JSONResponse(
                status_code=429,
                content={"error": "rate_limited", "detail": "Limite de requisições por hora atingido."},
            )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict[str, Any] | None]] = asyncio.Queue()
        trace_id = str(uuid.uuid4())

        def emit(event: str, data: dict[str, Any] | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        def worker() -> None:
            try:
                from langgraph.types import Command

                graph = get_graph()
                config = {"configurable": {"thread_id": thread_id}}
                stream = graph._compiled.stream(
                    Command(resume={"approved": body.approved}),
                    config=config,
                    stream_mode="updates",
                )
                for update in stream:
                    for node, payload in update.items():
                        if not payload:
                            continue
                        if node in ("synthesize", "respond_clarification", "confirm_dispatch"):
                            if "final_answer" in payload:
                                emit("final", {"answer": payload["final_answer"], "trace_id": trace_id})
            except Exception as exc:  # noqa: BLE001
                logger.exception("Erro no grafo (resume): %s", exc)
                emit("error", {"detail": "Erro interno. Tente novamente.", "trace_id": trace_id})
            finally:
                emit(*_SENTINEL)

        future = loop.run_in_executor(None, worker)

        async def event_stream():
            while True:
                try:
                    event, data = await asyncio.wait_for(queue.get(), timeout=15.0)
                except TimeoutError:
                    yield ": keepalive\n\n"
                    continue
                if (event, data) == _SENTINEL:
                    break
                yield _sse(event, data or {})
            await future

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Frontend buildado (Vite) servido pelo próprio gateway. Montado por último:
    # /health e /chat têm precedência; html=True serve index.html na raiz.
    if _FRONTEND_DIST.is_dir():
        app.mount("/", StaticFiles(directory=_FRONTEND_DIST, html=True), name="frontend")

    return app


def _default_graph_factory() -> GatewayGraph:
    """Factory padrão: cria o grafo com tracer Langfuse e settings do ambiente."""
    settings = load_settings()
    tracer = Tracer(settings)
    runner = DomainAgentRunner()
    return GatewayGraph(runner, tracer=tracer, settings=settings)


app = create_app()
