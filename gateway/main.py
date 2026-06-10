"""Experience Layer: endpoint único SSE do orquestrador.

`POST /chat {question}` → text/event-stream com eventos:
- `route`  — JSON do RoutePlan assim que o classificador conclui;
- `agent`  — um por domínio concluído: {domain, answer};
- `final`  — {answer, trace_id};
- `error`  — {detail, trace_id} em falha não tratada do grafo.

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
from typing import Any, Callable

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from gateway.agents import DomainAgentRunner
from gateway.graph import GatewayGraph

logging.basicConfig(level=logging.INFO, format="%(message)s")

_SENTINEL = ("__done__", None)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(graph_factory: Callable[[], GatewayGraph] | None = None) -> FastAPI:
    app = FastAPI(title="AI-Orchestrator Gateway", version="0.3.0")
    factory = graph_factory or (lambda: GatewayGraph(DomainAgentRunner()))

    def get_graph() -> GatewayGraph:
        # Lazy + cacheado: registry (OpenAPI dos serviços) só no primeiro /chat.
        if not hasattr(app.state, "graph"):
            app.state.graph = factory()
        return app.state.graph

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    async def chat(request: ChatRequest) -> StreamingResponse:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, dict[str, Any] | None]] = asyncio.Queue()
        trace_id = str(uuid.uuid4())

        def emit(event: str, data: dict[str, Any] | None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, (event, data))

        def worker() -> None:
            try:
                graph = get_graph()
                stream = graph.stream(
                    request.question,
                    trace_id=trace_id,
                    on_agent=lambda domain, answer: emit("agent", {"domain": domain, "answer": answer}),
                )
                for update in stream:
                    for node, payload in update.items():
                        if node == "classify":
                            emit("route", payload["route"])
                        elif node in ("synthesize", "respond_clarification"):
                            emit("final", {"answer": payload["final_answer"], "trace_id": trace_id})
            except Exception as exc:  # noqa: BLE001 — erro vira evento SSE, não 500 mudo
                emit("error", {"detail": str(exc), "trace_id": trace_id})
            finally:
                emit(*_SENTINEL)

        future = loop.run_in_executor(None, worker)

        async def event_stream():
            while True:
                event, data = await queue.get()
                if (event, data) == _SENTINEL:
                    break
                yield _sse(event, data or {})
            await future

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


app = create_app()
