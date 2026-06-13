"""Langfuse tracing — observabilidade LLM por request.

Cada request do /chat é um trace; cada nó do grafo é um span; cada chamada
ao LLM é uma generation. Degradação graceful: Langfuse fora → log warning,
request continua normal.
"""

from __future__ import annotations

import logging
from typing import Any

from gateway.config import Settings

logger = logging.getLogger("gateway")


class Tracer:
    """Wrapper sobre Langfuse SDK com degradação graceful."""

    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.langfuse_enabled
        self._langfuse = None
        if not self._enabled:
            return
        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(
                host=settings.langfuse_host,
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
            )
            logger.info("Langfuse tracing ativo: %s", settings.langfuse_host)
        except Exception as exc:
            logger.warning("Langfuse indisponível (tracing desativado): %s", exc)
            self._enabled = False

    def trace(self, *, trace_id: str, name: str = "chat", **kwargs: Any) -> TraceHandle:
        if not self._enabled or self._langfuse is None:
            return TraceHandle(None)
        try:
            t = self._langfuse.trace(id=trace_id, name=name, **kwargs)
            return TraceHandle(t)
        except Exception:
            return TraceHandle(None)

    def flush(self) -> None:
        if self._langfuse:
            try:
                self._langfuse.flush()
            except Exception:
                pass


class TraceHandle:
    """Handle de um trace/span com safe wrappers."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def span(self, *, name: str, **kwargs: Any) -> TraceHandle:
        if self._raw is None:
            return TraceHandle(None)
        try:
            return TraceHandle(self._raw.span(name=name, **kwargs))
        except Exception:
            return TraceHandle(None)

    def generation(self, *, name: str, model: str, input: Any, **kwargs: Any) -> GenerationHandle:
        if self._raw is None:
            return GenerationHandle(None)
        try:
            return GenerationHandle(self._raw.generation(name=name, model=model, input=input, **kwargs))
        except Exception:
            return GenerationHandle(None)

    def update(self, **kwargs: Any) -> None:
        if self._raw:
            try:
                self._raw.update(**kwargs)
            except Exception:
                pass

    def end(self, **kwargs: Any) -> None:
        if self._raw:
            try:
                self._raw.end(**kwargs)
            except Exception:
                pass


class GenerationHandle:
    """Handle de uma generation com safe wrappers."""

    def __init__(self, raw: Any) -> None:
        self._raw = raw

    def end(self, *, output: Any = None, **kwargs: Any) -> None:
        if self._raw:
            try:
                self._raw.end(output=output, **kwargs)
            except Exception:
                pass
