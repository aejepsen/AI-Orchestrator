"""Boundary público do gateway: access token + rate limit por IP.

Pensado para exposição via Cloudflare Tunnel: o IP real do cliente chega no
primeiro hop de `X-Forwarded-For`. Sem `ACCESS_TOKEN` configurado o /chat fica
aberto (modo dev) — logado como warning na criação do guard.

Estado em memória por processo: suficiente para a PoC (1 réplica). Em produção
multi-réplica o rate limit migraria para um store compartilhado (Redis).
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from collections import deque
from typing import Callable, Mapping

logger = logging.getLogger(__name__)


class AccessTokenGuard:
    """Compara `X-Access-Token` com `ACCESS_TOKEN` em tempo constante."""

    def __init__(self, expected: str | None = None) -> None:
        self._expected = expected if expected is not None else (os.environ.get("ACCESS_TOKEN") or None)
        if self._expected is None:
            logger.warning("ACCESS_TOKEN não configurado: POST /chat aberto (modo dev)")

    @property
    def enabled(self) -> bool:
        return self._expected is not None

    def allows(self, provided: str | None) -> bool:
        if self._expected is None:
            return True
        if not provided:
            return False
        return hmac.compare_digest(provided.encode(), self._expected.encode())


class RateLimiter:
    """Sliding window em memória por IP: máx. N requests por janela."""

    def __init__(
        self,
        max_requests: int | None = None,
        window_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_requests = (
            max_requests if max_requests is not None else int(os.environ.get("RATE_LIMIT_PER_HOUR", "10"))
        )
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}

    def allow(self, client_ip: str) -> bool:
        now = self._clock()
        hits = self._hits.setdefault(client_ip, deque())
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        return True


def client_ip(headers: Mapping[str, str], fallback: str) -> str:
    """IP real atrás do Cloudflare: primeiro hop de `X-Forwarded-For`."""
    forwarded = headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    return first or fallback
