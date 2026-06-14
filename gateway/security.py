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
            if os.environ.get("ALLOW_OPEN_ACCESS") == "1":
                logger.warning("ACCESS_TOKEN não configurado + ALLOW_OPEN_ACCESS=1: /chat aberto (modo dev)")
            else:
                logger.warning("ACCESS_TOKEN não configurado: /chat bloqueado (fail-closed)")

    @property
    def enabled(self) -> bool:
        return self._expected is not None

    def allows(self, provided: str | None) -> bool:
        if self._expected is None:
            # Fail-closed: sem token configurado, bloqueia em produção.
            # Modo dev explícito requer ALLOW_OPEN_ACCESS=1.
            return os.environ.get("ALLOW_OPEN_ACCESS") == "1"
        if not provided:
            return False
        return hmac.compare_digest(provided.encode(), self._expected.encode())


class RateLimiter:
    """Sliding window em memória por IP: máx. N requests por janela.

    Protegido contra memory leak: max_entries limita o número de IPs
    rastreados. Quando excedido, entradas com deque vazio são removidas;
    se ainda acima do limite, as entradas mais antigas são evictadas.
    """

    def __init__(
        self,
        max_requests: int | None = None,
        window_s: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
        max_entries: int = 10_000,
    ) -> None:
        self.max_requests = (
            max_requests if max_requests is not None else int(os.environ.get("RATE_LIMIT_PER_HOUR", "10"))
        )
        self.window_s = window_s
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._max_entries = max_entries
        self._call_count = 0

    def _evict(self) -> None:
        """Remove entradas expiradas; se ainda acima do limite, evicta as mais antigas."""
        # Fase 1: remover IPs sem hits (deques vazios).
        empty_keys = [k for k, v in self._hits.items() if not v]
        for k in empty_keys:
            del self._hits[k]
        if len(self._hits) <= self._max_entries:
            return
        # Fase 2: ordenar por timestamp mais recente e remover os mais antigos.
        by_latest = sorted(self._hits.items(), key=lambda kv: kv[1][-1] if kv[1] else 0.0)
        to_remove = len(self._hits) - self._max_entries
        for k, _ in by_latest[:to_remove]:
            del self._hits[k]

    def allow(self, client_ip: str) -> bool:
        now = self._clock()
        hits = self._hits.setdefault(client_ip, deque())
        while hits and now - hits[0] >= self.window_s:
            hits.popleft()
        if len(hits) >= self.max_requests:
            return False
        hits.append(now)
        # Eviction periódico: a cada 1000 requests, limpa entradas expiradas.
        self._call_count += 1
        if self._call_count >= 1000:
            self._call_count = 0
            self._evict()
        return True


def client_ip(headers: Mapping[str, str], fallback: str) -> str:
    """IP real atrás do Cloudflare — fallback chain confiável.

    CF-Connecting-IP é setado pelo Cloudflare e não pode ser forjado pelo
    cliente. X-Forwarded-For é facilmente spoofável com um header custom.
    """
    cf_ip = headers.get("cf-connecting-ip", "").strip()
    if cf_ip:
        return cf_ip
    real_ip = headers.get("x-real-ip", "").strip()
    if real_ip:
        return real_ip
    forwarded = headers.get("x-forwarded-for", "")
    first = forwarded.split(",")[0].strip()
    return first or fallback
