"""Circuit breaker por domínio para o executor de tools.

Implementação própria, sem dependências novas: janela de falhas
CONSECUTIVAS de transporte (timeout/conexão/5xx — 4xx é erro de negócio
e não conta); ao atingir o limiar o circuito ABRE por um cooldown e as
chamadas retornam resposta degradada imediata (o agente trata status
!= 2xx como observação e explica a indisponibilidade). Após o cooldown,
half-open: 1 probe; sucesso fecha, falha reabre.

Thread-safe (o dispatch do grafo usa ThreadPoolExecutor): lock por
domínio. Transições de estado são logadas em JSON no logger "gateway"
(trace_id não disponível nesta camada; logamos domain/state/failures).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable

logger = logging.getLogger("gateway")

CLOSED = "closed"
OPEN = "open"
HALF_OPEN = "half_open"


def open_response(domain: str) -> dict[str, Any]:
    """Resposta degradada explícita devolvida ao agente com o circuito aberto."""
    return {
        "status": 0,
        "body": {
            "error": "service_unavailable",
            "detail": (
                f"O serviço de {domain} está temporariamente indisponível; "
                "tente novamente em instantes."
            ),
            "rule": "circuit_breaker",
        },
    }


def transport_error_response(domain: str) -> dict[str, Any]:
    """Observação devolvida ao agente quando uma chamada individual falha no transporte."""
    return {
        "status": 0,
        "body": {
            "error": "service_unavailable",
            "detail": (
                f"Falha de comunicação com o serviço de {domain}; "
                "tente novamente em instantes."
            ),
            "rule": "transport_error",
        },
    }


class CircuitBreaker:
    """Estado por domínio: CLOSED → (N falhas consecutivas) → OPEN → (cooldown) → HALF_OPEN."""

    def __init__(
        self,
        *,
        threshold: int = 3,
        cooldown_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._threshold = threshold
        self._cooldown_s = cooldown_s
        self._clock = clock
        self._domains: dict[str, dict[str, Any]] = {}
        self._registry_lock = threading.Lock()

    def _entry(self, domain: str) -> dict[str, Any]:
        with self._registry_lock:
            if domain not in self._domains:
                self._domains[domain] = {
                    "lock": threading.Lock(),
                    "state": CLOSED,
                    "failures": 0,
                    "opened_at": 0.0,
                }
            return self._domains[domain]

    def _log_transition(self, domain: str, state: str, failures: int) -> None:
        logger.warning(
            json.dumps(
                {"event": "circuit_breaker", "domain": domain, "state": state, "failures": failures},
                ensure_ascii=False,
            )
        )

    def state(self, domain: str) -> str:
        entry = self._entry(domain)
        with entry["lock"]:
            return entry["state"]

    def allow(self, domain: str) -> bool:
        """True se a chamada pode prosseguir; em OPEN expirado vira o probe half-open."""
        entry = self._entry(domain)
        with entry["lock"]:
            if entry["state"] == CLOSED:
                return True
            if entry["state"] == OPEN:
                if self._clock() - entry["opened_at"] >= self._cooldown_s:
                    entry["state"] = HALF_OPEN
                    self._log_transition(domain, HALF_OPEN, entry["failures"])
                    return True
                return False
            return False  # HALF_OPEN: probe já em voo; demais chamadas degradam

    def record_success(self, domain: str) -> None:
        entry = self._entry(domain)
        with entry["lock"]:
            if entry["state"] != CLOSED:
                self._log_transition(domain, CLOSED, 0)
            entry["state"] = CLOSED
            entry["failures"] = 0

    def record_failure(self, domain: str) -> None:
        entry = self._entry(domain)
        with entry["lock"]:
            if entry["state"] == HALF_OPEN:
                entry["state"] = OPEN
                entry["opened_at"] = self._clock()
                self._log_transition(domain, OPEN, entry["failures"])
                return
            entry["failures"] += 1
            if entry["state"] == CLOSED and entry["failures"] >= self._threshold:
                entry["state"] = OPEN
                entry["opened_at"] = self._clock()
                self._log_transition(domain, OPEN, entry["failures"])
