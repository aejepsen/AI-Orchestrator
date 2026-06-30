"""Agregador de métricas Langfuse com cache e degradação graceful.

Consulta traces via SDK Python, agrega latência (avg/p50/p95), tokens,
contagem por routing layer e injection blocks. Cache de 30s evita
sobrecarga no Langfuse.
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

from gateway.config import Settings

logger = logging.getLogger("gateway")

_CACHE_TTL_S = 30.0


@dataclass
class _CachedMetrics:
    data: dict[str, Any] = field(default_factory=dict)
    fetched_at: float = 0.0

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.fetched_at > _CACHE_TTL_S


class MetricsCollector:
    """Coleta e agrega métricas do Langfuse com cache em memória."""

    def __init__(self, settings: Settings) -> None:
        self._enabled = settings.langfuse_enabled
        self._langfuse = None
        self._cache = _CachedMetrics()

        if not self._enabled:
            return
        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(
                host=settings.langfuse_host,
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
            )
        except Exception as exc:
            logger.warning("MetricsCollector: Langfuse indisponível: %s", exc)
            self._enabled = False

    def collect(self) -> dict[str, Any]:
        """Retorna métricas agregadas. Cache de 30s."""
        if not self._enabled or self._langfuse is None:
            return _empty_metrics(available=False)

        if not self._cache.expired:
            return self._cache.data

        try:
            result = self._fetch_and_aggregate()
            self._cache = _CachedMetrics(data=result, fetched_at=time.monotonic())
            return result
        except Exception as exc:
            logger.warning("MetricsCollector: falha ao coletar: %s", exc)
            # Se tem cache antigo, retorna ele com flag stale
            if self._cache.data:
                return {**self._cache.data, "stale": True}
            return _empty_metrics(available=False)

    def _fetch_and_aggregate(self) -> dict[str, Any]:
        """Busca traces recentes e agrega métricas."""
        traces_response = self._langfuse.fetch_traces(limit=100)  # type: ignore[union-attr]
        traces = traces_response.data if hasattr(traces_response, "data") else []

        if not traces:
            return _empty_metrics(available=True)

        latencies: list[float] = []
        tokens_input = 0
        tokens_output = 0
        route_semantic = 0
        route_llm = 0
        injection_blocks = 0
        error_count = 0

        for trace in traces:
            # Latência (segundos) — calcula a partir de timestamps se disponível
            latency = _extract_latency(trace)
            if latency is not None and latency > 0:
                latencies.append(latency)

            # Tokens — extrair do usage do trace ou metadata
            usage = _safe_attr(trace, "usage")
            if usage:
                tokens_input += _safe_int(usage, "input", 0) or _safe_int(usage, "promptTokens", 0)
                tokens_output += _safe_int(usage, "output", 0) or _safe_int(usage, "completionTokens", 0)

            # Metadata com info de routing
            metadata = _safe_attr(trace, "metadata") or {}
            if isinstance(metadata, dict):
                layer = metadata.get("routing_layer", "")
                if layer == "semantic":
                    route_semantic += 1
                elif layer == "llm":
                    route_llm += 1
                if metadata.get("injection_blocked"):
                    injection_blocks += 1

            # Erros
            level = _safe_attr(trace, "level")
            if level and str(level).upper() == "ERROR":
                error_count += 1

        # Calcular percentis
        avg_latency = statistics.mean(latencies) if latencies else 0.0
        p50_latency = statistics.median(latencies) if latencies else 0.0
        p95_latency = _percentile(latencies, 0.95) if latencies else 0.0

        return {
            "available": True,
            "stale": False,
            "total_traces": len(traces),
            "latency": {
                "avg_ms": round(avg_latency * 1000, 1),
                "p50_ms": round(p50_latency * 1000, 1),
                "p95_ms": round(p95_latency * 1000, 1),
                "sample_size": len(latencies),
            },
            "tokens": {
                "input": tokens_input,
                "output": tokens_output,
                "total": tokens_input + tokens_output,
            },
            "routing": {
                "semantic": route_semantic,
                "llm": route_llm,
                "unclassified": len(traces) - route_semantic - route_llm,
            },
            "injection_blocks": injection_blocks,
            "error_count": error_count,
        }


def _empty_metrics(*, available: bool) -> dict[str, Any]:
    return {
        "available": available,
        "stale": False,
        "total_traces": 0,
        "latency": {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "sample_size": 0},
        "tokens": {"input": 0, "output": 0, "total": 0},
        "routing": {"semantic": 0, "llm": 0, "unclassified": 0},
        "injection_blocks": 0,
        "error_count": 0,
    }


def _extract_latency(trace: Any) -> float | None:
    """Extrai latência em segundos de um trace Langfuse."""
    # SDK v2: trace.latency (já em ms ou s dependendo da versão)
    latency = _safe_attr(trace, "latency")
    if latency is not None:
        # Se > 1000, provavelmente em ms
        return float(latency) / 1000.0 if float(latency) > 100 else float(latency)

    # Fallback: calcular de timestamps
    start = _safe_attr(trace, "timestamp") or _safe_attr(trace, "startTime")
    end = _safe_attr(trace, "endTime")
    if start and end:
        try:
            delta = end - start
            return delta.total_seconds() if hasattr(delta, "total_seconds") else float(delta)
        except Exception:
            pass
    return None


def _safe_attr(obj: Any, attr: str) -> Any:
    """Acessa atributo ou chave de dict de forma segura."""
    if isinstance(obj, dict):
        return obj.get(attr)
    return getattr(obj, attr, None)


def _safe_int(obj: Any, key: str, default: int = 0) -> int:
    val = _safe_attr(obj, key)
    if val is None:
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _percentile(data: list[float], p: float) -> float:
    """Calcula percentil sem numpy."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
