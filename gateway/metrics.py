"""Agregador de métricas Langfuse + OTel com cache e degradação graceful.

Duas fontes independentes:
- Langfuse: traces (latência avg/p50/p95, routing layer, injection, TTFT,
  OOD) + observations do tipo GENERATION (tokens — o usage vive na
  generation, NÃO no trace; ler trace.usage sempre retorna vazio no v2).
- OTel Collector: métricas gen_ai.* raspadas do exporter Prometheus
  (:8889). Verificação cruzada dos mesmos números por pipeline distinto.

Cache de 10s mantém o dashboard quase-live sem sobrecarregar o Langfuse.
"""

from __future__ import annotations

import logging
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from gateway.config import Settings

logger = logging.getLogger("gateway")

_CACHE_TTL_S = 10.0


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
        self._otel_prom_endpoint = settings.otel_prom_endpoint if settings.otel_enabled else ""

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
        """Retorna métricas agregadas (Langfuse + OTel). Cache de 10s."""
        if not self._cache.expired and self._cache.data:
            return self._cache.data

        if self._enabled and self._langfuse is not None:
            try:
                result = self._fetch_and_aggregate()
            except Exception as exc:
                logger.warning("MetricsCollector: falha ao coletar: %s", exc)
                # Se tem cache antigo, retorna ele com flag stale
                if self._cache.data:
                    return {**self._cache.data, "stale": True}
                result = _empty_metrics(available=False)
        else:
            result = _empty_metrics(available=False)

        # Fonte independente: métricas gen_ai.* do OTel Collector.
        result["otel"] = self._fetch_otel_summary()

        self._cache = _CachedMetrics(data=result, fetched_at=time.monotonic())
        return result

    def _fetch_and_aggregate(self) -> dict[str, Any]:
        """Busca traces + generations recentes e agrega métricas."""
        traces_response = self._langfuse.fetch_traces(limit=100)  # type: ignore[union-attr]
        traces = traces_response.data if hasattr(traces_response, "data") else []

        if not traces:
            return _empty_metrics(available=True)

        latencies: list[float] = []
        ttfts: list[float] = []
        ood_residuals: list[float] = []
        tools_used: list[int] = []
        route_semantic = 0
        route_llm = 0
        injection_blocks = 0
        clarifications = 0
        error_count = 0

        for trace in traces:
            # Latência (segundos) — calcula a partir de timestamps se disponível
            latency = _extract_latency(trace)
            if latency is not None and latency > 0:
                latencies.append(latency)

            # Metadata com routing, injection, TTFT e OOD (gravados pelo graph)
            metadata = _safe_attr(trace, "metadata") or {}
            if isinstance(metadata, dict):
                layer = metadata.get("routing_layer", "")
                if layer == "semantic":
                    route_semantic += 1
                elif layer in ("llm", "lexical"):
                    route_llm += 1
                if metadata.get("injection_blocked"):
                    injection_blocks += 1
                if metadata.get("clarification"):
                    clarifications += 1
                tools = metadata.get("tools_used")
                if isinstance(tools, (int, float)):
                    tools_used.append(int(tools))
                ttft = metadata.get("ttft_ms")
                if isinstance(ttft, (int, float)) and ttft > 0:
                    ttfts.append(float(ttft))
                ood = metadata.get("ood_residual")
                if isinstance(ood, (int, float)):
                    ood_residuals.append(float(ood))

            # Erros
            level = _safe_attr(trace, "level")
            if level and str(level).upper() == "ERROR":
                error_count += 1

        # Tokens: usage vive nas GENERATIONS (observations), não no trace.
        tokens_input, tokens_output, generations = self._fetch_generation_usage()

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
                "generations": generations,
            },
            "ttft": {
                "avg_ms": round(statistics.mean(ttfts), 1) if ttfts else 0,
                "p95_ms": round(_percentile(ttfts, 0.95), 1) if ttfts else 0,
                "sample_size": len(ttfts),
            },
            "ood": {
                "avg_residual": round(statistics.mean(ood_residuals), 4) if ood_residuals else 0,
                "max_residual": round(max(ood_residuals), 4) if ood_residuals else 0,
                "sample_size": len(ood_residuals),
            },
            "routing": {
                "semantic": route_semantic,
                "llm": route_llm,
                "unclassified": len(traces) - route_semantic - route_llm,
            },
            "tools": {
                "avg_per_task": round(statistics.mean(tools_used), 2) if tools_used else 0,
                "sample_size": len(tools_used),
            },
            "clarifications": clarifications,
            "injection_blocks": injection_blocks,
            "error_count": error_count,
        }

    def _fetch_generation_usage(self) -> tuple[int, int, int]:
        """Soma tokens das últimas generations (input, output, contagem)."""
        tokens_input = 0
        tokens_output = 0
        generations = 0
        try:
            obs_response = self._langfuse.fetch_observations(  # type: ignore[union-attr]
                type="GENERATION", limit=100
            )
            observations = obs_response.data if hasattr(obs_response, "data") else []
            for obs in observations:
                usage = _safe_attr(obs, "usage")
                if not usage:
                    continue
                inp = _safe_int(usage, "input", 0) or _safe_int(usage, "promptTokens", 0)
                out = _safe_int(usage, "output", 0) or _safe_int(usage, "completionTokens", 0)
                if inp or out:
                    generations += 1
                tokens_input += inp
                tokens_output += out
        except Exception as exc:
            logger.warning("MetricsCollector: falha ao buscar observations: %s", exc)
        return tokens_input, tokens_output, generations

    def _fetch_otel_summary(self) -> dict[str, Any]:
        """Raspa o exporter Prometheus do Collector e resume as métricas gen_ai.*."""
        if not self._otel_prom_endpoint:
            return {"available": False}
        try:
            resp = httpx.get(self._otel_prom_endpoint, timeout=3.0)
            resp.raise_for_status()
            return _parse_genai_prometheus(resp.text)
        except Exception as exc:
            logger.warning("MetricsCollector: OTel Collector inacessível: %s", exc)
            return {"available": False}


def _empty_metrics(*, available: bool) -> dict[str, Any]:
    return {
        "available": available,
        "stale": False,
        "total_traces": 0,
        "latency": {"avg_ms": 0, "p50_ms": 0, "p95_ms": 0, "sample_size": 0},
        "tokens": {"input": 0, "output": 0, "total": 0, "generations": 0},
        "ttft": {"avg_ms": 0, "p95_ms": 0, "sample_size": 0},
        "ood": {"avg_residual": 0, "max_residual": 0, "sample_size": 0},
        "routing": {"semantic": 0, "llm": 0, "unclassified": 0},
        "tools": {"avg_per_task": 0, "sample_size": 0},
        "clarifications": 0,
        "injection_blocks": 0,
        "error_count": 0,
    }


# Linha Prometheus: nome{labels} valor — só nos importam as gen_ai_*.
_PROM_LINE_RE = re.compile(r'^(gen_ai_[a-z_]+)(?:\{([^}]*)\})?\s+([0-9.e+-]+)\s*$')


def _parse_genai_prometheus(text: str) -> dict[str, Any]:
    """Resume o texto Prometheus do Collector em contadores gen_ai.*."""
    tokens_input = 0.0
    tokens_output = 0.0
    llm_calls = 0.0
    duration_sum = 0.0
    ttft_sum = 0.0
    ttft_count = 0.0

    for line in text.splitlines():
        match = _PROM_LINE_RE.match(line.strip())
        if not match:
            continue
        name, labels, value_raw = match.group(1), match.group(2) or "", match.group(3)
        try:
            value = float(value_raw)
        except ValueError:
            continue
        # O exporter Prometheus adiciona sufixo de unidade ("_seconds") ao
        # histograma de duração; o de tokens ({token}) fica sem sufixo.
        if name == "gen_ai_client_token_usage_sum":
            if 'gen_ai_token_type="input"' in labels:
                tokens_input += value
            elif 'gen_ai_token_type="output"' in labels:
                tokens_output += value
        elif name in ("gen_ai_client_operation_duration_seconds_count", "gen_ai_client_operation_duration_count"):
            llm_calls += value
        elif name in ("gen_ai_client_operation_duration_seconds_sum", "gen_ai_client_operation_duration_sum"):
            duration_sum += value
        elif name in ("gen_ai_server_time_to_first_token_seconds_sum", "gen_ai_server_time_to_first_token_sum"):
            ttft_sum += value
        elif name in ("gen_ai_server_time_to_first_token_seconds_count", "gen_ai_server_time_to_first_token_count"):
            ttft_count += value

    return {
        "available": True,
        "llm_calls": int(llm_calls),
        "tokens": {
            "input": int(tokens_input),
            "output": int(tokens_output),
            "total": int(tokens_input + tokens_output),
        },
        "avg_duration_ms": round(duration_sum / llm_calls * 1000, 1) if llm_calls else 0,
        "ttft_avg_ms": round(ttft_sum / ttft_count * 1000, 1) if ttft_count else 0,
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
