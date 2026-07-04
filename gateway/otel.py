"""OpenTelemetry — GenAI Semantic Conventions (gen_ai.*).

Instrumentação manual seguindo as GenAI Semantic Conventions do OTel:
um span por chamada LLM com gen_ai.request.model, gen_ai.usage.*,
gen_ai.response.finish_reasons e gen_ai.request.temperature; métricas
gen_ai.client.token.usage, gen_ai.client.operation.duration e
gen_ai.server.time_to_first_token exportadas via OTLP HTTP pro Collector.

O Collector faz fan-out: traces → Phoenix; métricas → Prometheus (:8889),
que o gateway raspa em /metrics como fonte independente do Langfuse.

Degradação graceful: OTEL_ENABLED=0, SDK ausente ou Collector fora →
record_llm_call vira no-op; o request nunca falha por observabilidade.
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger("gateway")

_tracer: Any = None
_token_usage_hist: Any = None
_duration_hist: Any = None
_ttft_hist: Any = None

# Ollama serve os modelos locais; valor de gen_ai.system nos spans/métricas.
_GEN_AI_SYSTEM = "ollama"


def init(settings: Any) -> bool:
    """Inicializa providers OTel. Chamar UMA vez no boot do gateway."""
    global _tracer, _token_usage_hist, _duration_hist, _ttft_hist
    if not getattr(settings, "otel_enabled", False):
        return False
    try:
        from opentelemetry import metrics, trace
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        endpoint = settings.otel_exporter_endpoint.rstrip("/")
        resource = Resource.create({"service.name": settings.otel_service_name})

        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces"))
        )
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer("gateway.genai")

        reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=f"{endpoint}/v1/metrics"),
            export_interval_millis=10_000,
        )
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))
        meter = metrics.get_meter("gateway.genai")
        _token_usage_hist = meter.create_histogram(
            "gen_ai.client.token.usage", unit="{token}",
            description="Tokens consumidos por chamada, por gen_ai.token.type (input/output).",
        )
        _duration_hist = meter.create_histogram(
            "gen_ai.client.operation.duration", unit="s",
            description="Duração da operação GenAI (chat/embeddings).",
        )
        _ttft_hist = meter.create_histogram(
            "gen_ai.server.time_to_first_token", unit="s",
            description="Tempo até o primeiro token no streaming.",
        )
        logger.info("OTel GenAI ativo: OTLP → %s", endpoint)
        return True
    except Exception as exc:
        logger.warning("OTel indisponível (instrumentação desativada): %s", exc)
        _tracer = None
        return False


def record_llm_call(
    *,
    operation: str,
    model: str,
    duration_s: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    temperature: float | None = None,
    finish_reason: str | None = None,
    ttft_s: float | None = None,
    error: str | None = None,
) -> None:
    """Registra span gen_ai.* + métricas de uma chamada LLM já concluída."""
    if _tracer is None:
        return
    try:
        attrs: dict[str, Any] = {
            "gen_ai.operation.name": operation,
            "gen_ai.system": _GEN_AI_SYSTEM,
            "gen_ai.request.model": model,
        }
        if temperature is not None:
            attrs["gen_ai.request.temperature"] = temperature
        if finish_reason:
            attrs["gen_ai.response.finish_reasons"] = [finish_reason]
        if input_tokens:
            attrs["gen_ai.usage.input_tokens"] = input_tokens
        if output_tokens:
            attrs["gen_ai.usage.output_tokens"] = output_tokens
        if ttft_s is not None:
            attrs["gen_ai.server.time_to_first_token"] = round(ttft_s, 4)
        if error:
            attrs["error.type"] = error

        # A chamada já terminou: span retroativo com start/end explícitos.
        end_ns = time.time_ns()
        start_ns = end_ns - int(duration_s * 1e9)
        span = _tracer.start_span(f"{operation} {model}", start_time=start_ns, attributes=attrs)
        if error:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.ERROR, error))
        span.end(end_time=end_ns)

        metric_attrs = {
            "gen_ai.operation.name": operation,
            "gen_ai.system": _GEN_AI_SYSTEM,
            "gen_ai.request.model": model,
        }
        if _duration_hist:
            _duration_hist.record(duration_s, metric_attrs)
        if _token_usage_hist:
            if input_tokens:
                _token_usage_hist.record(input_tokens, {**metric_attrs, "gen_ai.token.type": "input"})
            if output_tokens:
                _token_usage_hist.record(output_tokens, {**metric_attrs, "gen_ai.token.type": "output"})
        if _ttft_hist and ttft_s is not None:
            _ttft_hist.record(ttft_s, metric_attrs)
    except Exception:  # noqa: BLE001 — observabilidade nunca derruba o request
        pass
