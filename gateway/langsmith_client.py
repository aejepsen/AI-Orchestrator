"""LangSmith live metrics — query traces via SDK.

Busca dados reais do LangSmith Cloud (não estimados) para o dashboard.
A chave LANGSMITH_API_KEY autentica automaticamente. Sem chave → fallback para estimativas.

Métricas extraídas:
  - Latência média e P95 (extraído diretamente das runs)
  - Token usage real (prompt_tokens + completion_tokens)
  - Task success rate (% de runs sem erro)
  - Cost-per-task calculado com pricing do modelo local (GPU Ollama)
  - Métricas por run_type (llm, chain, tool)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger("gateway")

# Preço estimado por 1K tokens para Ollama local (GPU RTX 3060, R$0.70/kWh)
# ~200W × R$0.000194/s = R$0.0000389/s → ~R$0.001 por tarefa de 25s
_LOCAL_PRICE_PER_1K = 0.0001  # R$ por 1K tokens (GPU local depreciada)


def _get_client():
    """Lazy import do langsmith Client. Fallback None se chave ausente."""
    try:
        from langsmith import Client as LsClient
        api_key = os.environ.get("LANGSMITH_API_KEY", "")
        if not api_key or not api_key.startswith("ls"):
            logger.debug("LangSmith: sem API key válida, usando fallback")
            return None
        return LsClient()
    except ImportError:
        return None
    except Exception as exc:
        logger.debug("LangSmith: %s", exc)
        return None


def fetch_live_metrics(project: str = "ai-orchestrator", hours: int = 24) -> dict[str, Any] | None:
    """Busca métricas das últimas N horas de traces do LangSmith.

    Returns:
        dict com latency_avg, latency_p95, token_usage_avg, task_success_rate,
        cost_per_task, total_traces — ou None se LangSmith indisponível.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        start = datetime.now(timezone.utc) - timedelta(hours=hours)
        runs = list(client.list_runs(
            project_name=project,
            start_time=start,
            limit=200,
        ))
    except Exception as exc:
        logger.debug("LangSmith list_runs: %s", exc)
        return None

    if not runs:
        return None

    latencies: list[float] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    total_tokens: list[int] = []
    errors = 0
    run_types: dict[str, int] = {}

    for run in runs:
        rt = getattr(run, "run_type", "unknown") or "unknown"
        run_types[rt] = run_types.get(rt, 0) + 1

        if run.error:
            errors += 1

        # Latência em segundos
        if run.latency is not None:
            latencies.append(run.latency)

        # Token usage (prompt_tokens, completion_tokens, total_tokens)
        if run.prompt_tokens is not None:
            prompt_tokens.append(run.prompt_tokens)
        if run.completion_tokens is not None:
            completion_tokens.append(run.completion_tokens)
        if run.total_tokens is not None:
            total_tokens.append(run.total_tokens)

    total = len(runs)

    # Estatísticas
    def _p95(values: list[float]) -> float:
        if not values:
            return 0.0
        s = sorted(values)
        idx = int(len(s) * 0.95)
        return s[min(idx, len(s) - 1)]

    def _avg(values: list) -> float:
        return sum(values) / len(values) if values else 0.0

    latency_avg = _avg(latencies)
    latency_p95_val = _p95(latencies)
    avg_prompt = _avg(prompt_tokens)
    avg_completion = _avg(completion_tokens)
    avg_total = _avg(total_tokens)
    task_success = (total - errors) / total if total > 0 else 0.0

    # Cost-per-task: (prompt_tokens + 3 × completion_tokens) × preço por 1K
    # O fator 3× em completion reflete que geração é mais cara que prompt
    cost_per_task = (avg_prompt + 3 * avg_completion) * _LOCAL_PRICE_PER_1K / 1000

    return {
        "source": "langsmith_live",
        "project": project,
        "window_hours": hours,
        "total_traces": total,
        "run_types": run_types,
        "latency_avg_s": round(latency_avg, 3),
        "latency_p95_s": round(latency_p95_val, 3),
        "prompt_tokens_avg": round(avg_prompt, 0),
        "completion_tokens_avg": round(avg_completion, 0),
        "total_tokens_avg": round(avg_total, 0),
        "task_success_rate": round(task_success, 4),
        "cost_per_task_brl": round(cost_per_task, 6),
        "errors": errors,
        "successes": total - errors,
    }
