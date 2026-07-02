"""Testes das métricas agregadas do eval de domínios (observability-plan)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.eval_domains import aggregate_metrics  # noqa: E402


def _task(stop_reason: str = "answer", tools: int = 2) -> dict:
    return {"stop_reason": stop_reason, "tool_trace": [{"name": f"t{i}"} for i in range(tools)]}


def test_vazio_retorna_zeros() -> None:
    metrics = aggregate_metrics([])
    assert metrics == {
        "task_success_rate": 0.0,
        "tools_per_task_mean": 0.0,
        "tools_per_task_p95": 0.0,
    }


def test_task_success_rate() -> None:
    tasks = [_task(), _task(), _task("max_iters"), _task("timeout")]
    assert aggregate_metrics(tasks)["task_success_rate"] == 50.0


def test_tools_per_task_media_e_p95() -> None:
    tasks = [_task(tools=n) for n in (1, 2, 2, 3, 10)]
    metrics = aggregate_metrics(tasks)
    assert metrics["tools_per_task_mean"] == 3.6
    assert metrics["tools_per_task_p95"] == 10.0


def test_trace_ausente_conta_zero_tools() -> None:
    metrics = aggregate_metrics([{"stop_reason": "answer", "tool_trace": None}])
    assert metrics["tools_per_task_mean"] == 0.0
    assert metrics["task_success_rate"] == 100.0
