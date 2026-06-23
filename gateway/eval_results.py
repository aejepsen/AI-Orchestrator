"""Agregador de resultados de avaliacao (routing + injection) com cache 60s.

Le JSONs de evals/results/, agrega metricas por tipo e data, retorna
ultimos N runs ordenados por timestamp do filename.
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("gateway")

_CACHE_TTL_S = 60.0
_EVALS_DIR = Path(__file__).resolve().parent.parent / "evals" / "results"
_MAX_RUNS = 20


class _Cache:
    __slots__ = ("data", "fetched_at")

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.fetched_at: float = 0.0

    @property
    def expired(self) -> bool:
        return time.monotonic() - self.fetched_at > _CACHE_TTL_S


class EvalResultsCollector:
    """Coleta e agrega resultados de evals com cache em memoria."""

    def __init__(self) -> None:
        self._cache = _Cache()

    def collect(self) -> dict[str, Any]:
        if not self._cache.expired and self._cache.data:
            return self._cache.data

        try:
            result = self._aggregate()
            self._cache.data = result
            self._cache.fetched_at = time.monotonic()
            return result
        except Exception as exc:
            logger.warning("EvalResultsCollector: falha: %s", exc)
            if self._cache.data:
                return {**self._cache.data, "stale": True}
            return {"available": False}

    def _aggregate(self) -> dict[str, Any]:
        if not _EVALS_DIR.is_dir():
            return {"available": False}

        routing_files = sorted(_EVALS_DIR.glob("routing_*.json"), reverse=True)
        injection_files = sorted(_EVALS_DIR.glob("injection_*.json"), reverse=True)
        semiose_files = sorted(_EVALS_DIR.glob("semiose_*.json"), reverse=True)

        if not routing_files and not injection_files and not semiose_files:
            return {"available": False}

        routing_runs = [_parse_routing(f) for f in routing_files[:_MAX_RUNS]]
        routing_runs = [r for r in routing_runs if r is not None]

        injection_runs = [_parse_injection(f) for f in injection_files[:_MAX_RUNS]]
        injection_runs = [r for r in injection_runs if r is not None]

        # Aggregate routing
        routing_summary = _aggregate_routing(routing_runs)

        # Aggregate injection
        injection_summary = _aggregate_injection(injection_runs)

        # Model comparison
        models = _extract_models(routing_runs, injection_runs)

        # Semiose metrics (latest run only — deterministic offline eval)
        semiose_summary = _parse_semiose(semiose_files[0]) if semiose_files else None

        return {
            "available": True,
            "stale": False,
            "routing": routing_summary,
            "injection": injection_summary,
            "semiose": semiose_summary,
            "models": models,
            "total_runs": len(routing_runs) + len(injection_runs),
        }


def _ts_from_filename(path: Path) -> str:
    """Extrai timestamp legivel do filename: routing_20260614_135727 -> 2026-06-14 13:57:27."""
    m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", path.stem)
    if m:
        return f"{m[1]}-{m[2]}-{m[3]} {m[4]}:{m[5]}:{m[6]}"
    return path.stem


def _parse_routing(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    items = data.get("items", [])
    total = data.get("total", len(items))
    hits = data.get("hits", sum(1 for i in items if i.get("ok")))
    accuracy = hits / total if total > 0 else 0.0

    # Breakdown por dominio
    domain_stats: dict[str, dict[str, int]] = {}
    failed_queries: list[dict[str, Any]] = []

    for item in items:
        for domain in item.get("expect_domains", []):
            if domain not in domain_stats:
                domain_stats[domain] = {"total": 0, "correct": 0}
            domain_stats[domain]["total"] += 1
            if item.get("ok"):
                domain_stats[domain]["correct"] += 1

        if not item.get("ok"):
            failed_queries.append({
                "question": item.get("question", "")[:120],
                "expected": item.get("expect_domains", []),
                "got": item.get("got_domains", []),
            })

    domain_breakdown = {
        d: {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0.0,
        }
        for d, s in sorted(domain_stats.items())
    }

    return {
        "timestamp": _ts_from_filename(path),
        "file": path.name,
        "model": data.get("model", "unknown"),
        "total": total,
        "hits": hits,
        "accuracy": round(accuracy, 4),
        "passed": data.get("passed", accuracy >= data.get("gate", 0.9)),
        "gate": data.get("gate", 0.9),
        "domain_breakdown": domain_breakdown,
        "failed_queries": failed_queries[:10],
        "layer_counts": data.get("layer_counts", {}),
    }


def _parse_injection(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    cases = data.get("cases", [])
    total = data.get("total", len(cases))
    leaks = data.get("leaks", sum(1 for c in cases if c.get("leaked")))
    blocked = total - leaks

    # Confusion matrix para injection:
    # Positivo = "tentativa de injection detectada e bloqueada"
    # Cada caso e uma tentativa de injection (todas sao positivas no ground truth)
    # TP = bloqueou corretamente (leaked=false)
    # FN = deixou vazar (leaked=true)
    # FP/TN nao se aplicam diretamente (nao ha queries "normais" no injection eval)
    # Mas para completude: FP=0, TN=0 (todos os casos sao ataques)
    tp = blocked
    fn = leaks
    fp = 0  # Sem falsos positivos neste eval set
    tn = 0  # Sem true negatives neste eval set

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    leaked_cases = [
        {
            "id": c.get("id", ""),
            "question": c.get("question", "")[:120],
            "bad_route": c.get("bad_route", []),
        }
        for c in cases
        if c.get("leaked")
    ]

    avg_elapsed = sum(c.get("elapsed_s", 0) for c in cases) / len(cases) if cases else 0.0

    return {
        "timestamp": _ts_from_filename(path),
        "file": path.name,
        "total": total,
        "blocked": blocked,
        "leaks": leaks,
        "gate_pass": data.get("gate_pass", leaks == 0),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "leaked_cases": leaked_cases,
        "avg_elapsed_s": round(avg_elapsed, 2),
    }


def _aggregate_routing(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"runs": [], "avg_accuracy": 0.0, "domain_breakdown": {}}

    avg_accuracy = sum(r["accuracy"] for r in runs) / len(runs)

    # Merge domain breakdowns across all runs
    merged_domains: dict[str, dict[str, int]] = {}
    for run in runs:
        for domain, stats in run.get("domain_breakdown", {}).items():
            if domain not in merged_domains:
                merged_domains[domain] = {"total": 0, "correct": 0}
            merged_domains[domain]["total"] += stats["total"]
            merged_domains[domain]["correct"] += stats["correct"]

    domain_agg = {
        d: {
            "total": s["total"],
            "correct": s["correct"],
            "accuracy": round(s["correct"] / s["total"], 4) if s["total"] > 0 else 0.0,
        }
        for d, s in sorted(merged_domains.items())
    }

    return {
        "runs": runs,
        "avg_accuracy": round(avg_accuracy, 4),
        "domain_breakdown": domain_agg,
    }


def _aggregate_injection(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {"runs": [], "avg_f1": 0.0, "total_blocked": 0, "total_leaked": 0}

    avg_f1 = sum(r["f1"] for r in runs) / len(runs)
    total_blocked = sum(r["blocked"] for r in runs)
    total_leaked = sum(r["leaks"] for r in runs)

    return {
        "runs": runs,
        "avg_f1": round(avg_f1, 4),
        "total_blocked": total_blocked,
        "total_leaked": total_leaked,
    }


def _extract_models(
    routing_runs: list[dict[str, Any]],
    injection_runs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Agrupa metricas por modelo para comparacao side-by-side."""
    model_data: dict[str, dict[str, Any]] = {}

    # routing_runs vem ordenado do mais recente para o mais antigo (glob reverse):
    # a 1a ocorrencia de cada modelo e o run mais recente daquele modelo.
    for run in routing_runs:
        model = run.get("model", "unknown")
        if model not in model_data:
            model_data[model] = {"model": model, "routing_runs": 0, "routing_accuracy_sum": 0.0, "last_routing_accuracy": run["accuracy"], "injection_runs": 0, "injection_f1_sum": 0.0}
        model_data[model]["routing_runs"] += 1
        model_data[model]["routing_accuracy_sum"] += run["accuracy"]

    for run in injection_runs:
        # Injection evals nao tem campo model; vincula ao modelo mais recente de routing
        pass

    result = []
    for m in model_data.values():
        entry: dict[str, Any] = {"model": m["model"]}
        if m["routing_runs"] > 0:
            entry["avg_routing_accuracy"] = round(m["routing_accuracy_sum"] / m["routing_runs"], 4)
            entry["last_routing_accuracy"] = round(m["last_routing_accuracy"], 4)
            entry["routing_runs"] = m["routing_runs"]
        if m["injection_runs"] > 0:
            entry["avg_injection_f1"] = round(m["injection_f1_sum"] / m["injection_runs"], 4)
            entry["injection_runs"] = m["injection_runs"]
        result.append(entry)

    return sorted(result, key=lambda x: x.get("avg_routing_accuracy", 0), reverse=True)


def _parse_semiose(path: Path) -> dict[str, Any] | None:
    """Extrai metricas do eval de Semiose mais recente.

    O eval e offline/deterministico — o run mais recente reflete o estado atual
    do pipeline (harness + KG). Retorna as metricas agrupadas por camada (A/B/C).
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    metrics = data.get("metrics", {})
    gates = data.get("gates", {})
    modes = data.get("modes", {})

    def _g(name: str, raw: bool = False) -> dict[str, Any] | None:
        val = metrics.get(name)
        if val is None or (val == 0 and not raw):
            return None
        if raw:
            return {"value": val, "gate": None, "passed": None}
        gate_info = gates.get(name, {})
        return {
            "value": round(val, 4),
            "gate": gate_info.get("gate"),
            "passed": gate_info.get("passed"),
        }

    return {
        "timestamp": _ts_from_filename(path),
        "file": path.name,
        "modes": modes,
        "total_cases": metrics.get("total_cases", 0),
        "layer_a": {
            "label": "Camada A — Enricher",
            "entity_propagation_f1": _g("entity_propagation_f1"),
            "entity_propagation_precision": _g("entity_propagation_precision"),
            "entity_propagation_recall": _g("entity_propagation_recall"),
            "false_enrichment_rate": _g("false_enrichment_rate"),
            "topic_switch_accuracy": _g("topic_switch_accuracy"),
            "contextual_drift_score": _g("contextual_drift_score"),
            "enrichment_cosine_preservation": _g("enrichment_cosine_preservation"),
            "enrichment_bertscore": _g("enrichment_bertscore"),
            "cases_enriched": metrics.get("cases_enriched", 0),
        },
        "layer_b": {
            "label": "Camada B — Knowledge Graph",
            "geu": _g("geu"),
            "cdrr": _g("cdrr"),
            "relation_validity_at_5": _g("relation_validity_at_5"),
            "graph_latency_budget": _g("graph_latency_budget"),
        },
        "layer_b_proactive": {
            "label": "Camada B+ — Graph Health (Proativo)",
            "entity_coverage": _g("entity_coverage"),
            "graph_freshness": _g("graph_freshness"),
            "orphan_rate": _g("orphan_rate"),
            "cross_domain_density": _g("cross_domain_density"),
            "domain_entropy": _g("domain_entropy"),
            "total_nodes": _g("total_nodes", raw=True),
            "total_edges": _g("total_edges", raw=True),
        },
        "layer_c": {
            "label": "Camada C — Re-ranking",
            "contextual_gain_ratio": _g("contextual_gain_ratio"),
            "boost_precision": _g("boost_precision"),
            "exact_match_routing": _g("exact_match_routing"),
            "exact_match_routing_no_context": _g("exact_match_routing_no_context"),
            "routing_failure_rate": _g("routing_failure_rate"),
        },
    }
