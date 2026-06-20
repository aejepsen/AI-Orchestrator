"""Eval de Semiose: métricas de enriquecimento contextual, re-ranking e KG.

12 métricas organizadas em 4 camadas:
  A — Enricher:  Entity Propagation F1, Contextual Drift Score,
                 False Enrichment Rate (FER), Topic Switch Accuracy (TSA)
  B — KG:        Graph Expansion Utility (GEU), Cross-Domain Resolution Rate,
                 Relation Precision@5, Graph Latency Budget
  C — Re-rank:   Contextual Gain Ratio (CGR), Boost Precision
  E2E:           F1-Score Micro (routing), Semantic Preservation (BERTScore)

Uso:
    python -m evals.eval_semiose                          # offline (sem LLM/Qdrant)
    python -m evals.eval_semiose --semantic               # com Qdrant (leave-one-out)
    python -m evals.eval_semiose --full                   # E2E com LLM
    python -m evals.eval_semiose --neo4j                  # inclui métricas KG
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gateway.config import load_settings  # noqa: E402
from gateway.query_enricher import (  # noqa: E402
    ContextSignal,
    enrich_query,
    gather_signals,
)
from gateway.router import RoutePlan, _DOMAIN_KEYWORDS, classify_intent  # noqa: E402

GOLDEN = ROOT / "evals" / "golden_semiose.jsonl"
GOLDEN_ROUTING = ROOT / "evals" / "golden_routing.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"

# ── Gates (targets da tabela-resumo) ─────────────────────────────────────

GATES = {
    "entity_propagation_f1": 0.70,
    "contextual_drift_score": 0.10,  # máximo (lower is better)
    "false_enrichment_rate": 0.05,   # máximo (lower is better)
    "topic_switch_accuracy": 0.95,
    "contextual_gain_ratio": 0.30,
    "boost_precision": 0.90,
    "f1_micro_routing": 0.90,
    "geu": 0.60,
    "cdrr": 0.40,
    "graph_latency_budget": 1.30,    # máximo (lower is better)
    "relation_precision_at_5": 0.80,
}


def load_golden(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ── Helpers ──────────────────────────────────────────────────────────────


def _build_state(record: dict) -> dict[str, Any]:
    """Constrói GraphState simulado a partir de um record do golden."""
    state: dict[str, Any] = {"sanitized": record["question"]}

    if record.get("prev_question"):
        state["history"] = [
            {"role": "user", "content": record["prev_question"]},
            {"role": "assistant", "content": "(resposta anterior)"},
        ]

    if record.get("prev_domains"):
        state["_last_route"] = {"domains": record["prev_domains"]}

    return state


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


# ── Dataclasses de resultado ─────────────────────────────────────────────


@dataclass
class CaseResult:
    id: str
    question: str
    expect_domains: list[str]
    expect_enriched: bool
    expect_topic_switch: bool
    expect_entities: list[str]
    # Medições
    got_enriched: bool = False
    enriched_query: str = ""
    original_query: str = ""
    got_entities: list[str] = field(default_factory=list)
    got_topic_switch: bool = False
    # Routing (preenchido se --semantic ou --full)
    route_with_context: RoutePlan | None = None
    route_without_context: RoutePlan | None = None
    # Drift
    drift_score: float = 0.0
    # Timing
    elapsed_s: float = 0.0


@dataclass
class SemioseMetrics:
    """Todas as 12 métricas consolidadas."""

    # Camada A
    entity_propagation_f1: float = 0.0
    entity_propagation_precision: float = 0.0
    entity_propagation_recall: float = 0.0
    contextual_drift_score: float = 0.0
    false_enrichment_rate: float = 0.0
    topic_switch_accuracy: float = 0.0
    # Camada B
    geu: float = 0.0
    cdrr: float = 0.0
    relation_precision_at_5: float = 0.0
    graph_latency_budget: float = 0.0
    # Camada C
    contextual_gain_ratio: float = 0.0
    boost_precision: float = 0.0
    # E2E
    f1_micro_routing: float = 0.0
    f1_micro_routing_no_context: float = 0.0
    semantic_preservation: float = 0.0
    # Meta
    total_cases: int = 0
    cases_enriched: int = 0
    cases_topic_switch: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in self.__dict__.items()}

    def check_gates(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for metric, gate in GATES.items():
            value = getattr(self, metric, None)
            if value is None:
                continue
            # Métricas "lower is better"
            if metric in ("contextual_drift_score", "false_enrichment_rate", "graph_latency_budget"):
                passed = value <= gate
            else:
                passed = value >= gate
            results[metric] = {"value": round(value, 4), "gate": gate, "passed": passed}
        return results


# ── Camada A: Enricher ───────────────────────────────────────────────────


def eval_enricher(records: list[dict], embedder: Any = None) -> tuple[list[CaseResult], dict]:
    """Avalia Camada A: enricher determinístico (sem LLM)."""
    results: list[CaseResult] = []
    drift_scores: list[float] = []

    for record in records:
        started = time.monotonic()
        state = _build_state(record)
        signals = gather_signals(state, spacy_enabled=False)  # regex only para eval determinístico
        enriched_query, was_enriched = enrich_query(record["question"], signals)

        # Topic switch detection
        got_topic_switch = False
        if signals.last_domain and record.get("prev_domains"):
            from gateway.query_enricher import _has_strong_conflict
            got_topic_switch = _has_strong_conflict(record["question"], signals.last_domain)

        # Drift score (se embedder disponível)
        drift = 0.0
        if embedder and was_enriched:
            vecs = embedder.embed([record["question"], enriched_query])
            drift = 1.0 - _cosine_similarity(vecs[0], vecs[1])
        if was_enriched:
            drift_scores.append(drift)

        elapsed = round(time.monotonic() - started, 4)

        case = CaseResult(
            id=record["id"],
            question=record["question"],
            expect_domains=record["expect_domains"],
            expect_enriched=record["expect_enriched"],
            expect_topic_switch=record["expect_topic_switch"],
            expect_entities=record.get("expect_entities", []),
            got_enriched=was_enriched,
            enriched_query=enriched_query,
            original_query=record["question"],
            got_entities=signals.recent_entities,
            got_topic_switch=got_topic_switch,
            drift_score=drift,
            elapsed_s=elapsed,
        )
        results.append(case)

    # ── Entity Propagation F1 ────────────────────────────────────────
    tp, fp, fn = 0, 0, 0
    for case in results:
        expected = set(case.expect_entities)
        got = set(case.got_entities)
        tp += len(expected & got)
        fp += len(got - expected)
        fn += len(expected - got)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # ── False Enrichment Rate ────────────────────────────────────────
    # Enricher ativou quando NÃO deveria (topic_switch=true ou first turn)
    false_enrichments = sum(
        1 for c in results
        if c.got_enriched and not c.expect_enriched
    )
    total_enriched = sum(1 for c in results if c.got_enriched)
    fer = false_enrichments / total_enriched if total_enriched > 0 else 0.0

    # ── Topic Switch Accuracy ────────────────────────────────────────
    switch_cases = [c for c in results if c.expect_topic_switch or c.got_topic_switch]
    if switch_cases:
        tsa_correct = sum(1 for c in switch_cases if c.got_topic_switch == c.expect_topic_switch)
        tsa = tsa_correct / len(switch_cases)
    else:
        tsa = 1.0

    # ── Contextual Drift Score (média) ───────────────────────────────
    avg_drift = float(np.mean(drift_scores)) if drift_scores else 0.0

    return results, {
        "entity_propagation_f1": f1,
        "entity_propagation_precision": precision,
        "entity_propagation_recall": recall,
        "contextual_drift_score": avg_drift,
        "false_enrichment_rate": fer,
        "topic_switch_accuracy": tsa,
        "total_cases": len(results),
        "cases_enriched": total_enriched,
        "false_enrichments": false_enrichments,
        "switch_cases": len(switch_cases),
    }


# ── Camada C: Re-ranking contextual ─────────────────────────────────────


def eval_reranking(
    records: list[dict],
    semantic_router: Any,
    embedder: Any,
) -> dict:
    """Avalia Camada C: boost aditivo no SemanticRouter.

    Compara roteamento COM vs SEM context_domain (leave-one-out).
    """
    hits_with = 0
    hits_without = 0
    boost_flips = 0
    boost_correct_flips = 0
    total = 0

    for record in records:
        expected = set(record["expect_domains"])
        context_domain = record["prev_domains"][0] if record.get("prev_domains") else None

        # Rota SEM contexto
        route_no_ctx = semantic_router.route(
            record["question"],
            exclude_question=record["question"],
        )
        domains_no_ctx = set(route_no_ctx.domains) if route_no_ctx else set()

        # Rota COM contexto
        route_with_ctx = semantic_router.route(
            record["question"],
            exclude_question=record["question"],
            context_domain=context_domain,
        )
        domains_with_ctx = set(route_with_ctx.domains) if route_with_ctx else set()

        ok_without = domains_no_ctx == expected
        ok_with = domains_with_ctx == expected

        hits_without += ok_without
        hits_with += ok_with

        # Boost flipped top-1?
        if domains_with_ctx != domains_no_ctx:
            boost_flips += 1
            if ok_with:
                boost_correct_flips += 1

        total += 1

    acc_with = hits_with / total if total else 0.0
    acc_without = hits_without / total if total else 0.0

    # Contextual Gain Ratio
    if acc_without < 1.0:
        cgr = (acc_with - acc_without) / (1.0 - acc_without)
    else:
        cgr = 0.0

    # Boost Precision
    bp = boost_correct_flips / boost_flips if boost_flips > 0 else 1.0

    return {
        "contextual_gain_ratio": cgr,
        "boost_precision": bp,
        "f1_micro_routing": acc_with,
        "f1_micro_routing_no_context": acc_without,
        "boost_flips": boost_flips,
        "boost_correct_flips": boost_correct_flips,
        "total": total,
    }


# ── Camada B: Knowledge Graph ───────────────────────────────────────────


def eval_knowledge_graph(records: list[dict], kg: Any) -> dict:
    """Avalia Camada B: Graph Expansion Utility e métricas derivadas.

    Requer Neo4j rodando com dados do seed.
    """
    expand_calls = 0
    useful_expansions = 0
    cross_domain_useful = 0
    latencies_with: list[float] = []
    latencies_without: list[float] = []

    for record in records:
        entities = record.get("expect_entities", [])
        if not entities:
            # Sem entidades → sem expand_context
            t0 = time.monotonic()
            # Simula latência sem KG
            latencies_without.append(time.monotonic() - t0)
            continue

        for entity in entities:
            # Expand com KG
            t0 = time.monotonic()
            result = kg.expand(entity, "produto")  # type default
            elapsed = time.monotonic() - t0
            latencies_with.append(elapsed)

            expand_calls += 1
            related = result.get("body", {}).get("related", [])

            if related:
                useful_expansions += 1
                # Cross-domain: related entity de domínio diferente
                entity_domain = record["expect_domains"][0] if record["expect_domains"] else ""
                if any(r.get("domain") != entity_domain for r in related):
                    cross_domain_useful += 1

            # Baseline sem KG
            t0 = time.monotonic()
            latencies_without.append(time.monotonic() - t0)

    geu = useful_expansions / expand_calls if expand_calls else 0.0
    cdrr = cross_domain_useful / expand_calls if expand_calls else 0.0

    avg_with = float(np.mean(latencies_with)) if latencies_with else 0.0
    avg_without = float(np.mean(latencies_without)) if latencies_without else 0.001
    latency_budget = avg_with / avg_without if avg_without > 0 else 1.0

    return {
        "geu": geu,
        "cdrr": cdrr,
        "graph_latency_budget": latency_budget,
        "relation_precision_at_5": geu,  # proxy: useful = precision em top results
        "expand_calls": expand_calls,
        "useful_expansions": useful_expansions,
        "cross_domain_useful": cross_domain_useful,
        "avg_latency_with_ms": round(avg_with * 1000, 2),
        "avg_latency_without_ms": round(avg_without * 1000, 2),
    }


# ── E2E: Semantic Preservation (BERTScore) ──────────────────────────────


def eval_semantic_preservation(cases: list[CaseResult], embedder: Any) -> float:
    """BERTScore simplificado: cosseno entre query original e enriquecida.

    Mede se o enriquecimento preserva o significado (Δ > 0 = bom).
    Usa SBERT embeddings (já disponíveis no pipeline).
    """
    if not embedder:
        return 0.0

    enriched_cases = [c for c in cases if c.got_enriched]
    if not enriched_cases:
        return 0.0

    scores: list[float] = []
    for case in enriched_cases:
        vecs = embedder.embed([case.original_query, case.enriched_query])
        sim = _cosine_similarity(vecs[0], vecs[1])
        scores.append(sim)

    return float(np.mean(scores))


# ── Formatação ───────────────────────────────────────────────────────────


def _gate_status(metric: str, value: float) -> str:
    gate = GATES.get(metric)
    if gate is None:
        return ""
    if metric in ("contextual_drift_score", "false_enrichment_rate", "graph_latency_budget"):
        passed = value <= gate
        return f"{'PASS' if passed else 'FAIL'} (≤ {gate})"
    passed = value >= gate
    return f"{'PASS' if passed else 'FAIL'} (≥ {gate})"


def print_report(metrics: SemioseMetrics, cases: list[CaseResult]) -> None:
    """Imprime relatório formatado das métricas."""
    print("\n" + "=" * 72)
    print("  EVAL SEMIOSE — Métricas de Enriquecimento Contextual")
    print("=" * 72)

    print(f"\n  Total de casos: {metrics.total_cases}")
    print(f"  Casos enriquecidos: {metrics.cases_enriched}")
    print(f"  Trocas de tópico: {metrics.cases_topic_switch}")

    print("\n── Camada A — Enricher " + "─" * 49)
    print(f"  Entity Propagation F1:     {metrics.entity_propagation_f1:.4f}  "
          f"{_gate_status('entity_propagation_f1', metrics.entity_propagation_f1)}")
    print(f"    Precision:               {metrics.entity_propagation_precision:.4f}")
    print(f"    Recall:                  {metrics.entity_propagation_recall:.4f}")
    print(f"  Contextual Drift Score:    {metrics.contextual_drift_score:.4f}  "
          f"{_gate_status('contextual_drift_score', metrics.contextual_drift_score)}")
    print(f"  False Enrichment Rate:     {metrics.false_enrichment_rate:.4f}  "
          f"{_gate_status('false_enrichment_rate', metrics.false_enrichment_rate)}")
    print(f"  Topic Switch Accuracy:     {metrics.topic_switch_accuracy:.4f}  "
          f"{_gate_status('topic_switch_accuracy', metrics.topic_switch_accuracy)}")

    print("\n── Camada B — Knowledge Graph " + "─" * 42)
    if metrics.geu > 0 or metrics.cdrr > 0:
        print(f"  Graph Expansion Utility:   {metrics.geu:.4f}  "
              f"{_gate_status('geu', metrics.geu)}")
        print(f"  Cross-Domain Res. Rate:    {metrics.cdrr:.4f}  "
              f"{_gate_status('cdrr', metrics.cdrr)}")
        print(f"  Relation Precision@5:      {metrics.relation_precision_at_5:.4f}  "
              f"{_gate_status('relation_precision_at_5', metrics.relation_precision_at_5)}")
        print(f"  Graph Latency Budget:      {metrics.graph_latency_budget:.4f}x  "
              f"{_gate_status('graph_latency_budget', metrics.graph_latency_budget)}")
    else:
        print("  (Neo4j não disponível — use --neo4j para ativar)")

    print("\n── Camada C — Re-ranking Contextual " + "─" * 36)
    if metrics.contextual_gain_ratio > 0 or metrics.boost_precision > 0:
        print(f"  Contextual Gain Ratio:     {metrics.contextual_gain_ratio:.4f}  "
              f"{_gate_status('contextual_gain_ratio', metrics.contextual_gain_ratio)}")
        print(f"  Boost Precision:           {metrics.boost_precision:.4f}  "
              f"{_gate_status('boost_precision', metrics.boost_precision)}")
    else:
        print("  (Qdrant não disponível — use --semantic para ativar)")

    print("\n── End-to-End " + "─" * 58)
    if metrics.f1_micro_routing > 0:
        print(f"  F1-Score Micro (routing):  {metrics.f1_micro_routing:.4f}  "
              f"{_gate_status('f1_micro_routing', metrics.f1_micro_routing)}")
        print(f"    Sem contexto:            {metrics.f1_micro_routing_no_context:.4f}")
        delta = metrics.f1_micro_routing - metrics.f1_micro_routing_no_context
        print(f"    Delta:                   {delta:+.4f}")
    if metrics.semantic_preservation > 0:
        print(f"  Semantic Preservation:     {metrics.semantic_preservation:.4f}")

    # Gate summary
    gates = metrics.check_gates()
    active_gates = {k: v for k, v in gates.items() if v["value"] != 0}
    if active_gates:
        passed = sum(1 for v in active_gates.values() if v["passed"])
        total = len(active_gates)
        print(f"\n── Gates: {passed}/{total} PASS " + "─" * 42)
        for metric, info in sorted(active_gates.items()):
            status = "✓" if info["passed"] else "✗"
            print(f"  {status} {metric}: {info['value']:.4f} (gate {info['gate']})")

    # Failures detail
    failures = [c for c in cases if c.got_enriched != c.expect_enriched or c.got_topic_switch != c.expect_topic_switch]
    if failures:
        print(f"\n── Falhas ({len(failures)}) " + "─" * 50)
        for c in failures:
            reasons = []
            if c.got_enriched != c.expect_enriched:
                reasons.append(f"enriched={c.got_enriched} (esperado {c.expect_enriched})")
            if c.got_topic_switch != c.expect_topic_switch:
                reasons.append(f"topic_switch={c.got_topic_switch} (esperado {c.expect_topic_switch})")
            print(f"  [{c.id}] {c.question[:50]}...")
            print(f"          {', '.join(reasons)}")

    print("\n" + "=" * 72)


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval de Semiose — 12 métricas contextuais")
    parser.add_argument("--semantic", action="store_true", help="ativa Camada C (Qdrant leave-one-out)")
    parser.add_argument("--full", action="store_true", help="E2E com LLM + semantic + drift")
    parser.add_argument("--neo4j", action="store_true", help="ativa Camada B (requer Neo4j rodando)")
    parser.add_argument("--limit", type=int, default=None, help="avalia só N primeiros casos")
    parser.add_argument("--golden", default=str(GOLDEN), help="path do golden set")
    parser.add_argument(
        "--split",
        choices=["dev", "train", "val", "all"],
        default="dev",
        help="split do dataset: dev (30), train (80), val (40), all (150)",
    )
    args = parser.parse_args()

    # Resolve split → golden files
    golden_dir = Path(args.golden).parent
    if args.split == "dev":
        records = load_golden(Path(args.golden))
    elif args.split == "train":
        records = load_golden(golden_dir / "golden_semiose_train.jsonl")
    elif args.split == "val":
        records = load_golden(golden_dir / "golden_semiose_val.jsonl")
    elif args.split == "all":
        records = (
            load_golden(Path(args.golden))
            + load_golden(golden_dir / "golden_semiose_train.jsonl")
            + load_golden(golden_dir / "golden_semiose_val.jsonl")
        )

    if args.limit:
        records = records[: args.limit]

    settings = load_settings()
    embedder = None
    semantic = None
    kg = None

    # Embedder para drift score e semantic preservation
    if args.full or args.semantic:
        from gateway.embedder import SBERTEmbedder  # noqa: E402
        embedder = SBERTEmbedder(
            model_name=settings.sbert_model,
            cache_dir=settings.sbert_cache_dir,
        )

    # Semantic Router para Camada C
    if args.semantic or args.full:
        from gateway.semantic_router import SemanticRouter  # noqa: E402
        semantic = SemanticRouter(
            settings.qdrant_url,
            embedder,
            examples_path=str(GOLDEN_ROUTING),
            threshold=settings.semantic_threshold,
            top_k=settings.semantic_top_k,
        )
        semantic.ensure_ready()

    # Knowledge Graph para Camada B
    if args.neo4j:
        from gateway.knowledge_graph import KnowledgeGraph  # noqa: E402
        kg = KnowledgeGraph(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    # ── Camada A ─────────────────────────────────────────────────────
    print("Avaliando Camada A (Enricher)...")
    cases, enricher_metrics = eval_enricher(records, embedder=embedder)

    metrics = SemioseMetrics(
        entity_propagation_f1=enricher_metrics["entity_propagation_f1"],
        entity_propagation_precision=enricher_metrics["entity_propagation_precision"],
        entity_propagation_recall=enricher_metrics["entity_propagation_recall"],
        contextual_drift_score=enricher_metrics["contextual_drift_score"],
        false_enrichment_rate=enricher_metrics["false_enrichment_rate"],
        topic_switch_accuracy=enricher_metrics["topic_switch_accuracy"],
        total_cases=enricher_metrics["total_cases"],
        cases_enriched=enricher_metrics["cases_enriched"],
        cases_topic_switch=len([c for c in cases if c.expect_topic_switch]),
    )

    # ── Camada C ─────────────────────────────────────────────────────
    if semantic:
        print("Avaliando Camada C (Re-ranking contextual)...")
        rerank_metrics = eval_reranking(records, semantic, embedder)
        metrics.contextual_gain_ratio = rerank_metrics["contextual_gain_ratio"]
        metrics.boost_precision = rerank_metrics["boost_precision"]
        metrics.f1_micro_routing = rerank_metrics["f1_micro_routing"]
        metrics.f1_micro_routing_no_context = rerank_metrics["f1_micro_routing_no_context"]

    # ── Camada B ─────────────────────────────────────────────────────
    if kg:
        print("Avaliando Camada B (Knowledge Graph)...")
        kg_metrics = eval_knowledge_graph(records, kg)
        metrics.geu = kg_metrics["geu"]
        metrics.cdrr = kg_metrics["cdrr"]
        metrics.relation_precision_at_5 = kg_metrics["relation_precision_at_5"]
        metrics.graph_latency_budget = kg_metrics["graph_latency_budget"]

    # ── E2E: Semantic Preservation ───────────────────────────────────
    if embedder:
        print("Avaliando Semantic Preservation...")
        metrics.semantic_preservation = eval_semantic_preservation(cases, embedder)

    # ── Report ───────────────────────────────────────────────────────
    print_report(metrics, cases)

    # ── Persist results ──────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"semiose_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "golden": str(args.golden),
        "modes": {
            "semantic": args.semantic or args.full,
            "neo4j": args.neo4j,
            "full": args.full,
        },
        "metrics": metrics.as_dict(),
        "gates": metrics.check_gates(),
        "cases": [
            {
                "id": c.id,
                "question": c.question,
                "expect_enriched": c.expect_enriched,
                "got_enriched": c.got_enriched,
                "expect_topic_switch": c.expect_topic_switch,
                "got_topic_switch": c.got_topic_switch,
                "expect_entities": c.expect_entities,
                "got_entities": c.got_entities,
                "drift_score": c.drift_score,
                "elapsed_s": c.elapsed_s,
            }
            for c in cases
        ],
    }
    out.write_text(json.dumps(result_data, ensure_ascii=False, indent=2))
    print(f"\nResultados salvos em {out}")

    # Exit code: 0 se todos os gates ativos passam
    active_gates = {k: v for k, v in metrics.check_gates().items() if v["value"] != 0}
    all_passed = all(v["passed"] for v in active_gates.values()) if active_gates else True
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
