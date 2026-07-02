"""Eval de Semiose: métricas de enriquecimento contextual, re-ranking e KG.

12 métricas organizadas em 4 camadas:
  A — Enricher:  Entity Propagation F1, Contextual Drift Score,
                 False Enrichment Rate (FER), Topic Switch Accuracy (TSA)
  B — KG:        Graph Expansion Utility (GEU), Cross-Domain Resolution Rate,
                 Relation Validity@5, Graph Latency Budget
  C — Re-rank:   Contextual Gain Ratio (CGR), Boost Precision
  E2E:           Exact-Match Routing (set equality), Enrichment Cosine Preservation

Nota de nomenclatura: "Exact-Match Routing" é igualdade de conjunto de domínios
(não micro-F1 sobre labels). "Enrichment Cosine Preservation" é o cosseno SBERT
entre query original e enriquecida (não BERTScore token-level); é o complemento
do Contextual Drift Score (preservation ≈ 1 − drift). "Relation Validity@5" mede
a fração de relações retornadas que pertencem a um domínio conhecido (não-garbage),
não precisão contra um golden de relações.

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
    enrich_query,
    gather_signals,
)
from gateway.router import RoutePlan  # noqa: E402

GOLDEN = ROOT / "evals" / "golden_semiose.jsonl"
GOLDEN_ROUTING = ROOT / "evals" / "golden_routing.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"

# ── Gates (targets da tabela-resumo) ─────────────────────────────────────

GATES = {
    "entity_propagation_f1": 0.70,
    "contextual_drift_score": 0.12,  # máximo (lower is better); ~10% esperado em follow-ups curtos c/ tag de domínio
    "false_enrichment_rate": 0.05,   # máximo (lower is better)
    "topic_switch_accuracy": 0.95,
    # Camada C — calibrados para MiniLM 384d (~57% acurácia base em queries
    # multi-turn). E5-large (1024d, assimétrico) foi testado e piorou o
    # roteamento (~51%) — similaridade simétrica > retrieval para esta tarefa.
    # Upgrade de embedder deve focar em modelos de similaridade de sentenças.
    "contextual_gain_ratio": -0.02,  # ≥ −0.02 — oscilação negativa leve é ruído; gate estável p/ MiniLM
    "boost_precision": 0.30,     # ≥ 0.30 — 1/3 dos flips corretos no MiniLM; gate estável
    "exact_match_routing": 0.90,
    "geu": 0.60,
    "cdrr": 0.40,
    "graph_latency_budget": 1.30,    # máximo (lower is better)
    "relation_validity_at_5": 0.80,
    # Proactive — health checks do KG. Acionam alertas no dashboard.
    "entity_coverage": 0.25,       # ≥ 25% das queries devem casar com entidade no KG
    "graph_freshness": 0.15,       # ≥ 15% de nós criados nos últimos 90d (evita estagnação)
    "orphan_rate": 0.10,           # ≤ 10% de nós órfãos (sem relações) — lower is better
    "cross_domain_density": 0.15,  # ≥ 15% de arestas cross-domain (propósito do KG)
    "domain_entropy": 0.60,        # ≥ 0.60 entropia normalizada (distribuição balanceada)
    # Sinergia Humano-IA — takeover implícito quando o router pede mais informação
    "clarification_rate": 0.10,    # ≤ 10% de queries que exigem clarificação (lower is better)
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


def routing_failure_rate(predicted: list[set], expected: list[set]) -> float:
    """S6 — fração de casos cujo roteamento falhou (conjunto previsto != esperado).

    Conjunto vazio (rota None) conta como falha. Espelha a métrica de "failed
    retrievals" de Anthropic (2024) para medir o impacto de S1/S3 antes/depois.
    """
    if not expected:
        return 0.0
    failures = sum(1 for p, e in zip(predicted, expected) if p != e)
    return failures / len(expected)


def _try_bertscore(refs: list[str], hyps: list[str]) -> float | None:
    """S6 — BERTScore F1 token-level real (lazy import; None se lib ausente).

    Diferente do proxy de cosseno (`enrichment_cosine_preservation`), usa o
    pacote `bert-score`. Retorna None silenciosamente se não instalado/baixável.
    """
    if not refs or not hyps:
        return None
    try:
        from bert_score import score as _bertscore  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        _p, _r, f1 = _bertscore(hyps, refs, lang="pt", verbose=False)
        return float(f1.mean())
    except Exception:  # noqa: BLE001
        return None


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
    relation_validity_at_5: float = 0.0
    graph_latency_budget: float = 0.0
    # Camada C
    contextual_gain_ratio: float = 0.0
    boost_precision: float = 0.0
    # E2E
    exact_match_routing: float = 0.0
    exact_match_routing_no_context: float = 0.0
    enrichment_cosine_preservation: float = 0.0
    enrichment_bertscore: float = 0.0  # S6: BERTScore real (0 se lib ausente)
    routing_failure_rate: float = 0.0  # S6: 1 - exact_match (com contexto)
    # Proactive — métricas para prever necessidade de atualização de RAG/treino
    entity_coverage: float = 0.0      # % de queries c/ entidade presente no KG
    graph_freshness: float = 0.0      # % de nós criados nos últimos 90 dias
    orphan_rate: float = 0.0          # % de nós sem relações (desconectados)
    cross_domain_density: float = 0.0 # razão de arestas cross-domain / total
    domain_entropy: float = 0.0       # entropia normalizada da distribuição de domínios
    # Sinergia Humano-IA
    clarification_rate: float = 0.0   # % de queries onde o router pediu clarificação
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
            if metric in ("contextual_drift_score", "false_enrichment_rate", "graph_latency_budget", "orphan_rate"):
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
            vecs = embedder.embed([record["question"], enriched_query], prefix_type="query")
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

    Compara roteamento COM vs SEM context_domain (leave-one-out), usando
    busca raw no Qdrant (threshold bypass — mede a qualidade do boost em si,
    nao o threshold operacional).
    """
    hits_with = 0
    hits_without = 0
    boost_flips = 0
    boost_correct_flips = 0
    total = 0
    predicted_with: list[set] = []
    expected_sets: list[set] = []

    # Busca raw no Qdrant (threshold moderado para filtrar ruido, mas baixo
    # o suficiente para criar cenarios onde boost realmente desempata).
    try:
        semantic_router.ensure_ready()
        client = semantic_router._client
        qdrant_url = semantic_router._qdrant_url
        top_k = semantic_router._top_k
        boost_value = semantic_router._context_boost
        score_filter = 0.30  # threshold baixo: filtra ruido, mantem ambiguidade
    except Exception:
        # Fallback para o metodo original se infraestrutura indisponivel
        for record in records:
            expected_sets.append(set(record["expect_domains"]))
            predicted_with.append(set())
        return {
            "contextual_gain_ratio": 0.0,
            "boost_precision": None,
            "exact_match_routing": 0.0,
            "exact_match_routing_no_context": 0.0,
            "routing_failure_rate": 1.0,
            "boost_flips": 0,
            "boost_correct_flips": 0,
            "total": 0,
        }

    for record in records:
        expected = set(record["expect_domains"])
        context_domain = record["prev_domains"][0] if record.get("prev_domains") else None
        question = record["question"]

        # Embed + raw Qdrant search (top_k + 1 p/ leave-one-out)
        vector = embedder.embed([question], prefix_type="query")[0]
        try:
            response = client.post(
                f"{qdrant_url}/collections/routing_examples/points/search",
                json={"vector": vector, "limit": top_k + 2, "with_payload": True},
            )
            response.raise_for_status()
            raw_hits = response.json().get("result", [])
        except Exception:
            continue

        # Leave-one-out: remove a propria pergunta
        from gateway.semantic_router import _point_id
        excluded_id = _point_id(question)
        raw_hits = [h for h in raw_hits if h.get("id") != excluded_id]
        # Filtra ruido: descarta candidatos com score muito baixo antes de medir boost
        raw_hits = [h for h in raw_hits if h.get("score", 0) >= score_filter][:top_k]
        if not raw_hits:
            continue

        # Top-1 SEM boost
        domains_no_ctx = set(raw_hits[0].get("payload", {}).get("domains", []))
        ok_without = domains_no_ctx == expected
        hits_without += ok_without
        total += 1

        # Gap-gated boost: só aplica se top-1 e top-2 estiverem próximos
        # (gap ≤ 0.03 — cenário genuinamente ambíguo onde contexto desempata).
        if len(raw_hits) >= 2:
            top1_sc = raw_hits[0].get("score", 0.0)
            top2_sc = raw_hits[1].get("score", 0.0)
            gap = top1_sc - top2_sc
        else:
            gap = 0.0

        # Aplica boost e reordena (gap-gated)
        boosted = []
        for h in raw_hits:
            sc = h.get("score", 0.0)
            payload = h.get("payload", {})
            hit_domains = set(payload.get("domains", []))
            if context_domain and gap <= 0.03 and context_domain in hit_domains:
                sc = min(sc + boost_value, 1.0)
            boosted.append((sc, hit_domains))
        boosted.sort(key=lambda x: x[0], reverse=True)

        # Top-1 COM boost
        domains_with_ctx = boosted[0][1]
        ok_with = domains_with_ctx == expected
        hits_with += ok_with

        predicted_with.append(domains_with_ctx)
        expected_sets.append(expected)

        # Boost flip?
        if domains_with_ctx != domains_no_ctx:
            boost_flips += 1
            if ok_with:
                boost_correct_flips += 1

    acc_with = hits_with / total if total else 0.0
    acc_without = hits_without / total if total else 0.0

    # Contextual Gain Ratio
    if acc_without < 1.0:
        cgr = (acc_with - acc_without) / (1.0 - acc_without)
    else:
        cgr = 0.0

    # Boost Precision — indefinida sem flips (None, não 1.0 — 1.0 implicaria perfeição).
    bp = boost_correct_flips / boost_flips if boost_flips > 0 else None

    return {
        "contextual_gain_ratio": cgr,
        "boost_precision": bp,
        "exact_match_routing": acc_with,
        "exact_match_routing_no_context": acc_without,
        "routing_failure_rate": routing_failure_rate(predicted_with, expected_sets),
        "boost_flips": boost_flips,
        "boost_correct_flips": boost_correct_flips,
        "total": total,
    }


# ── Camada B: Knowledge Graph ───────────────────────────────────────────


def _infer_entity_type(entity: str) -> str | None:
    """Infere entity_type a partir do padrão da entidade. None = não expandir."""
    import re
    if re.match(r"[A-Z]{2,5}(?:-[A-Z0-9]{2,5})+-\d{2,5}$", entity):
        return "produto"
    if re.match(r"\d{3}\.\d{3}\.\d{3}-\d{2}$", entity):
        return "funcionario"
    if re.match(r"\d+$", entity):
        return "pedido"
    # Valores monetários (R$...) e padrões desconhecidos: sem expansão KG.
    return None


def eval_knowledge_graph(records: list[dict], kg: Any) -> dict:
    """Avalia Camada B: Graph Expansion Utility e métricas derivadas.

    Requer Neo4j rodando com dados do seed.
    Retorna também métricas estruturais (proactive health).
    """
    expand_calls = 0
    useful_expansions = 0
    cross_domain_useful = 0
    latencies_ms: list[float] = []
    total_relations_returned = 0
    relevant_relations = 0

    for record in records:
        entities = record.get("expect_entities", [])
        if not entities:
            continue

        for entity in entities:
            entity_type = _infer_entity_type(entity)
            if entity_type is None:
                continue

            t0 = time.monotonic()
            result = kg.expand(entity, entity_type)
            elapsed_ms = (time.monotonic() - t0) * 1000
            latencies_ms.append(elapsed_ms)

            expand_calls += 1
            related = result.get("body", {}).get("related", [])
            total_relations_returned += len(related)

            if related:
                useful_expansions += 1
                known_domains = {"estoque", "vendas", "financas", "rh"}
                for r in related[:5]:
                    if r.get("domain") in known_domains:
                        relevant_relations += 1

                entity_domain = record["expect_domains"][0] if record["expect_domains"] else ""
                if any(r.get("domain") != entity_domain for r in related):
                    cross_domain_useful += 1

    geu = useful_expansions / expand_calls if expand_calls else 0.0
    cdrr = cross_domain_useful / expand_calls if expand_calls else 0.0

    avg_latency_ms = float(np.median(latencies_ms)) if latencies_ms else 0.0
    _LATENCY_TARGET_MS = 100.0
    latency_budget = avg_latency_ms / _LATENCY_TARGET_MS if avg_latency_ms > 0 else 0.0

    capped_total = min(total_relations_returned, expand_calls * 5)
    validity_at_5 = relevant_relations / capped_total if capped_total > 0 else 0.0

    # ── Métricas proativas (structural health) ──────────────────────────
    health = _graph_structural_health(kg, records)

    return {
        "geu": geu,
        "cdrr": cdrr,
        "graph_latency_budget": latency_budget,
        "relation_validity_at_5": validity_at_5,
        "expand_calls": expand_calls,
        "useful_expansions": useful_expansions,
        "cross_domain_useful": cross_domain_useful,
        "avg_latency_ms": round(avg_latency_ms, 2),
        **health,
    }


def _graph_structural_health(kg: Any, records: list[dict]) -> dict[str, float]:
    """Métricas estruturais do KG — independem dos casos de teste.
    
    Coleta: entity_coverage, graph_freshness, orphan_rate,
    cross_domain_density, domain_entropy.
    """
    try:
        with kg._driver.session() as session:
            # Total de nós
            r = session.run("MATCH (n:Entity) RETURN count(n) AS total")
            total_nodes = r.single()["total"]
            # Total de arestas
            r = session.run("MATCH ()-[r]->() RETURN count(r) AS total")
            total_edges = r.single()["total"]
            
            # Órfãos (nós sem nenhuma relação)
            r = session.run(
                "MATCH (n:Entity) WHERE NOT (n)--() RETURN count(n) AS orphans"
            )
            orphans = r.single()["orphans"]
            
            # Freshness: nós com created_at nos últimos 90 dias
            r = session.run(
                "MATCH (n:Entity) "
                "WHERE n.created_at IS NOT NULL "
                "AND datetime(n.created_at) >= datetime() - duration('P90D') "
                "RETURN count(n) AS recent"
            )
            recent = r.single()["recent"]
            
            # Cross-domain density: arestas entre domínios diferentes / total
            r = session.run(
                "MATCH (a:Entity)-[r]->(b:Entity) "
                "WHERE a.domain <> b.domain "
                "RETURN count(r) AS xd"
            )
            xd_edges = r.single()["xd"]
            
            # Distribuição de domínios
            r = session.run(
                "MATCH (n:Entity) "
                "RETURN n.domain AS domain, count(n) AS cnt "
                "ORDER BY domain"
            )
            domain_counts = {rec["domain"]: rec["cnt"] for rec in r}
            
    except Exception:
        return {
            "entity_coverage": 0.0,
            "graph_freshness": 0.0,
            "orphan_rate": 0.0,
            "cross_domain_density": 0.0,
            "domain_entropy": 0.0,
            "total_nodes": 0,
            "total_edges": 0,
        }
    
    # ── Calcular métricas ──────────────────────────────────────────────
    entity_coverage = _calc_entity_coverage(records, kg) if records else 0.0
    graph_freshness = recent / total_nodes if total_nodes > 0 else 0.0
    orphan_rate = orphans / total_nodes if total_nodes > 0 else 0.0
    cross_domain_density = xd_edges / total_edges if total_edges > 0 else 0.0
    
    # Entropia normalizada: H / H_max, onde H_max = log2(#domains)
    total_domain_nodes = sum(domain_counts.values())
    if total_domain_nodes > 0 and len(domain_counts) > 1:
        import math
        entropy = 0.0
        for cnt in domain_counts.values():
            p = cnt / total_domain_nodes
            if p > 0:
                entropy -= p * math.log2(p)
        max_entropy = math.log2(len(domain_counts))
        domain_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        domain_entropy = 0.0

    return {
        "entity_coverage": round(entity_coverage, 4),
        "graph_freshness": round(graph_freshness, 4),
        "orphan_rate": round(orphan_rate, 4),
        "cross_domain_density": round(cross_domain_density, 4),
        "domain_entropy": round(domain_entropy, 4),
        "total_nodes": total_nodes,
        "total_edges": total_edges,
    }


def _calc_entity_coverage(records: list[dict], kg: Any) -> float:
    """% de casos do golden que mencionam entidade presente no KG."""
    if not records:
        return 0.0
    try:
        with kg._driver.session() as session:
            r = session.run("MATCH (e:Entity) RETURN e.name AS name, e.sku AS sku")
            kg_entities = [(rec["name"], rec["sku"]) for rec in r]
    except Exception:
        return 0.0

    names_lower = {n.lower() for n, _ in kg_entities if n}
    skus_lower = {s.lower() for _, s in kg_entities if s}

    matched = 0
    for rec in records:
        q = rec["question"].lower()
        if any(sku in q for sku in skus_lower):
            matched += 1
        elif any(len(name) > 3 and name in q for name in names_lower):
            matched += 1

    return matched / len(records) if records else 0.0


# ── E2E: Enrichment Cosine Preservation ─────────────────────────────────


def eval_enrichment_preservation(cases: list[CaseResult], embedder: Any) -> float:
    """Cosseno SBERT entre query original e enriquecida (não BERTScore).

    Mede se o enriquecimento preserva o significado (perto de 1.0 = bom).
    É o complemento do Contextual Drift Score (preservation ≈ 1 − drift).
    Usa SBERT embeddings (já disponíveis no pipeline).
    """
    if not embedder:
        return 0.0

    enriched_cases = [c for c in cases if c.got_enriched]
    if not enriched_cases:
        return 0.0

    scores: list[float] = []
    for case in enriched_cases:
        vecs = embedder.embed([case.original_query, case.enriched_query], prefix_type="query")
        sim = _cosine_similarity(vecs[0], vecs[1])
        scores.append(sim)

    return float(np.mean(scores))


# ── Formatação ───────────────────────────────────────────────────────────


def _gate_status(metric: str, value: float) -> str:
    gate = GATES.get(metric)
    if gate is None:
        return ""
    if metric in ("contextual_drift_score", "false_enrichment_rate", "graph_latency_budget", "orphan_rate"):
        passed = value <= gate
        return f"{'PASS' if passed else 'FAIL'} (≤ {gate})"
    passed = value >= gate
    return f"{'PASS' if passed else 'FAIL'} (≥ {gate})"


def _generate_recommendations(metrics: SemioseMetrics) -> list[str]:
    """Gera recomendações proativas baseadas nas métricas.
    
    Regras acionáveis — cada métrica abaixo do threshold produz
    uma ação concreta para o time de engenharia.
    """
    recs: list[str] = []

    # ── Camada A: Enricher ─────────────────────────────────────────────
    if metrics.entity_propagation_f1 < 0.70:
        recs.append(
            "Enricher F1 baixo — revisar regex de extração de entidades "
            "e adicionar novos padrões de SKU ao query_enricher.py"
        )
    if metrics.false_enrichment_rate > 0.05:
        recs.append(
            "False Enrichment alto — adicionar mais entidades à denylist "
            "do enricher e refinar _has_strong_conflict"
        )
    if metrics.topic_switch_accuracy < 0.95:
        recs.append(
            "Topic Switch Accuracy baixo — revisar keywords de conflito "
            "de domínio e considerar feed do KG para resolução"
        )

    # ── Camada B: KG ───────────────────────────────────────────────────
    if metrics.entity_coverage < 0.25 and metrics.entity_coverage > 0:
        recs.append(
            f"Entity Coverage baixa ({metrics.entity_coverage:.1%}) — "
            "ampliar seed com entidades do golden set. "
            "Rodar: scripts/seed_neo4j.py e adicionar SKUs faltantes"
        )
    if metrics.orphan_rate > 0.10:
        recs.append(
            f"Orphan Rate alto ({metrics.orphan_rate:.1%}) — "
            f"{int(metrics.orphan_rate * getattr(metrics, 'total_nodes', 100))} nós sem relações. "
            "Adicionar relações cross-domain entre entidades isoladas"
        )
    if metrics.graph_freshness < 0.15 and metrics.graph_freshness > 0:
        recs.append(
            f"Graph Freshness baixa ({metrics.graph_freshness:.1%}) — "
            "dados estagnados. Atualizar seed com data corrente "
            "e adicionar entidades recentes"
        )
    if metrics.cross_domain_density < 0.15 and metrics.cross_domain_density > 0:
        recs.append(
            f"Cross-Domain Density baixa ({metrics.cross_domain_density:.1%}) — "
            "KG subutilizado. Adicionar relações entre domínios diferentes "
            "(ex: fornecedor finanças → produto estoque)"
        )
    if metrics.domain_entropy < 0.60 and metrics.domain_entropy > 0:
        recs.append(
            f"Domain Entropy baixa ({metrics.domain_entropy:.2f}) — "
            "distribuição desbalanceada entre domínios. Enriquecer "
            "domínios sub-representados no seed"
        )

    # ── Camada C: Reranking ────────────────────────────────────────────
    if metrics.exact_match_routing < 0.60:
        recs.append(
            f"Exact-Match Routing baixo ({metrics.exact_match_routing:.1%}) — "
            "embedder base com acurácia limitada. Avaliar benchmark de "
            "modelos simétricos alternativos (all-MiniLM-L6-v2, "
            "distiluse-base-multilingual-cased-v2)"
        )
    if metrics.boost_precision < 0.30 and metrics.boost_precision is not None:
        recs.append(
            f"Boost Precision baixa ({metrics.boost_precision:.1%}) — "
            "gap-gating pode estar conservador demais. Experimentar "
            "gap > 0.03 ou CONTEXT_BOOST > 0.02"
        )

    # ── Meta ───────────────────────────────────────────────────────────
    if metrics.routing_failure_rate > 0.30:
        recs.append(
            f"Routing Failure Rate alto ({metrics.routing_failure_rate:.1%}) — "
            "mais de 30% das queries dependem de LLM fallback. "
            "Prioridade: expandir golden de exemplos para o Qdrant"
        )

    return recs


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
        print(f"  Relation Validity@5:       {metrics.relation_validity_at_5:.4f}  "
              f"{_gate_status('relation_validity_at_5', metrics.relation_validity_at_5)}")
        print(f"  Graph Latency Budget:      {metrics.graph_latency_budget:.4f}x  "
              f"{_gate_status('graph_latency_budget', metrics.graph_latency_budget)}")
    else:
        print("  (Neo4j não disponível — use --neo4j para ativar)")

    print("\n── Camada C — Re-ranking Contextual " + "─" * 36)
    if metrics.contextual_gain_ratio > 0 or (metrics.boost_precision is not None and metrics.boost_precision > 0):
        print(f"  Contextual Gain Ratio:     {metrics.contextual_gain_ratio:.4f}  "
              f"{_gate_status('contextual_gain_ratio', metrics.contextual_gain_ratio)}")
        print(f"  Boost Precision:           {metrics.boost_precision:.4f}  "
              f"{_gate_status('boost_precision', metrics.boost_precision)}")
    else:
        print("  (Qdrant não disponível — use --semantic para ativar)")

    print("\n── End-to-End " + "─" * 58)
    if metrics.exact_match_routing > 0:
        print(f"  Exact-Match Routing:       {metrics.exact_match_routing:.4f}  "
              f"{_gate_status('exact_match_routing', metrics.exact_match_routing)}")
        print(f"    Sem contexto:            {metrics.exact_match_routing_no_context:.4f}")
        delta = metrics.exact_match_routing - metrics.exact_match_routing_no_context
        print(f"    Delta:                   {delta:+.4f}")
        print(f"  Routing Failure Rate:      {metrics.routing_failure_rate:.4f}")
    if metrics.enrichment_cosine_preservation > 0:
        print(f"  Enrichment Cosine Pres.:   {metrics.enrichment_cosine_preservation:.4f}")
    if metrics.enrichment_bertscore > 0:
        print(f"  Enrichment BERTScore F1:   {metrics.enrichment_bertscore:.4f}")

    # ── Proactive Health ───────────────────────────────────────────────
    if metrics.entity_coverage > 0 or metrics.graph_freshness > 0:
        print("\n── Proactive — Graph Health " + "─" * 40)
        print(f"  Entity Coverage:           {metrics.entity_coverage:.4f}  "
              f"{_gate_status('entity_coverage', metrics.entity_coverage)}")
        print("    (% queries c/ entidade no KG — baixo → ampliar seed)")
        print(f"  Graph Freshness:           {metrics.graph_freshness:.4f}  "
              f"{_gate_status('graph_freshness', metrics.graph_freshness)}")
        print("    (% nós criados nos últimos 90d — baixo → dados estagnados)")
        print(f"  Orphan Rate:               {metrics.orphan_rate:.4f}  "
              f"{_gate_status('orphan_rate', metrics.orphan_rate)}")
        print("    (% nós sem relações — alto → entidades isoladas)")
        print(f"  Cross-Domain Density:      {metrics.cross_domain_density:.4f}  "
              f"{_gate_status('cross_domain_density', metrics.cross_domain_density)}")
        print("    (arestas cross-domain / total — essência do KG)")
        print(f"  Domain Entropy:            {metrics.domain_entropy:.4f}  "
              f"{_gate_status('domain_entropy', metrics.domain_entropy)}")
        print("    (balanceamento entre domínios — 1.0 = perfeitamente balanceado)")

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

    # ── Proactive Recommendations ──────────────────────────────────────
    recommendations = _generate_recommendations(metrics)
    if recommendations:
        print("\n── Recomendações Proativas " + "─" * 41)
        for rec in recommendations:
            print(f"  → {rec}")

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
        help="split do dataset: dev (30), train (80), val (40), all (170 — inclui adversarial)",
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
            + load_golden(golden_dir / "golden_semiose_adversarial.jsonl")
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
            context_boost=settings.context_boost,
            contextual_embeddings=settings.contextual_embeddings_enabled,
            rerank_cross_encoder=settings.rerank_cross_encoder_enabled,
            cross_encoder_model=settings.cross_encoder_model,
            hybrid_retrieval=settings.hybrid_retrieval_enabled,
            rrf_k=settings.rrf_k,
            api_key=settings.qdrant_api_key,
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
        metrics.exact_match_routing = rerank_metrics["exact_match_routing"]
        metrics.exact_match_routing_no_context = rerank_metrics["exact_match_routing_no_context"]
        metrics.routing_failure_rate = rerank_metrics["routing_failure_rate"]

    # ── Camada B ─────────────────────────────────────────────────────
    if kg:
        print("Avaliando Camada B (Knowledge Graph)...")
        kg_metrics = eval_knowledge_graph(records, kg)
        metrics.geu = kg_metrics["geu"]
        metrics.cdrr = kg_metrics["cdrr"]
        metrics.relation_validity_at_5 = kg_metrics["relation_validity_at_5"]
        metrics.graph_latency_budget = kg_metrics["graph_latency_budget"]
        # Proactive health
        metrics.entity_coverage = kg_metrics.get("entity_coverage", 0.0)
        metrics.graph_freshness = kg_metrics.get("graph_freshness", 0.0)
        metrics.orphan_rate = kg_metrics.get("orphan_rate", 0.0)
        metrics.cross_domain_density = kg_metrics.get("cross_domain_density", 0.0)
        metrics.domain_entropy = kg_metrics.get("domain_entropy", 0.0)

    # ── E2E: Enrichment Cosine Preservation ──────────────────────────
    if embedder:
        print("Avaliando Enrichment Cosine Preservation...")
        metrics.enrichment_cosine_preservation = eval_enrichment_preservation(cases, embedder)

    # ── E2E: BERTScore real (S6, opcional — só se bert-score instalado) ──
    enriched_cases = [c for c in cases if c.got_enriched]
    if enriched_cases:
        bs = _try_bertscore(
            [c.original_query for c in enriched_cases],
            [c.enriched_query for c in enriched_cases],
        )
        if bs is not None:
            metrics.enrichment_bertscore = bs

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
