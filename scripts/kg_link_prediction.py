"""Fase B do parecer CliffordNet: embeddings rotacionais para curadoria do KG.

Treina link prediction (RotatE — rotações no plano complexo — e TransE como
baseline) sobre as triplas do Neo4j e gera SUGESTÕES de relações ausentes,
filtradas por assinatura de tipo (relação só é candidata entre tipos de
entidade já observados para ela — PyKEEN não conhece o schema).

Saída: relatório JSON (MRR/Hits@10 por modelo) + CSV de candidatas para
REVISÃO HUMANA. Este script NUNCA escreve no Neo4j — curadoria assistida,
não automática (parecer §4 Fase 2 / feedback do usuário).

Gates: Hits@10 >= 0.5 no split de teste E RotatE >= TransE (a rotação tem
que pagar o próprio custo); >= 5 candidatas plausíveis na revisão.

Nota de honestidade estatística: ~295 triplas → ~30 no teste; métricas têm
variância alta. Tratar como demonstração + assistente de curadoria.

Uso:
    .venv/bin/python scripts/kg_link_prediction.py            # treina + sugere
    .venv/bin/python scripts/kg_link_prediction.py --top 30   # mais candidatas
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "evals" / "results"
GATE_HITS_AT_10 = 0.5
SEED = 42


# -- helpers puros (testáveis sem pykeen/neo4j) --------------------------------


def entity_label(name: str, entity_type: str) -> str:
    """Label única e legível: nomes podem colidir entre tipos."""
    return f"{entity_type}:{name}"


def relation_signatures(triples: list[tuple[str, str, str]]) -> dict[str, set[tuple[str, str]]]:
    """Assinaturas observadas por relação: {rel: {(tipo_head, tipo_tail)}}."""
    signatures: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for head, relation, tail in triples:
        signatures[relation].add((head.split(":", 1)[0], tail.split(":", 1)[0]))
    return dict(signatures)


def candidate_triples(
    triples: list[tuple[str, str, str]],
    *,
    max_per_relation: int = 5000,
) -> list[tuple[str, str, str]]:
    """Triplas type-compatible ausentes do KG (espaço de busca das sugestões)."""
    existing = set(triples)
    entities_by_type: dict[str, list[str]] = defaultdict(list)
    for head, _, tail in triples:
        for entity in (head, tail):
            etype = entity.split(":", 1)[0]
            if entity not in entities_by_type[etype]:
                entities_by_type[etype].append(entity)

    candidates: list[tuple[str, str, str]] = []
    for relation, signatures in relation_signatures(triples).items():
        seen = 0
        for head_type, tail_type in sorted(signatures):
            for head in entities_by_type[head_type]:
                for tail in entities_by_type[tail_type]:
                    if head == tail:
                        continue
                    triple = (head, relation, tail)
                    if triple in existing:
                        continue
                    candidates.append(triple)
                    seen += 1
                    if seen >= max_per_relation:
                        break
                if seen >= max_per_relation:
                    break
            if seen >= max_per_relation:
                break
    return candidates


# -- extração e treino ----------------------------------------------------------


def fetch_triples() -> list[tuple[str, str, str]]:
    from neo4j import GraphDatabase

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        match = re.search(r"NEO4J_PASSWORD=(.*)", (ROOT / ".env").read_text())
        password = match.group(1).strip() if match else "changeme"
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    driver = GraphDatabase.driver(uri, auth=(os.environ.get("NEO4J_USER", "neo4j"), password))
    query = (
        "MATCH (h)-[r]->(t) "
        "RETURN h.name AS head, h.type AS head_type, type(r) AS relation, "
        "t.name AS tail, t.type AS tail_type"
    )
    with driver.session() as session:
        rows = session.run(query).data()
    driver.close()
    return [
        (entity_label(row["head"], row["head_type"]), row["relation"], entity_label(row["tail"], row["tail_type"]))
        for row in rows
        if row["head"] and row["tail"] and row["head_type"] and row["tail_type"]
    ]


def train_and_evaluate(triples: list[tuple[str, str, str]], model_name: str):
    import numpy as np
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory

    factory = TriplesFactory.from_labeled_triples(
        np.array(triples, dtype=str), create_inverse_triples=True
    )
    training, testing, validation = factory.split([0.8, 0.1, 0.1], random_state=SEED)
    # Config vencedora do sweep 2026-07-02 (3 seeds): RotatE d64 + NSSA n16 →
    # MRR 0.171±0.080 vs TransE 0.056±0.019 (rotação = 3× o baseline). Loss
    # self-adversarial é a do paper do RotatE (Sun et al., 2019).
    extra = (
        dict(
            loss="NSSA",
            loss_kwargs=dict(margin=6.0, adversarial_temperature=1.0),
            negative_sampler_kwargs=dict(num_negs_per_pos=16),
        )
        if model_name == "RotatE"
        else {}
    )
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model=model_name,
        model_kwargs=dict(embedding_dim=64),
        training_kwargs=dict(num_epochs=600, batch_size=64),
        stopper="early",
        stopper_kwargs=dict(frequency=20, patience=6, relative_delta=0.002),
        random_seed=SEED,
        device="cpu",
        **extra,
    )
    metrics = result.metric_results
    return result, factory, {
        "mrr": round(float(metrics.get_metric("inverse_harmonic_mean_rank")), 4),
        "hits_at_1": round(float(metrics.get_metric("hits_at_1")), 4),
        "hits_at_10": round(float(metrics.get_metric("hits_at_10")), 4),
    }


def score_candidates(result, factory, candidates: list[tuple[str, str, str]], top: int):
    """Score das candidatas com ranking dentro de cada relação (scores entre
    relações não são comparáveis diretamente)."""
    import torch

    entity_to_id = factory.entity_to_id
    relation_to_id = factory.relation_to_id
    valid = [
        c for c in candidates
        if c[0] in entity_to_id and c[2] in entity_to_id and c[1] in relation_to_id
    ]
    if not valid:
        return []
    hrt = torch.as_tensor(
        [[entity_to_id[h], relation_to_id[r], entity_to_id[t]] for h, r, t in valid],
        dtype=torch.long,
    )
    with torch.inference_mode():
        scores = result.model.score_hrt(hrt).squeeze(-1).numpy()

    by_relation: dict[str, list[tuple[float, tuple[str, str, str]]]] = defaultdict(list)
    for triple, score in zip(valid, scores):
        by_relation[triple[1]].append((float(score), triple))
    ranked: list[dict] = []
    for relation, scored in by_relation.items():
        scored.sort(key=lambda pair: -pair[0])
        for rank, (score, (head, _, tail)) in enumerate(scored[: max(3, top // len(by_relation) + 1)], 1):
            ranked.append(
                {"relation": relation, "head": head, "tail": tail, "score": round(score, 4), "rank_in_relation": rank}
            )
    ranked.sort(key=lambda item: (item["rank_in_relation"], -item["score"]))
    return ranked[:top]


def main() -> int:
    parser = argparse.ArgumentParser(description="Link prediction no KG (curadoria assistida)")
    parser.add_argument("--top", type=int, default=20, help="máx. de candidatas no CSV")
    args = parser.parse_args()

    started = time.monotonic()
    triples = fetch_triples()
    print(f"KG: {len(triples)} triplas, {len({e for h, _, t in triples for e in (h, t)})} entidades, "
          f"{len({r for _, r, _ in triples})} relações")

    results = {}
    best = None
    for model_name in ("TransE", "RotatE"):
        result, factory, metrics = train_and_evaluate(triples, model_name)
        results[model_name] = metrics
        print(f"{model_name}: MRR={metrics['mrr']} Hits@1={metrics['hits_at_1']} Hits@10={metrics['hits_at_10']}")
        if best is None or metrics["mrr"] > results[best[2]]["mrr"]:
            best = (result, factory, model_name)

    best_result, best_factory, best_name = best
    gate_hits = results[best_name]["hits_at_10"] >= GATE_HITS_AT_10
    gate_rotate = results["RotatE"]["mrr"] >= results["TransE"]["mrr"]
    print(f"\nMelhor modelo: {best_name} | gate Hits@10>={GATE_HITS_AT_10}: {'PASS' if gate_hits else 'FAIL'} "
          f"| RotatE>=TransE (MRR): {'PASS' if gate_rotate else 'FAIL'}")

    candidates = candidate_triples(triples)
    suggestions = score_candidates(best_result, best_factory, candidates, args.top)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"kg_suggestions_{stamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["relation", "head", "tail", "score", "rank_in_relation"])
        writer.writeheader()
        writer.writerows(suggestions)

    report_path = RESULTS_DIR / f"kg_link_prediction_{stamp}.json"
    report_path.write_text(
        json.dumps(
            {
                "phase": "kg_link_prediction",
                "timestamp": datetime.now().isoformat(),
                "triples": len(triples),
                "models": results,
                "best_model": best_name,
                "gate_hits_at_10": GATE_HITS_AT_10,
                "gate_pass": bool(gate_hits),
                "gate_rotate_beats_transe": bool(gate_rotate),
                "candidates_scored": len(candidates),
                "suggestions_csv": csv_path.name,
                "suggestions": suggestions,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nSugestões (top {len(suggestions)}) → {csv_path}")
    for item in suggestions[:10]:
        print(f"  [{item['relation']}] {item['head']} → {item['tail']} (score {item['score']})")
    print(f"Relatório: {report_path}")
    print(f"Duração: {(time.monotonic() - started) / 60:.1f} min")
    return 0 if gate_hits else 1


if __name__ == "__main__":
    raise SystemExit(main())
