"""S4 — build offline dos resumos de comunidade do KG (GraphRAG mínimo).

Louvain (networkx) sobre o grafo do Neo4j → 1 resumo LLM por comunidade
(grounded: o prompt só recebe os fatos das arestas da comunidade; proibido
inventar). Artefato em ./models/kg_communities.json (volume do gateway,
fora do git — mesmo padrão do injection classifier).

Uso:
    set -a; source .env; set +a
    .venv/bin/python scripts/build_kg_communities.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT_PATH = ROOT / "models" / "kg_communities.json"
MIN_COMMUNITY_SIZE = 4
SEED = 42

_SUMMARY_SYSTEM = """Você resume um agrupamento de entidades de um grafo corporativo.
Escreva um parágrafo de 3-5 frases em português descrevendo o que conecta essas
entidades (quem compra o quê, quem trabalha onde, o que requer aprovação de quem),
EXCLUSIVAMENTE com base nos FATOS listados. Não invente números, nomes ou relações.
Responda apenas o parágrafo."""


def fetch_graph():
    from neo4j import GraphDatabase

    password = os.environ.get("NEO4J_PASSWORD")
    if not password:
        match = re.search(r"NEO4J_PASSWORD=(.*)", (ROOT / ".env").read_text())
        password = match.group(1).strip() if match else "changeme"
    driver = GraphDatabase.driver(
        os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.environ.get("NEO4J_USER", "neo4j"), password),
    )
    with driver.session() as session:
        rows = session.run(
            "MATCH (h)-[r]->(t) RETURN h.name AS h, h.type AS ht, h.domain AS hd, "
            "type(r) AS rel, t.name AS t, t.type AS tt, t.domain AS td"
        ).data()
    driver.close()
    return rows


def main() -> int:
    import networkx as nx

    from gateway.config import load_settings
    from gateway.llm import OllamaClient

    started = time.monotonic()
    rows = fetch_graph()
    graph = nx.Graph()
    facts_by_edge: dict[tuple[str, str], list[str]] = {}
    domains_by_node: dict[str, str] = {}
    for row in rows:
        head, tail = f"{row['ht']}:{row['h']}", f"{row['tt']}:{row['t']}"
        graph.add_edge(head, tail)
        facts_by_edge.setdefault((head, tail), []).append(f"{row['h']} {row['rel']} {row['t']}")
        domains_by_node[head] = row.get("hd") or ""
        domains_by_node[tail] = row.get("td") or ""

    communities = nx.community.louvain_communities(graph, seed=SEED)
    modularity = nx.community.modularity(graph, communities)
    big = [sorted(c) for c in communities if len(c) >= MIN_COMMUNITY_SIZE]
    print(f"Grafo: {graph.number_of_nodes()} nós / {graph.number_of_edges()} arestas | "
          f"Louvain: {len(communities)} comunidades (modularity {modularity:.3f}), "
          f"{len(big)} com tamanho >= {MIN_COMMUNITY_SIZE}")

    settings = load_settings()
    llm = OllamaClient(
        settings.ollama_url, settings.model, timeout_s=settings.llm_timeout_s, keep_alive=settings.keep_alive
    )

    output = []
    for index, members in enumerate(big):
        member_set = set(members)
        facts = [
            fact
            for (head, tail), fact_list in facts_by_edge.items()
            if head in member_set and tail in member_set
            for fact in fact_list
        ]
        domains = sorted({domains_by_node.get(m, "") for m in members} - {""})
        response = llm.chat(
            [
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {"role": "user", "content": "FATOS:\n" + "\n".join(f"- {f}" for f in facts[:80])},
            ]
        )
        entities = [m.split(":", 1)[1] for m in members]
        output.append(
            {
                "id": index,
                "size": len(members),
                "domains": domains,
                "entities": entities,
                "summary": response.content.strip(),
            }
        )
        print(f"  comunidade {index}: {len(members)} entidades, domínios {domains} — resumo ok")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "generated_at": datetime.now().isoformat(),
                "algorithm": f"louvain(seed={SEED})",
                "modularity": round(modularity, 4),
                "model": settings.model,
                "communities": output,
            },
            ensure_ascii=False,
            indent=1,
        )
    )
    print(f"\nArtefato: {OUT_PATH} ({len(output)} comunidades)")
    print(f"Duração: {(time.monotonic() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
