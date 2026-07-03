"""Eval de retrieval do RAG de políticas — gate: Recall@3 >= 0.8.

Para cada pergunta do golden, verifica se algum dos top-3 chunks retornados
por `search_documents` contém a substring esperada (a regra correta). Mede
RETRIEVAL, não geração — a fidelidade da resposta final já é coberta pelo
eval de faithfulness.

Uso:
    set -a; source .env; set +a
    SBERT_CACHE_DIR=$PWD/models .venv/bin/python evals/eval_docs.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.config import load_settings  # noqa: E402
from gateway.document_search import DocumentSearch  # noqa: E402
from gateway.embedder import SBERTEmbedder  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"
GATE_RECALL = 0.80

# (pergunta, substring que o chunk correto contém)
GOLDEN = [
    ("Qual o limite de desconto que um vendedor pode dar?", "10%"),
    ("Até quanto o gerente pode aprovar de despesa?", "50.000"),
    ("Quem aprova despesas acima de cinquenta mil reais?", "diretor"),
    ("Quantos dias de férias por ano o funcionário tem direito?", "30 dias"),
    ("Em quantos períodos posso fracionar as férias?", "3 períodos"),
    ("Qual o limite de reembolso para viagens?", "3.000"),
    ("Quanto posso pedir de reembolso de refeição por dia?", "100,00"),
    ("Qual o teto mensal do reembolso de home office?", "500,00"),
    ("Como é calculado o percentual de comissão dos vendedores?", "3,5%"),
    ("O que acontece quando o estoque atinge o ponto de reposição?", "reposição"),
    ("Como funciona a reserva de itens do estoque?", "disponível"),
    ("Existe um teto absoluto de desconto nos pedidos?", "20%"),
]


def main() -> int:
    settings = load_settings()
    embedder = SBERTEmbedder(model_name=settings.sbert_model, cache_dir=settings.sbert_cache_dir)
    search = DocumentSearch(settings.qdrant_url, embedder, api_key=settings.qdrant_api_key)

    started = time.monotonic()
    outcomes = []
    for question, expected in GOLDEN:
        result = search.search(question)
        results = result.get("body", {}).get("results", [])
        hit = any(expected.casefold() in r["text"].casefold() for r in results)
        outcomes.append({"question": question, "expected": expected, "hit": hit,
                         "top": results[0]["section"] if results else None})
        print(f"{'OK  ' if hit else 'MISS'} {question[:70]}"
              + ("" if hit else f" | top: {results[0]['section'] if results else 'vazio'}"))

    recall = sum(o["hit"] for o in outcomes) / len(outcomes)
    gate = recall >= GATE_RECALL
    print(f"\nRecall@3: {sum(o['hit'] for o in outcomes)}/{len(outcomes)} = {recall:.1%} "
          f"— gate {GATE_RECALL:.0%}: {'PASS' if gate else 'FAIL'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"docs_retrieval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({
        "phase": "docs_retrieval",
        "timestamp": datetime.now().isoformat(),
        "recall_at_3": round(recall, 4),
        "gate": GATE_RECALL,
        "gate_pass": gate,
        "cases": outcomes,
    }, ensure_ascii=False, indent=2))
    print(f"Relatório: {out} | Duração: {time.monotonic() - started:.1f}s")
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
