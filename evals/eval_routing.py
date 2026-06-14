"""Eval de roteamento: classify_intent contra o MoE real (golden_routing.jsonl).

Acerto = igualdade de conjunto dos domínios + flag de clarification correta.
Gate: ≥ 90% de acurácia (critério de aceite da PoC, PLANO_EXECUCAO §4).

Uso:
    .venv/bin/python evals/eval_routing.py
    .venv/bin/python evals/eval_routing.py --limit 10
    .venv/bin/python evals/eval_routing.py --only-fails evals/results/routing_<ts>.json
    .venv/bin/python evals/eval_routing.py --semantic   # camada Qdrant em leave-one-out
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gateway.config import load_settings  # noqa: E402
from gateway.embedder import SBERTEmbedder  # noqa: E402
from gateway.llm import OllamaClient  # noqa: E402
from gateway.router import RoutePlan, classify_intent  # noqa: E402
from gateway.semantic_router import SemanticRouter  # noqa: E402

from evals.common import CaseTimeout, with_deadline  # noqa: E402

GOLDEN = ROOT / "evals" / "golden_routing.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
GATE = 0.90


def load_golden() -> list[dict]:
    with GOLDEN.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_failed_questions(results_path: str) -> set[str]:
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))
    return {item["question"] for item in data["items"] if not item["ok"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval de roteamento (gate >= 90%)")
    parser.add_argument("--limit", type=int, default=None, help="avalia só as N primeiras perguntas")
    parser.add_argument("--only-fails", default=None, help="re-roda só as falhas de um results.json anterior")
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="ativa a camada semântica (Qdrant) em leave-one-out: a própria pergunta é excluída do índice",
    )
    args = parser.parse_args()

    records = load_golden()
    if args.only_fails:
        failed = load_failed_questions(args.only_fails)
        records = [r for r in records if r["question"] in failed]
        if not records:
            print("Nenhuma falha a re-rodar — o results.json informado está 100%.")
            return 0
    if args.limit:
        records = records[: args.limit]

    settings = load_settings()
    llm = OllamaClient(settings.ollama_url, settings.model, timeout_s=settings.llm_timeout_s, keep_alive=settings.keep_alive)
    semantic = None
    if args.semantic:
        embedder = SBERTEmbedder(
            model_name=settings.sbert_model,
            cache_dir=settings.sbert_cache_dir,
        )
        semantic = SemanticRouter(
            settings.qdrant_url,
            embedder,
            examples_path=str(GOLDEN),
            threshold=settings.semantic_threshold,
            top_k=settings.semantic_top_k,
        )
        semantic.ensure_ready()

        class _LeaveOneOut:
            """A própria pergunta está no índice — excluída para não casar consigo mesma."""

            def __init__(self, inner: SemanticRouter) -> None:
                self._inner = inner

            def route(self, question: str, **_: object):
                return self._inner.route(question, exclude_question=question)

        semantic = _LeaveOneOut(semantic)

    items: list[dict] = []
    hits = 0
    layer_counts: dict[str, int] = {"semantic": 0, "llm": 0, "lexical": 0}
    for i, record in enumerate(records, 1):
        started = time.monotonic()
        try:
            route = with_deadline(classify_intent, record["question"], llm, semantic=semantic)
        except CaseTimeout as exc:
            route = RoutePlan(domains=[], plan="", clarification=f"watchdog: {exc}")
        if route.plan.startswith("Roteado por similaridade"):
            layer = "semantic"
        elif route.plan.startswith("Roteado por keywords"):
            layer = "lexical"
        else:
            layer = "llm"
        layer_counts[layer] += 1
        elapsed = round(time.monotonic() - started, 2)

        got_domains = set(route.domains)
        got_clarification = bool(route.clarification)
        ok = got_domains == set(record["expect_domains"]) and got_clarification == record["expect_clarification"]
        hits += ok

        status = "OK  " if ok else "FAIL"
        print(
            f"[{i:02d}/{len(records)}] {status} {elapsed:5.1f}s "
            f"esperado={sorted(record['expect_domains'])} clar={record['expect_clarification']} | "
            f"obtido={sorted(got_domains)} clar={got_clarification} | {record['question'][:70]}"
        )
        items.append(
            {
                "question": record["question"],
                "expect_domains": record["expect_domains"],
                "expect_clarification": record["expect_clarification"],
                "got_domains": sorted(got_domains),
                "got_clarification": got_clarification,
                "plan": route.plan,
                "layer": layer,
                "elapsed_s": elapsed,
                "ok": ok,
            }
        )

    accuracy = hits / len(records) if records else 0.0
    passed = accuracy >= GATE
    print(f"\nAcurácia: {hits}/{len(records)} = {accuracy:.1%} — gate {GATE:.0%}: {'PASS' if passed else 'FAIL'}")
    if args.semantic:
        avg = {
            layer: round(
                sum(it["elapsed_s"] for it in items if it["layer"] == layer) / count, 2
            )
            for layer, count in layer_counts.items()
            if count
        }
        print(f"Camadas: {layer_counts} | latência média por camada (s): {avg}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"routing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "model": settings.model,
                "semantic": args.semantic,
                "layer_counts": layer_counts,
                "total": len(records),
                "hits": hits,
                "accuracy": round(accuracy, 4),
                "gate": GATE,
                "passed": passed,
                "items": items,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Resultados salvos em {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
