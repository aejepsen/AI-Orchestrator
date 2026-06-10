"""Eval de roteamento: classify_intent contra o MoE real (golden_routing.jsonl).

Acerto = igualdade de conjunto dos domínios + flag de clarification correta.
Gate: ≥ 90% de acurácia (critério de aceite da PoC, PLANO_EXECUCAO §4).

Uso:
    .venv/bin/python evals/eval_routing.py
    .venv/bin/python evals/eval_routing.py --limit 10
    .venv/bin/python evals/eval_routing.py --only-fails evals/results/routing_<ts>.json
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
from gateway.llm import OllamaClient  # noqa: E402
from gateway.router import classify_intent  # noqa: E402

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
    llm = OllamaClient(settings.ollama_url, settings.model, timeout_s=settings.llm_timeout_s)

    items: list[dict] = []
    hits = 0
    for i, record in enumerate(records, 1):
        started = time.monotonic()
        route = classify_intent(record["question"], llm)
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
                "elapsed_s": elapsed,
                "ok": ok,
            }
        )

    accuracy = hits / len(records) if records else 0.0
    passed = accuracy >= GATE
    print(f"\nAcurácia: {hits}/{len(records)} = {accuracy:.1%} — gate {GATE:.0%}: {'PASS' if passed else 'FAIL'}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"routing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(),
                "model": settings.model,
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
