"""Eval de Faithfulness (RAG Triad — governança): a resposta é fiel às observações?

Juiz LLM local avalia cada task de um relatório do eval_domains (fase2_*.json):
dado pergunta + observações das tools (status/body) + resposta final, o juiz
responde `{"faithful": bool, "motivo": str}`. Métrica: `faithfulness_rate` =
fração de respostas fiéis entre as julgadas. Gate: ≥ 0.90.

Desvio do observability-plan: em vez do FaithfulnessEvaluator do Phoenix (que
espera OpenAI por default), juiz direto via OllamaClient — zero-cloud, mesmo
padrão in-house dos demais evals. Phoenix segue como visualização opcional.

Requer relatório gerado após a captura de `body` na tool_trace (2026-07-02);
tasks sem observações com body são contadas como `skipped`.

Uso:
    .venv/bin/python evals/eval_faithfulness.py                   # último fase2_*.json
    .venv/bin/python evals/eval_faithfulness.py --results <path>
    .venv/bin/python evals/eval_faithfulness.py --limit 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.config import load_settings  # noqa: E402
from gateway.llm import OllamaClient  # noqa: E402

RESULTS_DIR = ROOT / "evals" / "results"
GATE = 0.90

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_JUDGE_SYSTEM = """Você é um auditor de fidelidade de respostas de um assistente corporativo.
Sua única tarefa: verificar se a RESPOSTA é sustentada pelas OBSERVAÇÕES (retornos reais de API).

Regras:
- "faithful": true somente se TODO fato, número, nome e conclusão da RESPOSTA estiver \
sustentado pelas OBSERVAÇÕES (ou for consequência aritmética direta delas).
- Qualquer dado inventado, número divergente ou afirmação sem base nas OBSERVAÇÕES → "faithful": false.
- Recusa correta (ex.: pedir campos obrigatórios, informar erro da API) é fiel se coerente com as OBSERVAÇÕES.
- Responda APENAS o JSON: {"faithful": true|false, "motivo": "<curto, em PT-BR>"}"""


def judge_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    observations = [
        {"tool": t["name"], "args": t.get("args"), "status": t.get("status"), "body": t.get("body")}
        for t in task.get("tool_trace", [])
    ]
    user = (
        f"PERGUNTA:\n{task['task']}\n\n"
        f"OBSERVAÇÕES (retornos de API):\n{json.dumps(observations, ensure_ascii=False, indent=1)}\n\n"
        f"RESPOSTA:\n{task.get('final_answer') or ''}"
    )
    return [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}]


def parse_verdict(content: str) -> dict[str, Any] | None:
    """Extrai o JSON do veredicto; tolerante a texto ao redor. None se inválido."""
    match = _JSON_OBJECT.search(content or "")
    if not match:
        return None
    try:
        verdict = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(verdict, dict) or not isinstance(verdict.get("faithful"), bool):
        return None
    return {"faithful": verdict["faithful"], "motivo": str(verdict.get("motivo", ""))}


def judgeable(task: dict[str, Any]) -> bool:
    """Task julgável = tem resposta final e ao menos uma observação com body."""
    if not task.get("final_answer"):
        return False
    return any(t.get("body") for t in task.get("tool_trace", []))


def aggregate(verdicts: list[dict[str, Any]], skipped: int) -> dict[str, Any]:
    judged = [v for v in verdicts if v.get("verdict") is not None]
    faithful = sum(1 for v in judged if v["verdict"]["faithful"])
    rate = faithful / len(judged) if judged else 0.0
    return {
        "faithfulness_rate": round(rate, 4),
        "faithful": faithful,
        "judged": len(judged),
        "judge_errors": len(verdicts) - len(judged),
        "skipped": skipped,
        "gate": GATE,
        "gate_pass": bool(judged) and rate >= GATE,
    }


def _latest_fase2() -> Path | None:
    candidates = sorted(RESULTS_DIR.glob("fase2_*.json"))
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval de faithfulness (juiz LLM local, gate >= 90%)")
    parser.add_argument("--results", default=None, help="fase2_*.json a julgar (default: mais recente)")
    parser.add_argument("--limit", type=int, default=None, help="julga só as N primeiras tasks")
    args = parser.parse_args()

    results_path = Path(args.results) if args.results else _latest_fase2()
    if not results_path or not results_path.exists():
        print("Nenhum relatório fase2_*.json encontrado — rode evals/eval_domains.py antes.", file=sys.stderr)
        return 2
    report = json.loads(results_path.read_text())
    tasks = report.get("tasks", [])
    if args.limit:
        tasks = tasks[: args.limit]

    settings = load_settings()
    llm = OllamaClient(
        settings.ollama_url, settings.model, timeout_s=settings.llm_timeout_s, keep_alive=settings.keep_alive
    )

    verdicts: list[dict[str, Any]] = []
    skipped = 0
    started = time.monotonic()
    for i, task in enumerate(tasks, start=1):
        if not judgeable(task):
            skipped += 1
            print(f"[{i:>2}/{len(tasks)}] SKIP  {task['id']:<12} (sem resposta ou sem body na trace)", flush=True)
            continue
        try:
            response = llm.chat(judge_messages(task), format="json")
            verdict = parse_verdict(response.content)
        except Exception as exc:  # noqa: BLE001 — eval reporta, não quebra
            verdict = None
            print(f"[{i:>2}/{len(tasks)}] ERRO  {task['id']:<12} {type(exc).__name__}: {exc}", flush=True)
        verdicts.append({"id": task["id"], "domain": task["domain"], "verdict": verdict})
        if verdict is not None:
            label = "FIEL " if verdict["faithful"] else "INFIEL"
            motivo = "" if verdict["faithful"] else f" | {verdict['motivo'][:120]}"
            print(f"[{i:>2}/{len(tasks)}] {label} {task['id']:<12}{motivo}", flush=True)

    metrics = aggregate(verdicts, skipped)
    print("\n" + "=" * 64)
    print(
        f"Faithfulness: {metrics['faithful']}/{metrics['judged']} = "
        f"{metrics['faithfulness_rate']:.1%} — gate {GATE:.0%}: "
        f"{'PASS' if metrics['gate_pass'] else 'FAIL'} "
        f"(skipped={metrics['skipped']}, judge_errors={metrics['judge_errors']})"
    )
    print("=" * 64)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"faithfulness_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(
        json.dumps(
            {
                "phase": "faithfulness",
                "timestamp": datetime.now().isoformat(),
                "model_judge": settings.model,
                "source_results": str(results_path.name),
                "metrics": metrics,
                "verdicts": verdicts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Relatório: {out}")
    print(f"Duração: {(time.monotonic() - started) / 60:.1f} min")
    return 0 if metrics["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
