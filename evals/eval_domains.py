"""Eval determinístico dos subagentes por domínio (Fase 2).

Roda as tasks de `golden_domains.jsonl` sequencialmente contra os serviços
e o MoE reais, valida tools usadas, status esperados e substrings na resposta,
e aplica o gate de >=80% por domínio. Relatório JSON em `evals/results/`.

Uso:
    python evals/eval_domains.py                       # tudo
    python evals/eval_domains.py --domains rh vendas   # subconjunto de domínios
    python evals/eval_domains.py --ids rh-03 vendas-05 # re-rodar tasks específicas

Formato de cada record do golden:
    id, domain, task,
    expect_tools: lista de tools obrigatórias na trace (item str = exato;
                  item lista = qualquer um daquele grupo satisfaz),
    expect_answer_any: passa se QUALQUER substring aparecer (case-insensitive),
    expect_status (opcional): status HTTP que deve aparecer na trace (ex.: 422).
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.agents import AgentResult, DomainAgentRunner  # noqa: E402
from gateway.config import DOMAINS, load_settings  # noqa: E402
from gateway.llm import OllamaClient  # noqa: E402

GOLDEN_PATH = ROOT / "evals" / "golden_domains.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
GATE_PCT = 80.0


@dataclass
class TaskOutcome:
    record: dict[str, Any]
    result: AgentResult | None = None
    failures: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.record["id"],
            "domain": self.record["domain"],
            "task": self.record["task"],
            "passed": self.passed,
            "failures": self.failures,
            "error": self.error,
            "final_answer": self.result.final_answer if self.result else None,
            "tool_trace": self.result.trace_as_dicts() if self.result else [],
            "iters": self.result.iters if self.result else 0,
            "elapsed_s": self.result.elapsed_s if self.result else 0.0,
            "stop_reason": self.result.stop_reason if self.result else None,
        }


def load_golden(domains: list[str] | None, ids: list[str] | None) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in GOLDEN_PATH.read_text().splitlines() if line.strip()]
    if domains:
        records = [r for r in records if r["domain"] in domains]
    if ids:
        records = [r for r in records if r["id"] in ids]
    return records


def check(record: dict[str, Any], result: AgentResult) -> list[str]:
    failures: list[str] = []
    trace_names = [t.name for t in result.tool_trace]
    trace_statuses = [t.status for t in result.tool_trace]

    for expected in record.get("expect_tools", []):
        options = expected if isinstance(expected, list) else [expected]
        if not any(option in trace_names for option in options):
            failures.append(f"tool ausente na trace: {' ou '.join(options)} (trace: {trace_names})")

    expected_status = record.get("expect_status")
    if expected_status is not None and expected_status not in trace_statuses:
        failures.append(f"status {expected_status} ausente na trace (statuses: {trace_statuses})")

    answer = (result.final_answer or "").casefold()
    substrings = record.get("expect_answer_any", [])
    if substrings and not any(s.casefold() in answer for s in substrings):
        failures.append(f"nenhuma substring esperada na resposta: {substrings}")

    if result.stop_reason != "answer":
        failures.append(f"loop terminou por {result.stop_reason} (sem resposta final)")
    return failures


def warm_up(settings) -> float:
    """Uma chamada curta para garantir o MoE residente (cold load ~55 s)."""
    print("Warm-up do MoE (pode levar ~1 min em cold load)...", flush=True)
    llm = OllamaClient(settings.ollama_url, settings.model, timeout_s=settings.llm_timeout_s)
    started = time.monotonic()
    llm.chat([{"role": "user", "content": "Responda apenas: ok"}])
    elapsed = time.monotonic() - started
    print(f"Warm-up concluído em {elapsed:.1f}s.\n", flush=True)
    return elapsed


def run(records: list[dict[str, Any]]) -> list[TaskOutcome]:
    settings = load_settings()
    warm_up(settings)
    runner = DomainAgentRunner(settings=settings)
    outcomes: list[TaskOutcome] = []

    for i, record in enumerate(records, start=1):
        outcome = TaskOutcome(record=record)
        try:
            outcome.result = runner.run(record["domain"], record["task"])
            outcome.failures = check(record, outcome.result)
        except Exception as exc:  # noqa: BLE001 — eval reporta, não quebra
            outcome.error = f"{type(exc).__name__}: {exc}"
        outcomes.append(outcome)

        status = "PASS" if outcome.passed else "FAIL"
        detail = "" if outcome.passed else " | " + "; ".join(outcome.failures or [outcome.error or ""])
        elapsed = f"{outcome.result.elapsed_s:.1f}s" if outcome.result else "-"
        print(f"[{i:>2}/{len(records)}] {status} {record['id']:<12} ({elapsed}){detail}", flush=True)
        if not outcome.passed and outcome.result:
            print(f"        resposta: {outcome.result.final_answer[:300]}", flush=True)
            print(f"        trace: {[(t.name, t.status) for t in outcome.result.tool_trace]}", flush=True)
    return outcomes


def summarize(outcomes: list[TaskOutcome]) -> tuple[dict[str, dict[str, Any]], bool]:
    summary: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        domain_outcomes = [o for o in outcomes if o.record["domain"] == domain]
        if not domain_outcomes:
            continue
        passed = sum(o.passed for o in domain_outcomes)
        pct = 100.0 * passed / len(domain_outcomes)
        summary[domain] = {
            "passed": passed,
            "total": len(domain_outcomes),
            "pct": round(pct, 1),
            "gate": pct >= GATE_PCT,
        }
    all_pass = all(d["gate"] for d in summary.values())

    print("\n" + "=" * 64)
    print(f"{'Domínio':<12}{'Pass':>6}{'Total':>7}{'Score':>9}{'Gate >=80%':>14}")
    print("-" * 64)
    for domain, stats in summary.items():
        gate = "OK" if stats["gate"] else "REPROVADO"
        print(f"{domain:<12}{stats['passed']:>6}{stats['total']:>7}{stats['pct']:>8.1f}%{gate:>14}")
    total_passed = sum(d["passed"] for d in summary.values())
    total = sum(d["total"] for d in summary.values())
    print("-" * 64)
    print(f"{'TOTAL':<12}{total_passed:>6}{total:>7}{100.0 * total_passed / total:>8.1f}%")
    print("=" * 64)
    return summary, all_pass


def save_report(outcomes: list[TaskOutcome], summary: dict[str, dict[str, Any]]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    path = RESULTS_DIR / f"fase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(
        json.dumps(
            {
                "phase": "fase2_subagentes",
                "timestamp": datetime.now().isoformat(),
                "model": settings.model,
                "gate_pct": GATE_PCT,
                "summary": summary,
                "tasks": [o.as_dict() for o in outcomes],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Eval dos subagentes por domínio (Fase 2)")
    parser.add_argument("--domains", nargs="*", choices=DOMAINS, help="Limita a domínios específicos")
    parser.add_argument("--ids", nargs="*", help="Re-roda apenas tasks específicas (ex.: rh-03)")
    args = parser.parse_args()

    records = load_golden(args.domains, args.ids)
    if not records:
        print("Nenhuma task selecionada.", file=sys.stderr)
        return 2

    started = time.monotonic()
    outcomes = run(records)
    summary, all_pass = summarize(outcomes)
    report = save_report(outcomes, summary)
    print(f"\nRelatório: {report}")
    print(f"Duração total: {(time.monotonic() - started) / 60:.1f} min")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
