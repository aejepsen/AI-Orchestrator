"""Testes das funções puras do eval de faithfulness (juiz LLM local)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.eval_faithfulness import aggregate, judge_messages, judgeable, parse_verdict  # noqa: E402


def _task(**over) -> dict:
    task = {
        "id": "financas-01",
        "domain": "financas",
        "task": "Quais contas a pagar vencem hoje?",
        "final_answer": "Duas contas vencem hoje: #3 (R$ 300) e #5 (R$ 900).",
        "tool_trace": [
            {"name": "list_accounts", "args": {"type": "pagar"}, "status": 200, "body": '[{"id": 3}]'}
        ],
    }
    task.update(over)
    return task


class TestParseVerdict:
    def test_json_puro(self) -> None:
        assert parse_verdict('{"faithful": true, "motivo": "ok"}') == {"faithful": True, "motivo": "ok"}

    def test_json_com_texto_ao_redor(self) -> None:
        verdict = parse_verdict('Veredicto: {"faithful": false, "motivo": "número inventado"} fim')
        assert verdict == {"faithful": False, "motivo": "número inventado"}

    def test_invalido_retorna_none(self) -> None:
        assert parse_verdict("não sou json") is None
        assert parse_verdict('{"faithful": "sim"}') is None
        assert parse_verdict("") is None


class TestJudgeable:
    def test_task_completa_e_julgavel(self) -> None:
        assert judgeable(_task())

    def test_sem_resposta_pula(self) -> None:
        assert not judgeable(_task(final_answer=None))

    def test_trace_sem_body_pula(self) -> None:
        trace = [{"name": "t", "args": {}, "status": 200, "body": None}]
        assert not judgeable(_task(tool_trace=trace))


class TestJudgeMessages:
    def test_inclui_pergunta_observacoes_e_resposta(self) -> None:
        messages = judge_messages(_task())
        assert messages[0]["role"] == "system"
        user = messages[1]["content"]
        assert "Quais contas a pagar vencem hoje?" in user
        assert "list_accounts" in user
        assert "Duas contas vencem hoje" in user


class TestAggregate:
    def test_metricas_e_gate(self) -> None:
        verdicts = [
            {"id": "a", "verdict": {"faithful": True, "motivo": ""}},
            {"id": "b", "verdict": {"faithful": True, "motivo": ""}},
            {"id": "c", "verdict": {"faithful": False, "motivo": "x"}},
            {"id": "d", "verdict": None},
        ]
        metrics = aggregate(verdicts, skipped=2)
        assert metrics["judged"] == 3
        assert metrics["faithful"] == 2
        assert metrics["faithfulness_rate"] == 0.6667
        assert metrics["judge_errors"] == 1
        assert metrics["skipped"] == 2
        assert not metrics["gate_pass"]

    def test_vazio_nao_passa_gate(self) -> None:
        metrics = aggregate([], skipped=0)
        assert metrics["judged"] == 0 and not metrics["gate_pass"]
