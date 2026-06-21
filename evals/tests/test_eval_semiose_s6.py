"""Teste da métrica S6 routing_failure_rate (função pura, sem modelos)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from evals.eval_semiose import routing_failure_rate  # noqa: E402


def test_failure_rate_tudo_certo():
    pred = [{"vendas"}, {"rh"}]
    exp = [{"vendas"}, {"rh"}]
    assert routing_failure_rate(pred, exp) == 0.0


def test_failure_rate_tudo_errado():
    pred = [{"rh"}, set()]
    exp = [{"vendas"}, {"rh"}]
    assert routing_failure_rate(pred, exp) == 1.0


def test_failure_rate_conjunto_vazio_conta_como_falha():
    pred = [set(), {"vendas"}]
    exp = [{"vendas"}, {"vendas"}]
    assert routing_failure_rate(pred, exp) == 0.5


def test_failure_rate_sem_casos_retorna_zero():
    assert routing_failure_rate([], []) == 0.0
