"""Testes da lógica pura do seed do Neo4j (alçada de aprovação)."""

from __future__ import annotations

from scripts.seed_neo4j import _required_approver


def test_alcada_auto_aprovada_ate_5k():
    assert _required_approver(5_000.00) is None
    assert _required_approver(487.90) is None


def test_alcada_gerente_ate_50k():
    assert _required_approver(5_000.01) == "gerente"
    assert _required_approver(24_000.00) == "gerente"
    assert _required_approver(50_000.00) == "gerente"


def test_alcada_diretor_acima_50k():
    assert _required_approver(50_000.01) == "diretor"
    assert _required_approver(62_000.00) == "diretor"
