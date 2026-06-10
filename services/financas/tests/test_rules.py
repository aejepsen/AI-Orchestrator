"""Testes das regras puras de Finanças."""

from datetime import date

import pytest

from common import RuleViolation
from financas import rules


class TestApprovalAuthority:
    def test_expense_up_to_5000_is_auto_approved(self):
        assert rules.required_approver_role(5_000.00) is None
        rules.validate_expense_approval(4_999.99, None)

    def test_expense_up_to_50000_requires_gerente(self):
        assert rules.required_approver_role(50_000.00) == "gerente"
        rules.validate_expense_approval(50_000.00, "gerente")

    def test_expense_above_50000_requires_diretor(self):
        assert rules.required_approver_role(50_000.01) == "diretor"
        rules.validate_expense_approval(60_000.00, "diretor")

    def test_diretor_can_approve_mid_tier_expense(self):
        rules.validate_expense_approval(30_000.00, "diretor")

    def test_mid_tier_expense_without_approver_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_expense_approval(10_000.00, None)
        assert exc.value.error == "insufficient_approval_authority"

    def test_high_expense_with_gerente_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_expense_approval(80_000.00, "gerente")
        assert "diretor" in exc.value.detail


class TestSettlement:
    def test_open_payable_can_be_paid(self):
        rules.validate_settlement("pagar", "aberta", "pay")

    def test_open_receivable_can_be_received(self):
        rules.validate_settlement("receber", "aberta", "receive")

    def test_paying_a_receivable_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_settlement("receber", "aberta", "pay")
        assert exc.value.error == "wrong_account_type"

    def test_paying_twice_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_settlement("pagar", "paga", "pay")
        assert exc.value.error == "account_already_settled"


class TestCashflow:
    ACCOUNTS = [
        {"type": "receber", "amount": 1_000.00, "due_date": date(2026, 6, 10)},
        {"type": "receber", "amount": 500.00, "due_date": date(2026, 6, 20)},
        {"type": "pagar", "amount": 300.00, "due_date": date(2026, 6, 15)},
        {"type": "pagar", "amount": 999.00, "due_date": date(2026, 7, 1)},  # fora do período
    ]

    def test_cashflow_sums_only_period(self):
        flow = rules.compute_cashflow(self.ACCOUNTS, date(2026, 6, 1), date(2026, 6, 30))
        assert flow == {"entradas": 1_500.00, "saidas": 300.00, "saldo": 1_200.00}

    def test_cashflow_empty_period(self):
        flow = rules.compute_cashflow(self.ACCOUNTS, date(2027, 1, 1), date(2027, 1, 31))
        assert flow == {"entradas": 0, "saidas": 0, "saldo": 0}
