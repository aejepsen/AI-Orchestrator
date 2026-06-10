"""Testes das regras puras de Vendas."""

import pytest

from common import RuleViolation
from vendas import rules


class TestDiscountPolicy:
    def test_vendedor_up_to_10_pct(self):
        rules.validate_discount(10.0, "vendedor")

    def test_vendedor_above_10_pct_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_discount(10.1, "vendedor")
        assert exc.value.error == "discount_above_role_limit"
        assert "gerente" in exc.value.detail  # detail sugere o papel com alçada

    def test_gerente_up_to_20_pct(self):
        rules.validate_discount(20.0, "gerente")

    def test_gerente_above_20_pct_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_discount(20.1, "gerente")
        assert exc.value.error == "discount_above_policy_maximum"

    def test_above_20_pct_fails_for_any_role(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_discount(35.0, "vendedor")
        assert exc.value.error == "discount_above_policy_maximum"

    def test_zero_discount_always_passes(self):
        rules.validate_discount(0.0, "vendedor")


class TestOrderTotals:
    def test_gross_and_net_with_discount(self):
        items = [
            {"quantity": 2, "unit_price": 100.00},
            {"quantity": 1, "unit_price": 50.00},
        ]
        assert rules.order_totals(items, 10.0) == {"gross_total": 250.00, "net_total": 225.00}

    def test_no_discount_keeps_gross(self):
        items = [{"quantity": 3, "unit_price": 33.33}]
        totals = rules.order_totals(items, 0.0)
        assert totals["gross_total"] == totals["net_total"] == 99.99


class TestCommission:
    def test_2_pct_up_to_10k(self):
        assert rules.compute_commission(10_000.00) == {"rate": 0.02, "commission": 200.00}

    def test_3_5_pct_above_10k(self):
        assert rules.compute_commission(10_000.01) == {"rate": 0.035, "commission": 350.00}

    def test_high_ticket_commission(self):
        assert rules.compute_commission(80_000.00) == {"rate": 0.035, "commission": 2_800.00}
