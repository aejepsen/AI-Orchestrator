"""Testes das regras puras de Estoque."""

import pytest

from common import RuleViolation
from estoque import rules


class TestAvailability:
    def test_available_discounts_reservations(self):
        assert rules.available(on_hand=50, reserved=12) == 38

    def test_available_can_reach_zero(self):
        assert rules.available(on_hand=10, reserved=10) == 0


class TestReservation:
    def test_reservation_within_available_passes(self):
        rules.validate_reservation("SKU-X", on_hand=50, reserved=12, quantity=38)

    def test_reservation_above_available_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_reservation("SKU-X", on_hand=50, reserved=12, quantity=39)
        assert exc.value.error == "insufficient_stock"
        assert "38" in exc.value.detail  # detail informa o disponível real

    def test_reservation_on_fully_reserved_stock_fails(self):
        with pytest.raises(RuleViolation):
            rules.validate_reservation("SKU-X", on_hand=10, reserved=10, quantity=1)


class TestReplenishment:
    def test_below_reorder_point_suggests_quantity(self):
        # disponível 6, ponto 12 → sugestão 12*2 - 6 = 18
        assert rules.replenishment_suggestion(on_hand=8, reserved=2, reorder_point=12) == 18

    def test_at_reorder_point_does_not_suggest(self):
        assert rules.replenishment_suggestion(on_hand=12, reserved=0, reorder_point=12) is None

    def test_above_reorder_point_does_not_suggest(self):
        assert rules.replenishment_suggestion(on_hand=100, reserved=10, reorder_point=20) is None

    def test_reservations_can_trigger_replenishment(self):
        # físico acima do ponto, mas disponível abaixo por causa das reservas
        assert rules.replenishment_suggestion(on_hand=25, reserved=10, reorder_point=20) == 25
