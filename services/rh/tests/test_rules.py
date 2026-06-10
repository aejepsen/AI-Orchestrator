"""Testes das regras puras de RH."""

from datetime import date

import pytest

from common import RuleViolation
from rh import rules

HIRE = date(2024, 3, 1)
ELIGIBLE_START = date(2026, 1, 10)  # bem após 12 meses


class TestAddMonths:
    def test_simple(self):
        assert rules.add_months(date(2024, 3, 1), 12) == date(2025, 3, 1)

    def test_clamps_to_short_month(self):
        assert rules.add_months(date(2024, 1, 31), 1) == date(2024, 2, 28)


class TestVacationEligibility:
    def test_before_12_months_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_vacation_request(HIRE, date(2025, 1, 10), 14, [])
        assert exc.value.error == "vacation_not_yet_entitled"

    def test_exactly_at_12_months_passes(self):
        rules.validate_vacation_request(HIRE, date(2025, 3, 1), 30, [])


class TestVacationFractioning:
    def test_period_under_5_days_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_vacation_request(HIRE, ELIGIBLE_START, 4, [])
        assert exc.value.error == "vacation_period_too_short"

    def test_fourth_period_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_vacation_request(HIRE, ELIGIBLE_START, 5, [14, 6, 5])
        assert exc.value.error == "vacation_too_many_periods"

    def test_three_valid_periods_pass(self):
        rules.validate_vacation_request(HIRE, ELIGIBLE_START, 14, [])
        rules.validate_vacation_request(HIRE, ELIGIBLE_START, 10, [14])
        rules.validate_vacation_request(HIRE, ELIGIBLE_START, 6, [14, 10])

    def test_request_blocking_14_day_period_fails(self):
        # 13 + 13 deixaria só 4 dias: nenhum período ≥14 seria possível.
        with pytest.raises(RuleViolation) as exc:
            rules.validate_vacation_request(HIRE, ELIGIBLE_START, 13, [13])
        assert exc.value.error == "vacation_no_long_period_possible"

    def test_short_period_first_is_allowed_while_14_still_fits(self):
        rules.validate_vacation_request(HIRE, ELIGIBLE_START, 10, [])  # restam 20, cabe um de 14


class TestVacationBalance:
    def test_over_balance_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_vacation_request(HIRE, ELIGIBLE_START, 25, [10])
        assert exc.value.error == "vacation_insufficient_balance"

    def test_balance_helper(self):
        assert rules.vacation_balance([14, 6]) == 10


class TestReimbursement:
    def test_travel_within_limit(self):
        rules.validate_reimbursement("viagem", 3_000.00, None, 0)

    def test_travel_over_limit_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_reimbursement("viagem", 3_000.01, None, 0)
        assert exc.value.error == "reimbursement_over_travel_limit"

    def test_meal_requires_days(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_reimbursement("refeicao", 90.00, None, 0)
        assert exc.value.error == "reimbursement_days_required"

    def test_meal_over_daily_limit_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_reimbursement("refeicao", 250.00, 2, 0)
        assert exc.value.error == "reimbursement_over_meal_limit"

    def test_meal_within_daily_limit(self):
        rules.validate_reimbursement("refeicao", 200.00, 2, 0)

    def test_home_office_monthly_accumulation_fails(self):
        with pytest.raises(RuleViolation) as exc:
            rules.validate_reimbursement("home_office", 200.00, None, 320.00)
        assert exc.value.error == "reimbursement_over_home_office_limit"

    def test_home_office_within_remaining_budget(self):
        rules.validate_reimbursement("home_office", 180.00, None, 320.00)
