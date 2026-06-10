"""Testes de API do serviço RH."""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from rh.main import app

THIS_MONTH = date.today().strftime("%Y-%m")
THIS_YEAR = date.today().year
FUTURE_START = (date.today() + timedelta(days=30)).isoformat()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "rh.db"))
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "service": "rh"}


def test_seed_loaded(client):
    assert len(client.get("/employees").json()) >= 8


def test_get_employee_not_found(client):
    resp = client.get("/employees/9999")
    assert resp.status_code == 404
    assert resp.json()["error"] == "employee_not_found"


def test_headcount_by_department(client):
    rows = client.get("/headcount", params={"department": "Engenharia"}).json()
    assert rows == [{"department": "Engenharia", "headcount": 4}]


def test_vacation_balance_reflects_seed(client):
    # Mariana (id=1) já usou 10 dias no ano corrente (seed).
    balance = client.get("/employees/1/vacation-balance", params={"acquisition_year": THIS_YEAR}).json()
    assert balance["taken_days"] == 10
    assert balance["balance"] == 20


def test_vacation_request_happy_path(client):
    resp = client.post(
        "/vacations",
        json={"employee_id": 2, "start_date": FUTURE_START, "days": 15, "acquisition_year": THIS_YEAR},
    )
    assert resp.status_code == 201
    assert resp.json()["days"] == 15


def test_vacation_before_12_months_returns_422(client):
    # Larissa (id=9) foi admitida há ~90 dias.
    resp = client.post(
        "/vacations",
        json={"employee_id": 9, "start_date": FUTURE_START, "days": 14, "acquisition_year": THIS_YEAR},
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "vacation_not_yet_entitled"
    assert "12 meses" in body["rule"]


def test_vacation_over_balance_returns_422(client):
    # Mariana (id=1) tem só 20 dias de saldo.
    resp = client.post(
        "/vacations",
        json={"employee_id": 1, "start_date": FUTURE_START, "days": 25, "acquisition_year": THIS_YEAR},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "vacation_insufficient_balance"


def test_vacation_fourth_period_returns_422(client):
    for days in (14, 6, 5):
        assert (
            client.post(
                "/vacations",
                json={"employee_id": 3, "start_date": FUTURE_START, "days": days, "acquisition_year": THIS_YEAR},
            ).status_code
            == 201
        )
    resp = client.post(
        "/vacations",
        json={"employee_id": 3, "start_date": FUTURE_START, "days": 5, "acquisition_year": THIS_YEAR},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "vacation_too_many_periods"


def test_reimbursement_travel_over_limit_returns_422(client):
    resp = client.post(
        "/reimbursements",
        json={"employee_id": 4, "category": "viagem", "amount": 4_500.00, "month": THIS_MONTH},
    )
    assert resp.status_code == 422
    assert resp.json()["error"] == "reimbursement_over_travel_limit"


def test_reimbursement_home_office_accumulates_per_month(client):
    # André (id=6) já tem R$320 reembolsados no mês (seed); restam R$180.
    over = client.post(
        "/reimbursements",
        json={"employee_id": 6, "category": "home_office", "amount": 200.00, "month": THIS_MONTH},
    )
    assert over.status_code == 422
    assert over.json()["error"] == "reimbursement_over_home_office_limit"
    ok = client.post(
        "/reimbursements",
        json={"employee_id": 6, "category": "home_office", "amount": 180.00, "month": THIS_MONTH},
    )
    assert ok.status_code == 201


def test_reimbursement_meal_happy_path(client):
    resp = client.post(
        "/reimbursements",
        json={"employee_id": 5, "category": "refeicao", "amount": 280.00, "days": 3, "month": THIS_MONTH},
    )
    assert resp.status_code == 201
