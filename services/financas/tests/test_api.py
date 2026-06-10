"""Testes de API do serviço Finanças."""

from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from financas.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "financas.db"))
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "service": "financas"}


def test_seed_loaded(client):
    accounts = client.get("/accounts").json()
    assert len(accounts) >= 8


def test_list_filters_by_type_and_status(client):
    for account in client.get("/accounts", params={"type": "pagar", "status": "aberta"}).json():
        assert account["type"] == "pagar" and account["status"] == "aberta"


def test_get_account_not_found(client):
    resp = client.get("/accounts/9999")
    assert resp.status_code == 404
    body = resp.json()
    assert body["error"] == "account_not_found"
    assert {"error", "detail", "rule"} <= body.keys()


def test_create_small_expense_auto_approved(client):
    resp = client.post(
        "/accounts",
        json={
            "type": "pagar",
            "description": "Café para o escritório",
            "counterparty": "Mercado União",
            "amount": 250.00,
            "due_date": "2026-07-01",
        },
    )
    assert resp.status_code == 201
    assert resp.json()["status"] == "aberta"


def test_create_expense_without_authority_returns_422(client):
    resp = client.post(
        "/accounts",
        json={
            "type": "pagar",
            "description": "Reforma da recepção",
            "counterparty": "Construtora Lima",
            "amount": 75_000.00,
            "due_date": "2026-08-01",
            "approver_role": "gerente",
        },
    )
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "insufficient_approval_authority"
    assert "diretor" in body["detail"]


def test_pay_and_repay_flow(client):
    created = client.post(
        "/accounts",
        json={
            "type": "pagar",
            "description": "Manutenção do ar-condicionado",
            "counterparty": "Clima Frio Serviços",
            "amount": 900.00,
            "due_date": "2026-06-20",
        },
    ).json()
    paid = client.post(f"/accounts/{created['id']}/pay", json={"settled_date": "2026-06-18"})
    assert paid.status_code == 200
    assert paid.json()["status"] == "paga"
    again = client.post(f"/accounts/{created['id']}/pay")
    assert again.status_code == 422
    assert again.json()["error"] == "account_already_settled"


def test_receive_on_payable_returns_422(client):
    payable = client.get("/accounts", params={"type": "pagar", "status": "aberta"}).json()[0]
    resp = client.post(f"/accounts/{payable['id']}/receive")
    assert resp.status_code == 422
    assert resp.json()["error"] == "wrong_account_type"


def test_cashflow_window(client):
    start = date.today().isoformat()
    end = (date.today() + timedelta(days=30)).isoformat()
    flow = client.get("/cashflow", params={"start": start, "end": end}).json()
    assert flow["saldo"] == round(flow["entradas"] - flow["saidas"], 2)
    assert flow["entradas"] > 0
