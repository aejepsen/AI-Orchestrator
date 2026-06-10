"""Testes de API do serviço Vendas."""

import pytest
from fastapi.testclient import TestClient

from vendas.main import app

ORDER_PAYLOAD = {
    "customer": "Padaria Estrela do Sul",
    "salesperson": "Rafael Monteiro",
    "role": "vendedor",
    "discount_pct": 5.0,
    "items": [
        {"sku": "TEC-MEC-005", "quantity": 2, "unit_price": 520.00},
        {"sku": "MOU-ERG-006", "quantity": 2, "unit_price": 310.00},
    ],
    "order_date": "2026-06-09",
}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "vendas.db"))
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "service": "vendas"}


def test_seed_loaded(client):
    orders = client.get("/orders").json()
    assert len(orders) >= 8
    assert all(o["items"] for o in orders)


def test_get_order_not_found(client):
    resp = client.get("/orders/9999")
    assert resp.status_code == 404
    assert resp.json()["error"] == "order_not_found"


def test_create_order_computes_totals(client):
    resp = client.post("/orders", json=ORDER_PAYLOAD)
    assert resp.status_code == 201
    order = resp.json()
    assert order["gross_total"] == 1_660.00
    assert order["net_total"] == 1_577.00  # 5% de desconto


def test_create_order_vendedor_above_10_pct_returns_422(client):
    resp = client.post("/orders", json={**ORDER_PAYLOAD, "discount_pct": 15.0})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "discount_above_role_limit"
    assert {"error", "detail", "rule"} <= body.keys()


def test_create_order_gerente_15_pct_passes(client):
    resp = client.post("/orders", json={**ORDER_PAYLOAD, "role": "gerente", "discount_pct": 15.0})
    assert resp.status_code == 201


def test_create_order_above_20_pct_always_returns_422(client):
    resp = client.post("/orders", json={**ORDER_PAYLOAD, "role": "gerente", "discount_pct": 25.0})
    assert resp.status_code == 422
    assert resp.json()["error"] == "discount_above_policy_maximum"


def test_commission_low_ticket_uses_2_pct(client):
    order = client.post("/orders", json=ORDER_PAYLOAD).json()  # líquido R$1.577
    commission = client.get(f"/orders/{order['id']}/commission").json()
    assert commission["rate"] == 0.02
    assert commission["commission"] == 31.54


def test_commission_high_ticket_uses_3_5_pct(client):
    payload = {
        **ORDER_PAYLOAD,
        "discount_pct": 0,
        "items": [{"sku": "NTB-DEV-004", "quantity": 2, "unit_price": 9_800.00}],
    }
    order = client.post("/orders", json=payload).json()  # líquido R$19.600
    commission = client.get(f"/orders/{order['id']}/commission").json()
    assert commission["rate"] == 0.035
    assert commission["commission"] == 686.00
