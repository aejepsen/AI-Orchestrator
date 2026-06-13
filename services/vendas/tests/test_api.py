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


# ── POST /sellers ─────────────────────────────────────────────────────────


def test_create_seller_happy_path(client):
    resp = client.post("/sellers", json={"name": "Carlos Silva", "region": "Norte"})
    assert resp.status_code == 201
    body = resp.json()
    assert body["name"] == "Carlos Silva"
    assert body["region"] == "Norte"
    assert "id" in body


def test_create_seller_missing_name_returns_422(client):
    resp = client.post("/sellers", json={"name": "", "region": "Sul"})
    assert resp.status_code == 422


def test_list_sellers(client):
    sellers = client.get("/sellers").json()
    assert len(sellers) >= 2  # seeded


# ── PUT /sellers/{id} ────────────────────────────────────────────────────


def test_update_seller_happy_path(client):
    created = client.post("/sellers", json={"name": "Teste Update", "region": "Leste"}).json()
    resp = client.put(f"/sellers/{created['id']}", json={"name": "Teste Atualizado"})
    assert resp.status_code == 200
    assert resp.json()["name"] == "Teste Atualizado"
    assert resp.json()["region"] == "Leste"  # unchanged


def test_update_seller_not_found(client):
    resp = client.put("/sellers/9999", json={"name": "Ninguém"})
    assert resp.status_code == 404
    assert resp.json()["error"] == "seller_not_found"


# ── DELETE /sellers/{id} ─────────────────────────────────────────────────


def test_delete_seller_happy_path(client):
    created = client.post("/sellers", json={"name": "Descartável", "region": "Oeste"}).json()
    resp = client.delete(f"/sellers/{created['id']}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] == created["id"]


def test_delete_seller_with_active_orders_returns_422(client):
    # "Rafael Monteiro" has active orders from seed
    sellers = client.get("/sellers").json()
    rafael = next(s for s in sellers if s["name"] == "Rafael Monteiro")
    resp = client.delete(f"/sellers/{rafael['id']}")
    assert resp.status_code == 422
    assert resp.json()["error"] == "seller_has_active_orders"


def test_delete_seller_not_found(client):
    resp = client.delete("/sellers/9999")
    assert resp.status_code == 404


# ── PUT /orders/{id} ─────────────────────────────────────────────────────


def test_update_order_discount(client):
    order = client.post("/orders", json=ORDER_PAYLOAD).json()
    resp = client.put(f"/orders/{order['id']}", json={"discount_pct": 8.0})
    assert resp.status_code == 200
    updated = resp.json()
    assert updated["discount_pct"] == 8.0
    assert updated["gross_total"] == 1_660.00
    assert updated["net_total"] == 1_527.20


def test_update_order_quantity_and_price(client):
    order = client.post("/orders", json=ORDER_PAYLOAD).json()
    resp = client.put(f"/orders/{order['id']}", json={"quantity": 10, "unit_price": 100.0})
    assert resp.status_code == 200
    updated = resp.json()
    # 2 items, each now qty=10 price=100 → gross=2000, 5% discount → net=1900
    assert updated["gross_total"] == 2_000.00
    assert updated["net_total"] == 1_900.00


def test_update_order_completed_returns_422(client):
    order = client.post("/orders", json=ORDER_PAYLOAD).json()
    # Manually set status to concluido
    from vendas import db
    with db.connect() as conn:
        conn.execute("UPDATE orders SET status = 'concluido' WHERE id = ?", (order["id"],))
        conn.commit()
    resp = client.put(f"/orders/{order['id']}", json={"discount_pct": 2.0})
    assert resp.status_code == 422
    assert resp.json()["error"] == "order_not_editable"


def test_update_order_not_found(client):
    resp = client.put("/orders/9999", json={"discount_pct": 1.0})
    assert resp.status_code == 404


def test_update_order_discount_above_role_limit_returns_422(client):
    order = client.post("/orders", json=ORDER_PAYLOAD).json()
    resp = client.put(f"/orders/{order['id']}", json={"discount_pct": 15.0})
    assert resp.status_code == 422
    assert resp.json()["error"] == "discount_above_role_limit"


# ── DELETE /orders/{id} ──────────────────────────────────────────────────


def test_delete_order_happy_path(client):
    order = client.post("/orders", json=ORDER_PAYLOAD).json()
    resp = client.delete(f"/orders/{order['id']}")
    assert resp.status_code == 200
    assert resp.json()["cancelled"] == order["id"]
    # Verify it's now cancelled
    check = client.get(f"/orders/{order['id']}").json()
    assert check["status"] == "cancelado"


def test_delete_order_completed_returns_422(client):
    order = client.post("/orders", json=ORDER_PAYLOAD).json()
    from vendas import db
    with db.connect() as conn:
        conn.execute("UPDATE orders SET status = 'concluido' WHERE id = ?", (order["id"],))
        conn.commit()
    resp = client.delete(f"/orders/{order['id']}")
    assert resp.status_code == 422
    assert resp.json()["error"] == "order_already_completed"


def test_delete_order_not_found(client):
    resp = client.delete("/orders/9999")
    assert resp.status_code == 404
