"""Testes de API do serviço Estoque."""

import pytest
from fastapi.testclient import TestClient

from estoque.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "estoque.db"))
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "service": "estoque"}


def test_seed_loaded(client):
    assert len(client.get("/products").json()) >= 8


def test_get_product_computes_availability(client):
    # MON-27P-003: 45 físico, 10 reservados no seed.
    product = client.get("/products/MON-27P-003").json()
    assert product["on_hand"] == 45
    assert product["reserved"] == 10
    assert product["available"] == 35


def test_get_product_not_found(client):
    resp = client.get("/products/SKU-INEXISTENTE")
    assert resp.status_code == 404
    assert resp.json()["error"] == "product_not_found"


def test_reservation_happy_path_updates_availability(client):
    resp = client.post("/reservations", json={"sku": "CAD-ERG-001", "quantity": 5})
    assert resp.status_code == 201
    assert client.get("/products/CAD-ERG-001").json()["available"] == 27


def test_reservation_above_available_returns_422(client):
    # NTB-DEV-004: 6 físico, 2 reservados → 4 disponíveis.
    resp = client.post("/reservations", json={"sku": "NTB-DEV-004", "quantity": 5})
    assert resp.status_code == 422
    body = resp.json()
    assert body["error"] == "insufficient_stock"
    assert {"error", "detail", "rule"} <= body.keys()


def test_release_reservation_restores_availability(client):
    created = client.post("/reservations", json={"sku": "HEA-BTH-009", "quantity": 10}).json()
    assert client.get("/products/HEA-BTH-009").json()["available"] == 30
    released = client.post(f"/reservations/{created['id']}/release")
    assert released.status_code == 200
    assert released.json()["status"] == "liberada"
    assert client.get("/products/HEA-BTH-009").json()["available"] == 40


def test_release_twice_returns_422(client):
    created = client.post("/reservations", json={"sku": "HEA-BTH-009", "quantity": 1}).json()
    client.post(f"/reservations/{created['id']}/release")
    resp = client.post(f"/reservations/{created['id']}/release")
    assert resp.status_code == 422
    assert resp.json()["error"] == "reservation_already_released"


def test_replenishment_lists_low_stock_skus(client):
    items = {i["sku"]: i for i in client.get("/replenishment").json()}
    # MES-ELE-002: disponível 8 < ponto 12 → sugestão 16; NTB-DEV-004: disponível 4 < 8 → 12.
    assert items["MES-ELE-002"]["suggested_quantity"] == 16
    assert items["NTB-DEV-004"]["suggested_quantity"] == 12
    assert "CAD-ERG-001" not in items
