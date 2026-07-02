"""Testes do `POST /admin/reset` (uso exclusivo dos evals — reset entre runs)."""

import pytest
from fastapi.testclient import TestClient

from estoque.main import app as estoque_app
from financas.main import app as financas_app
from rh.main import app as rh_app
from vendas.main import app as vendas_app

APPS = {
    "estoque": estoque_app,
    "financas": financas_app,
    "rh": rh_app,
    "vendas": vendas_app,
}


@pytest.fixture(params=sorted(APPS))
def service_client(request, tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / f"{request.param}.db"))
    with TestClient(APPS[request.param]) as client:
        yield request.param, client


def test_reset_retorna_ok(service_client):
    service, client = service_client
    response = client.post("/admin/reset")
    assert response.status_code == 200
    assert response.json() == {"status": "reset", "service": service}


def test_reset_fora_do_openapi(service_client):
    _, client = service_client
    paths = client.get("/openapi.json").json()["paths"]
    assert "/admin/reset" not in paths


def test_reset_exige_internal_key(service_client, monkeypatch):
    _, client = service_client
    monkeypatch.setenv("INTERNAL_API_KEY", "chave-teste")
    assert client.post("/admin/reset").status_code == 401
    assert client.post("/admin/reset", headers={"X-Internal-Key": "chave-teste"}).status_code == 200


def test_reset_descarta_estado_e_restaura_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "financas.db"))
    with TestClient(financas_app) as client:
        baseline = len(client.get("/accounts").json())
        created = client.post(
            "/accounts",
            json={
                "type": "pagar",
                "description": "Material de escritório",
                "counterparty": "Papelaria Central",
                "amount": 300.0,
                "due_date": "2026-08-01",
            },
        )
        assert created.status_code in (200, 201)
        assert len(client.get("/accounts").json()) == baseline + 1

        assert client.post("/admin/reset").status_code == 200
        assert len(client.get("/accounts").json()) == baseline


def test_reset_e_idempotente(service_client):
    _, client = service_client
    first = client.post("/admin/reset")
    second = client.post("/admin/reset")
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
