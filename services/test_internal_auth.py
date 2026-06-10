"""Testes da autenticação interna (X-Internal-Key) — middleware comum aos 4 serviços.

Exaustivo no serviço Finanças (o middleware vive em `common.py`);
smoke nos outros 3 confirma que cada main.py registrou o middleware.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

KEY = "test-internal-key"

SERVICES = {
    "financas": ("financas.main", "/accounts"),
    "rh": ("rh.main", "/employees"),
    "estoque": ("estoque.main", "/products"),
    "vendas": ("vendas.main", "/orders"),
}


def _client(monkeypatch, tmp_path, service: str, *, key: str | None) -> TestClient:
    module_name, _ = SERVICES[service]
    monkeypatch.setenv("DB_PATH", str(tmp_path / f"{service}.db"))
    if key is None:
        monkeypatch.delenv("INTERNAL_API_KEY", raising=False)
    else:
        monkeypatch.setenv("INTERNAL_API_KEY", key)
    module = __import__(module_name, fromlist=["app"])
    return TestClient(module.app)


class TestFinancasExhaustive:
    def test_business_route_without_header_is_401_envelope(self, monkeypatch, tmp_path):
        with _client(monkeypatch, tmp_path, "financas", key=KEY) as client:
            resp = client.get("/accounts")
        assert resp.status_code == 401
        body = resp.json()
        assert body["error"] == "unauthorized"
        assert body["rule"] == "internal_api_key"
        assert body["detail"]

    def test_business_route_with_wrong_key_is_401(self, monkeypatch, tmp_path):
        with _client(monkeypatch, tmp_path, "financas", key=KEY) as client:
            resp = client.get("/accounts", headers={"X-Internal-Key": "wrong"})
        assert resp.status_code == 401
        assert resp.json()["error"] == "unauthorized"

    def test_business_route_with_correct_key_is_200(self, monkeypatch, tmp_path):
        with _client(monkeypatch, tmp_path, "financas", key=KEY) as client:
            resp = client.get("/accounts", headers={"X-Internal-Key": KEY})
        assert resp.status_code == 200

    def test_mutating_route_without_header_is_401(self, monkeypatch, tmp_path):
        with _client(monkeypatch, tmp_path, "financas", key=KEY) as client:
            resp = client.post("/accounts", json={})
        assert resp.status_code == 401

    @pytest.mark.parametrize("path", ["/health", "/openapi.json", "/docs"])
    def test_public_paths_exempt_without_header(self, monkeypatch, tmp_path, path):
        with _client(monkeypatch, tmp_path, "financas", key=KEY) as client:
            assert client.get(path).status_code == 200

    def test_without_env_service_runs_open(self, monkeypatch, tmp_path):
        with _client(monkeypatch, tmp_path, "financas", key=None) as client:
            assert client.get("/accounts").status_code == 200


@pytest.mark.parametrize("service", ["rh", "estoque", "vendas"])
class TestOtherServicesSmoke:
    def test_without_header_is_401(self, monkeypatch, tmp_path, service):
        _, path = SERVICES[service]
        with _client(monkeypatch, tmp_path, service, key=KEY) as client:
            resp = client.get(path)
        assert resp.status_code == 401
        assert resp.json()["error"] == "unauthorized"

    def test_with_correct_key_is_200(self, monkeypatch, tmp_path, service):
        _, path = SERVICES[service]
        with _client(monkeypatch, tmp_path, service, key=KEY) as client:
            assert client.get(path, headers={"X-Internal-Key": KEY}).status_code == 200

    def test_health_exempt(self, monkeypatch, tmp_path, service):
        with _client(monkeypatch, tmp_path, service, key=KEY) as client:
            assert client.get("/health").status_code == 200
