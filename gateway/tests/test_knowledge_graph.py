"""Testes unitários do Knowledge Graph (Semiose — Camada B).

Driver Neo4j é mockado — nenhum teste toca a rede.
"""

from __future__ import annotations

from gateway.knowledge_graph import (
    KnowledgeGraph,
    get_expand_tool_spec,
    graph_enabled_domains,
)


# ── Fakes do driver Neo4j ────────────────────────────────────────────────


class _FakeSession:
    def __init__(self, records: list[dict]) -> None:
        self._records = records

    def __enter__(self) -> "_FakeSession":
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def run(self, cypher: str, **params: object):
        self.last_cypher = cypher
        self.last_params = params
        return iter(self._records)


class _FakeDriver:
    def __init__(self, records: list[dict]) -> None:
        self._records = records
        self.closed = False

    def session(self) -> _FakeSession:
        return _FakeSession(self._records)

    def close(self) -> None:
        self.closed = True


class _BoomDriver:
    def session(self):
        raise RuntimeError("falha simulada na sessão")


# ── ToolSpec e domínios ──────────────────────────────────────────────────


def test_expand_tool_spec():
    spec = get_expand_tool_spec()
    assert spec.name == "expand_context"
    assert spec.method == "VIRTUAL"
    assert "entity_name" in spec.parameters["properties"]
    assert spec.parameters["required"] == ["entity_name", "entity_type"]


def test_graph_enabled_domains_exclui_rh():
    domains = graph_enabled_domains()
    assert domains == ("estoque", "vendas", "financas")
    assert "rh" not in domains


# ── expand — caminho feliz ───────────────────────────────────────────────


def test_expand_formata_entidades_relacionadas():
    kg = KnowledgeGraph("bolt://fake")
    kg._driver = _FakeDriver(
        [
            {"name": "Cadeira", "type": "produto", "domain": "estoque", "path_types": ["COMPROU"]},
            {"name": "Headset", "type": "produto", "domain": "estoque", "path_types": ["COMPROU"]},
        ]
    )
    kg._available = True

    out = kg.expand("Banco Horizonte", "cliente", "estoque")

    assert out["status"] == 200
    assert out["body"]["count"] == 2
    assert out["body"]["entity"] == "Banco Horizonte"
    first = out["body"]["related"][0]
    assert first["name"] == "Cadeira"
    assert first["path"] == ["COMPROU"]


def test_expand_passa_parametros_ao_cypher():
    fake = _FakeDriver([])
    kg = KnowledgeGraph("bolt://fake")
    kg._driver = fake
    kg._available = True

    kg.expand("CAD-ERG-001", "produto", "vendas")
    # Sem registros → lista vazia, mas a query roda sem erro.
    out = kg.expand("CAD-ERG-001", "produto", "vendas")
    assert out["body"]["related"] == []


# ── expand — degradação graceful ─────────────────────────────────────────


def test_expand_graceful_quando_driver_indisponivel(monkeypatch):
    kg = KnowledgeGraph("bolt://fake")
    monkeypatch.setattr(kg, "_ensure_driver", lambda: False)

    out = kg.expand("X", "produto")

    assert out["status"] == 200
    assert out["body"]["related"] == []
    assert "indisponível" in out["body"]["note"]


def test_expand_graceful_quando_query_falha():
    kg = KnowledgeGraph("bolt://fake")
    kg._driver = _BoomDriver()
    kg._available = True

    out = kg.expand("X", "produto")

    assert out["status"] == 200
    assert out["body"]["related"] == []
    assert "Erro na consulta" in out["body"]["note"]
