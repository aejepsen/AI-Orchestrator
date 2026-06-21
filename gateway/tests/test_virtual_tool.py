"""Testes do mecanismo de virtual tools no ToolRegistry (Semiose — Camada B).

Virtual tools executam lógica local (ex.: Neo4j expand_context) em vez de HTTP.
Nenhum teste toca a rede.
"""

from __future__ import annotations

from typing import Any

from gateway.tools.registry import ToolRegistry, ToolSpec


def _virtual_spec() -> ToolSpec:
    return ToolSpec(
        name="expand_context",
        description="expansão de contexto via grafo",
        method="VIRTUAL",
        path="",
        path_params=(),
        query_params=(),
        body_params=("entity_name",),
        parameters={
            "type": "object",
            "properties": {"entity_name": {"type": "string"}},
            "required": ["entity_name"],
        },
    )


def test_tools_for_inclui_virtual_tool():
    reg = ToolRegistry({"estoque": "http://x"})
    reg.preload("estoque", {})  # evita HTTP para specs reais
    reg.register_virtual_tool(("estoque",), _virtual_spec(), lambda args: {"status": 200, "body": args})

    names = [t["function"]["name"] for t in reg.tools_for("estoque")]
    assert "expand_context" in names


def test_virtual_tool_escopada_aos_dominios():
    reg = ToolRegistry({"estoque": "http://x", "rh": "http://y"})
    reg.preload("estoque", {})
    reg.preload("rh", {})
    reg.register_virtual_tool(("estoque",), _virtual_spec(), lambda args: {"status": 200, "body": args})

    estoque_names = [t["function"]["name"] for t in reg.tools_for("estoque")]
    rh_names = [t["function"]["name"] for t in reg.tools_for("rh")]
    assert "expand_context" in estoque_names
    assert "expand_context" not in rh_names


def test_execute_virtual_tool_dispatch_para_executor():
    captured: dict[str, Any] = {}

    def executor(args: dict[str, Any]) -> dict[str, Any]:
        captured.update(args)
        return {"status": 200, "body": {"echo": args}}

    reg = ToolRegistry({"estoque": "http://x"})
    reg.register_virtual_tool(("estoque",), _virtual_spec(), executor)

    out = reg.execute("estoque", "expand_context", {"entity_name": "CAD-ERG-001"})
    assert out["status"] == 200
    assert out["body"]["echo"] == {"entity_name": "CAD-ERG-001"}
    assert captured == {"entity_name": "CAD-ERG-001"}


def test_execute_virtual_tool_nao_faz_http():
    # Sem preload e sem servidor: se tentasse HTTP, levantaria. Virtual evita isso.
    reg = ToolRegistry({"estoque": "http://invalid.invalid"})
    reg.register_virtual_tool(("estoque",), _virtual_spec(), lambda args: {"status": 200, "body": {}})

    out = reg.execute("estoque", "expand_context", {"entity_name": "X"})
    assert out["status"] == 200


def test_execute_virtual_tool_erro_retorna_degradado():
    def boom(args: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("falha no executor")

    reg = ToolRegistry({"estoque": "http://x"})
    reg.register_virtual_tool(("estoque",), _virtual_spec(), boom)

    out = reg.execute("estoque", "expand_context", {})
    assert out["status"] == 0
    assert out["body"]["error"] == "service_unavailable"
    assert out["body"]["rule"] == "transport_error"
