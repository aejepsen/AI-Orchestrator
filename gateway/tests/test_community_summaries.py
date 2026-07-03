"""Testes do S4 — summarize_community (resumos de comunidade pré-gerados)."""

from __future__ import annotations

import json
from pathlib import Path

from gateway.community_summaries import CommunityIndex, get_summarize_tool_spec


def _index(tmp_path: Path) -> CommunityIndex:
    data = {
        "communities": [
            {
                "id": 0,
                "size": 4,
                "domains": ["vendas", "estoque"],
                "entities": ["Hotel Miramar", "Monitor 27 4K", "Camila Rocha", "Eletrônicos"],
                "summary": "O Hotel Miramar compra eletrônicos vendidos pela Camila Rocha.",
            },
            {
                "id": 1,
                "size": 5,
                "domains": ["rh", "financas"],
                "entities": ["Carlos Souza", "Engenharia", "diretor"],
                "summary": "Funcionários de Engenharia e as alçadas de aprovação.",
            },
        ]
    }
    path = tmp_path / "kg_communities.json"
    path.write_text(json.dumps(data, ensure_ascii=False))
    return CommunityIndex(str(path))


def test_lookup_por_entidade(tmp_path: Path) -> None:
    result = _index(tmp_path).lookup(entity="hotel miramar")
    assert result["status"] == 200
    assert result["body"][0]["community_id"] == 0
    assert "Hotel Miramar" in result["body"][0]["entities"]


def test_lookup_por_entidade_parcial_e_sem_acento(tmp_path: Path) -> None:
    result = _index(tmp_path).lookup(entity="miramar")
    assert result["status"] == 200
    result = _index(tmp_path).lookup(entity="eletronicos")
    assert result["status"] == 200


def test_lookup_por_dominio(tmp_path: Path) -> None:
    result = _index(tmp_path).lookup(domain="rh")
    assert result["status"] == 200
    assert result["body"][0]["community_id"] == 1


def test_nao_encontrado_devolve_404_acionavel(tmp_path: Path) -> None:
    result = _index(tmp_path).lookup(entity="entidade fantasma")
    assert result["status"] == 404
    assert "detail" in result["body"]


def test_arquivo_ausente_degrada(tmp_path: Path) -> None:
    index = CommunityIndex(str(tmp_path / "nao_existe.json"))
    assert not index.available


def test_spec_e_virtual_com_campos_de_resposta() -> None:
    spec = get_summarize_tool_spec()
    assert spec.method == "VIRTUAL"
    assert spec.name == "summarize_community"
    assert "Resposta: campos" in spec.description  # padrão do resíduo estoque-03
