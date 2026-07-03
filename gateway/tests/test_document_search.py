"""Testes do RAG de políticas: chunker, ids idempotentes e executor da tool."""

from __future__ import annotations

import json

import httpx

from gateway.document_search import (
    DOCUMENTS_COLLECTION,
    DocumentSearch,
    chunk_id,
    chunk_markdown,
    get_search_tool_spec,
)
from gateway.tests.test_semantic_router import FakeEmbedder

POLICY = """# Política de Férias

## Direito e saldo

- 30 dias por ano aquisitivo, após 12 meses.

## Fracionamento

- Até 3 períodos; um deles com 14 dias ou mais.
"""


class TestChunker:
    def test_chunk_por_secao(self) -> None:
        chunks = chunk_markdown(POLICY, document="politica_ferias")
        assert [c["section"] for c in chunks] == ["Direito e saldo", "Fracionamento"]
        assert all(c["document"] == "Política de Férias" for c in chunks)
        assert "14 dias" in chunks[1]["text"]

    def test_texto_sem_headings_vira_um_chunk(self) -> None:
        chunks = chunk_markdown("regra única sem seções", document="doc")
        assert len(chunks) == 1
        assert chunks[0]["section"] == "doc"

    def test_chunk_id_deterministico(self) -> None:
        chunks = chunk_markdown(POLICY, document="politica_ferias")
        assert chunk_id(chunks[0]) == chunk_id(dict(chunks[0]))
        assert chunk_id(chunks[0]) != chunk_id(chunks[1])


def _search(hits: list[dict] | None, status: int = 200) -> DocumentSearch:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/collections/{DOCUMENTS_COLLECTION}/points/search"
        assert json.loads(request.content)["limit"] == 3
        return httpx.Response(status, json={"result": hits or []})

    return DocumentSearch(
        "http://qdrant.test",
        FakeEmbedder(),
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


class TestDocumentSearch:
    def test_resultados_formatados(self) -> None:
        hits = [
            {
                "score": 0.87,
                "payload": {"document": "Política de Férias", "section": "Fracionamento", "text": "Até 3 períodos."},
            }
        ]
        result = _search(hits).search("posso fracionar as férias?")
        assert result["status"] == 200
        top = result["body"]["results"][0]
        assert top["section"] == "Fracionamento" and top["score"] == 0.87

    def test_query_vazia_422(self) -> None:
        result = _search([]).search("  ")
        assert result["status"] == 422

    def test_qdrant_fora_degrada_com_nota(self) -> None:
        result = _search(None, status=500).search("limite de desconto")
        assert result["status"] == 200
        assert result["body"]["results"] == []
        assert "note" in result["body"]

    def test_spec_virtual_com_campos_de_resposta(self) -> None:
        spec = get_search_tool_spec()
        assert spec.method == "VIRTUAL"
        assert spec.name == "search_documents"
        assert "Resposta:" in spec.description
