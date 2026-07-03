"""RAG sobre documentos não estruturados (políticas internas) — extensão do backlog.

Corpus: `docs/policies/*.md` (versionado — as políticas derivam das regras REAIS
dos microsserviços e não podem contradizê-las). Ingestão offline
(`scripts/ingest_documents.py`) chunk por seção → SBERT → Qdrant collection
`documents`. Runtime: tool virtual `search_documents` disponível a TODOS os
domínios (política é leitura cross-domain; least-privilege preservado — a tool
só lê a collection de documentos).

Opt-in: RAG_DOCS_ENABLED=1. Degradação graceful: Qdrant/collection fora →
resposta 200 com nota, nunca derruba o agente (padrão do expand_context).
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Any

import httpx

from gateway.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

DOCUMENTS_COLLECTION = "documents"

_SEARCH_TOOL = ToolSpec(
    name="search_documents",
    description=(
        "Busca trechos das POLÍTICAS INTERNAS da empresa (alçadas de aprovação, "
        "descontos e comissões, férias, reembolsos, estoque e reservas). Use quando a "
        "pergunta é sobre REGRA/POLÍTICA ('qual o limite...', 'quem pode...', 'como "
        "funciona...'), não sobre dados de um registro específico. "
        "Resposta: lista de itens com document (string), section (string), text (string), score (number)."
    ),
    method="VIRTUAL",
    path="",
    path_params=(),
    query_params=(),
    body_params=("query",),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Pergunta ou termos sobre a política (ex.: 'limite de desconto do vendedor').",
            }
        },
        "required": ["query"],
    },
)


def get_search_tool_spec() -> ToolSpec:
    return _SEARCH_TOOL


def chunk_markdown(text: str, *, document: str) -> list[dict[str, str]]:
    """Chunk por seção de heading: cada bloco `#/##` vira um chunk coeso.

    Políticas são curtas e hierárquicas — seção inteira preserva a regra
    completa (limite + exceção) melhor que janelas de tamanho fixo.
    """
    chunks: list[dict[str, str]] = []
    title = document
    section = ""
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            chunks.append({"document": title, "section": section or title, "text": body})
        buffer.clear()

    for line in text.splitlines():
        heading = re.match(r"^(#{1,3})\s+(.*)", line)
        if heading:
            flush()
            if heading.group(1) == "#":
                title = heading.group(2).strip()
                section = ""
            else:
                section = heading.group(2).strip()
            continue
        buffer.append(line)
    flush()
    return chunks


def chunk_id(chunk: dict[str, str]) -> str:
    """UUID determinístico por conteúdo — upsert idempotente (padrão do router)."""
    digest = hashlib.sha256(
        f"{chunk['document']}|{chunk['section']}|{chunk['text']}".encode()
    ).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


class DocumentSearch:
    """Busca kNN na collection de documentos; executor da tool virtual."""

    def __init__(
        self,
        qdrant_url: str,
        embedder: Any,
        *,
        top_k: int = 3,
        client: httpx.Client | None = None,
        api_key: str | None = None,
    ) -> None:
        self._qdrant_url = qdrant_url.rstrip("/")
        self._embedder = embedder
        self._top_k = top_k
        headers = {"api-key": api_key} if api_key else {}
        self._client = client or httpx.Client(timeout=10.0, headers=headers)

    def search(self, query: str) -> dict[str, Any]:
        """Executor: {"status", "body"} como as tools HTTP; nunca levanta."""
        if not (query or "").strip():
            return {
                "status": 422,
                "body": {"error": "query_vazia", "detail": "Informe o que buscar na política."},
            }
        try:
            vector = self._embedder.embed([query], prefix_type="query")[0]
            response = self._client.post(
                f"{self._qdrant_url}/collections/{DOCUMENTS_COLLECTION}/points/search",
                json={"vector": vector, "limit": self._top_k, "with_payload": True},
            )
            response.raise_for_status()
            hits = response.json().get("result", [])
        except Exception as exc:  # noqa: BLE001 — tool degrada, agente segue
            logger.warning("search_documents indisponível (%s)", exc)
            return {
                "status": 200,
                "body": {"results": [], "note": "Base de políticas indisponível no momento."},
            }
        return {
            "status": 200,
            "body": {
                "results": [
                    {
                        "document": h.get("payload", {}).get("document", ""),
                        "section": h.get("payload", {}).get("section", ""),
                        "text": h.get("payload", {}).get("text", ""),
                        "score": round(h.get("score", 0.0), 4),
                    }
                    for h in hits
                ]
            },
        }
