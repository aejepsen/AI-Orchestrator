"""S4 — GraphRAG global mínimo: resumos de comunidades do KG como tool virtual.

GraphRAG completo (Edge et al., 2024) usa detecção de comunidades + resumos
pré-gerados para responder perguntas RELACIONAIS/GLOBAIS que a travessia
dirigida (expand_context, Camada B) não cobre — "quais clientes dependem de
fornecedores em atraso?". Esta é a versão mínima honesta para o KG do projeto:

- Offline: `scripts/build_kg_communities.py` roda Louvain no grafo do Neo4j e
  gera um resumo LLM por comunidade, gravado em JSON (mesmo padrão de artefato
  do injection classifier: volume ./models, fora do git).
- Runtime: tool virtual `summarize_community` serve os resumos pré-gerados —
  zero Neo4j e zero LLM no caminho da request. Opt-in: GRAPHRAG_ENABLED=1.

Degradação graceful: arquivo ausente/inválido → tool não é registrada.
"""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

from gateway.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

_SUMMARIZE_TOOL = ToolSpec(
    name="summarize_community",
    description=(
        "Resumo pré-computado de uma comunidade do grafo de conhecimento — grupo de "
        "entidades densamente conectadas entre domínios. Use para perguntas RELACIONAIS "
        "ou PANORÂMICAS ('como X se conecta ao restante?', 'qual o contexto em volta de Y?'). "
        "Informe uma entidade (ex.: 'Hotel Miramar', 'CAD-ERG-001') ou um domínio. "
        "Resposta: campos community_id, domains, entities, summary (texto)."
    ),
    method="VIRTUAL",
    path="",
    path_params=(),
    query_params=(),
    body_params=("entity", "domain"),
    parameters={
        "type": "object",
        "properties": {
            "entity": {
                "type": "string",
                "description": "Nome da entidade para localizar a comunidade dela (opcional).",
            },
            "domain": {
                "type": "string",
                "description": "Domínio (financas|rh|estoque|vendas) para listar comunidades (opcional).",
            },
        },
        "required": [],
    },
)


def get_summarize_tool_spec() -> ToolSpec:
    return _SUMMARIZE_TOOL


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


class CommunityIndex:
    """Índice em memória dos resumos de comunidade pré-gerados (JSON offline)."""

    def __init__(self, path: str) -> None:
        self._communities: list[dict] = []
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self._communities = [
                c for c in data.get("communities", []) if c.get("summary") and c.get("entities")
            ]
        except Exception as exc:  # noqa: BLE001 — artefato opcional
            logger.warning("CommunityIndex indisponível (%s) — summarize_community desativada", exc)

    @property
    def available(self) -> bool:
        return bool(self._communities)

    def lookup(self, entity: str = "", domain: str = "") -> dict:
        """Executor da tool: {"status", "body"} como as tools HTTP."""
        matches: list[dict] = []
        entity_norm = _norm(entity) if entity else ""
        domain_norm = _norm(domain) if domain else ""
        for community in self._communities:
            if entity_norm and any(entity_norm in _norm(e) for e in community["entities"]):
                matches.append(community)
                continue
            if domain_norm and domain_norm in [_norm(d) for d in community.get("domains", [])]:
                matches.append(community)
        if not entity_norm and not domain_norm:
            matches = self._communities
        if not matches:
            return {
                "status": 404,
                "body": {
                    "error": "community_not_found",
                    "detail": "Nenhuma comunidade contém essa entidade/domínio. "
                    "Tente o nome exato da entidade ou um dos domínios: financas, rh, estoque, vendas.",
                },
            }
        body = [
            {
                "community_id": c["id"],
                "domains": c.get("domains", []),
                "entities": c["entities"][:12],
                "summary": c["summary"],
            }
            for c in matches[:3]
        ]
        return {"status": 200, "body": body}
