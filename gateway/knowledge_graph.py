"""Camada B — Teia de Signos: Knowledge Graph via Neo4j.

Expande o contexto de entidades além do que busca vetorial alcança,
usando travessia de relações no grafo de conhecimento.

Integração: registrado como virtual tool no ToolRegistry. O subagente
chama `expand_context` quando detecta necessidade de relação cruzada.
O CircuitBreaker do registry protege contra Neo4j fora.

Princípios:
- Tool call, não nó do grafo — arquitetura limpa, opt-in por domínio.
- LIMIT por domínio (não global) — distribuição previsível.
- Degradação graceful: Neo4j fora → tool retorna lista vazia, agente continua.
- Seed script separado para popular entidades iniciais.
"""

from __future__ import annotations

import logging
from typing import Any

from gateway.tools.registry import ToolSpec

logger = logging.getLogger(__name__)

# Domínios que se beneficiam de expansão por grafo.
# RH raramente precisa — relações são simples (funcionário → departamento).
GRAPH_ENABLED_DOMAINS = ("estoque", "vendas", "financas")

_EXPAND_TOOL = ToolSpec(
    name="expand_context",
    description=(
        "Expande o contexto de uma entidade buscando relações no grafo de conhecimento. "
        "Use quando precisar descobrir dependências, fornecedores, produtos relacionados "
        "ou vínculos entre entidades de diferentes domínios. "
        "Retorna entidades relacionadas com tipo de relação e domínio de origem."
    ),
    method="VIRTUAL",
    path="",
    path_params=(),
    query_params=(),
    body_params=("entity_name", "entity_type", "target_domain"),
    parameters={
        "type": "object",
        "properties": {
            "entity_name": {
                "type": "string",
                "description": "Nome ou identificador da entidade (ex: 'CAD-001', 'Maria Silva').",
            },
            "entity_type": {
                "type": "string",
                "enum": ["produto", "funcionario", "fornecedor", "departamento", "pedido", "cliente"],
                "description": "Tipo da entidade no grafo.",
            },
            "target_domain": {
                "type": "string",
                "enum": ["estoque", "vendas", "financas", "rh"],
                "description": "Domínio alvo para filtrar relações. Omitir para todos os domínios.",
                "default": "",
            },
        },
        "required": ["entity_name", "entity_type"],
    },
)

# Cypher parametrizado — LIMIT por domínio, DISTINCT, sem injeção.
# Match por name OU sku (enricher extrai SKUs, seed armazena nomes descritivos).
_EXPAND_CYPHER = """\
MATCH (e:Entity)
WHERE (e.name = $entity_name OR e.sku = $entity_name) AND e.type = $entity_type
WITH e
    MATCH (e)-[r*1..3]-(related:Entity)
    WHERE related <> e AND ($target_domain = '' OR related.domain = $target_domain)
    WITH related, [rel IN r | type(rel)] AS path_types, e
    // Prioriza cross-domain: entidades de domínio diferente do nó origem
    // aparecem primeiro no resultado (push-down no ORDER BY).
    WITH related, path_types, CASE WHEN related.domain = e.domain THEN 1 ELSE 0 END AS same_domain
    ORDER BY same_domain ASC, related.domain, related.name
    RETURN related.name AS name,
           related.type AS type,
           related.domain AS domain,
           path_types
    LIMIT $limit
"""

# LIMIT adaptativo: mais contexto quando menos domínios.
_LIMIT_PER_DOMAIN = 5
_LIMIT_SINGLE_DOMAIN = 10


class KnowledgeGraph:
    """Adapter Neo4j para expansão de contexto por travessia de grafo."""

    def __init__(self, uri: str, auth: tuple[str, str] | None = None) -> None:
        self._uri = uri
        self._auth = auth
        self._driver: Any = None  # lazy
        self._available: bool | None = None

    @property
    def available(self) -> bool:
        """True após driver conectado com sucesso."""
        if self._available is not None:
            return self._available
        return self._ensure_driver()

    def _ensure_driver(self) -> bool:
        """Lazy-load do driver Neo4j. Nunca bloqueia se indisponível."""
        if self._driver is not None:
            return self._available or False
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(self._uri, auth=self._auth)
            self._driver.verify_connectivity()
            self._available = True
            logger.info("KnowledgeGraph: conectado a %s", self._uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning("KnowledgeGraph: indisponível (%s) — expand_context retornará vazio", exc)
            self._available = False
        return self._available or False

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            self._available = None

    def expand(self, entity_name: str, entity_type: str, target_domain: str = "") -> dict[str, Any]:
        """Executa travessia no grafo e retorna entidades relacionadas.

        Retorna formato compatível com tool result: {"status": int, "body": ...}.
        """
        if not self._ensure_driver():
            return {"status": 200, "body": {"related": [], "note": "Knowledge graph indisponível."}}

        limit = _LIMIT_SINGLE_DOMAIN if target_domain else _LIMIT_PER_DOMAIN * 4

        try:
            with self._driver.session() as session:
                result = session.run(
                    _EXPAND_CYPHER,
                    entity_name=entity_name,
                    entity_type=entity_type,
                    target_domain=target_domain,
                    limit=limit,
                )
                related = [
                    {
                        "name": record["name"],
                        "type": record["type"],
                        "domain": record["domain"],
                        "path": record["path_types"],
                    }
                    for record in result
                ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("KnowledgeGraph: query falhou (%s)", exc)
            return {"status": 200, "body": {"related": [], "note": f"Erro na consulta: {exc}"}}

        return {
            "status": 200,
            "body": {
                "entity": entity_name,
                "type": entity_type,
                "related": related,
                "count": len(related),
            },
        }


def get_expand_tool_spec() -> ToolSpec:
    """Retorna o ToolSpec para registro no ToolRegistry."""
    return _EXPAND_TOOL


def graph_enabled_domains() -> tuple[str, ...]:
    """Domínios que recebem a tool expand_context."""
    return GRAPH_ENABLED_DOMAINS
