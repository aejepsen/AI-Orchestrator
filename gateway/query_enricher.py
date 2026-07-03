"""Enriquecimento contextual da query (Camada A — Semiose).

Transforma a query crua em query contextualizada usando metadata estruturada
do estado conversacional. Nunca concatena texto cru do history — só usa campos
validados (_last_route, entidades extraídas por regex).

Posição no grafo: sanitize → enrich → classify.

Princípios:
- Harness-first: zero chamada LLM, determinístico.
- Segurança: enricher só injeta metadata controlada, nunca user input.
- Degradação graceful: sem contexto → retorna query original.
- spaCy opt-in (SPACY_ENABLED): lazy-load, só quando regex não encontra nada.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from gateway.router import _DOMAIN_KEYWORDS, _normalize

# S5 — Multi-query expansion (modo LLM, opt-in via MULTI_QUERY_ENABLED).
_MULTI_QUERY_SYSTEM = """Você reformula perguntas corporativas para busca semântica.
Gere {n} reformulações CURTAS da pergunta do usuário que preservem exatamente a
intenção e TODAS as entidades (nomes, SKUs, valores, datas, departamentos).
Não responda a pergunta. Não adicione informação nova.
Responda APENAS JSON: {{"variants": ["...", "..."]}}"""


def expand_query_llm(llm: Any, question: str, *, n: int = 2) -> list[str]:
    """N paráfrases da pergunta via LLM local (S5). Lança em falha — o caller
    (SemanticRouter.route) trata como opcional e cai no fallback."""
    response = llm.chat(
        [
            {"role": "system", "content": _MULTI_QUERY_SYSTEM.format(n=n)},
            {"role": "user", "content": question},
        ],
        format="json",
    )
    data = json.loads(response.content)
    variants = data.get("variants")
    if not isinstance(variants, list):
        raise ValueError(f"expansão sem campo 'variants': {response.content[:120]!r}")
    return [str(v) for v in variants][:n]

# ---------------------------------------------------------------------------
# Regex de entidades estruturadas (patterns conhecidos dos 4 domínios)
# ---------------------------------------------------------------------------

# SKU: formato de 3+ partes do seed (ex.: CAD-ERG-001, MON-27P-003). Restritivo
# por design — evita falsos positivos com siglas genéricas (ex.: "ABC-123").
# Limitação conhecida: SKUs de 2 partes (ex.: "MON-027") NÃO casam; se novos
# formatos surgirem, ampliar este pattern e atualizar _infer_entity_type no eval.
_SKU_RE = re.compile(r"\b[A-Z]{2,5}(?:-[A-Z0-9]{2,5})+-\d{2,5}\b")
_CPF_RE = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
_CNPJ_RE = re.compile(r"\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b")
_MONEY_RE = re.compile(r"R\$\s*[\d.,]+")
_ORDER_ID_RE = re.compile(r"\b(?:pedido|PED)\s*#?\s*(\d{3,})\b", re.IGNORECASE)

_ENTITY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("sku", _SKU_RE),
    ("cpf", _CPF_RE),
    ("cnpj", _CNPJ_RE),
    ("valor", _MONEY_RE),
    ("pedido", _ORDER_ID_RE),
]


# ---------------------------------------------------------------------------
# spaCy extractor (lazy, opt-in)
# ---------------------------------------------------------------------------

class _SpacyExtractor:
    """Carrega pt_core_news_sm lazily. Nunca bloqueia se indisponível."""

    def __init__(self) -> None:
        self._nlp: Any = None
        self._available: bool | None = None

    def _ensure_model(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import spacy
            self._nlp = spacy.load("pt_core_news_sm")
            self._available = True
        except Exception:  # noqa: BLE001
            self._available = False
        return self._available

    def extract(self, text: str) -> list[str]:
        if not self._ensure_model():
            return []
        doc = self._nlp(text)
        return [ent.text for ent in doc.ents if ent.label_ in ("PER", "ORG", "LOC")]


_spacy = _SpacyExtractor()


# ---------------------------------------------------------------------------
# ContextSignal — metadata estruturada extraída do state
# ---------------------------------------------------------------------------

@dataclass
class ContextSignal:
    """Sinais de contexto extraídos do GraphState para enriquecimento."""

    last_domain: str | None = None
    recent_entities: list[str] = field(default_factory=list)
    enriched: bool = False


def _regex_extract(text: str) -> list[str]:
    """Extrai entidades estruturadas por regex (SKU, CPF, CNPJ, valores, pedidos)."""
    entities: list[str] = []
    for _label, pattern in _ENTITY_PATTERNS:
        entities.extend(pattern.findall(text))
    return entities


def _kg_entity_type(entity: str) -> str | None:
    """Infere o tipo de nó do KG a partir do padrão da entidade extraída."""
    if _SKU_RE.match(entity):
        return "produto"
    if _CPF_RE.match(entity):
        return "funcionario"
    if _ORDER_ID_RE.match(entity):
        return "pedido"
    return None


def kg_neighbors(entities: list[str], kg: Any, *, limit: int = 6) -> list[str]:
    """Vizinhos 1-hop das entidades resolvidas no KG (Semiose — Camada A+B).

    Determinístico e graceful: KG fora/entidade não resolvida → lista vazia.
    Retorna strings compactas "nome (tipo/domínio)" sem duplicatas, ordem estável.
    """
    if not kg or not entities:
        return []
    facts: list[str] = []
    for ent in entities[:3]:
        etype = _kg_entity_type(ent)
        if etype is None:
            continue
        try:
            result = kg.expand(ent, etype)
        except Exception:  # noqa: BLE001 — KG nunca quebra o pipeline
            continue
        for rel in result.get("body", {}).get("related", []):
            facts.append(f"{rel['name']} ({rel['type']}/{rel['domain']})")
    seen: set[str] = set()
    deduped: list[str] = []
    for fact in facts:
        if fact not in seen:
            seen.add(fact)
            deduped.append(fact)
    return deduped[:limit]


# Entidades estruturadas que implicam domínio (complementa _DOMAIN_KEYWORDS).
_ENTITY_DOMAIN_MAP: dict[str, str] = {
    "sku": "estoque",
    "pedido": "vendas",
    "cpf": "rh",
}

# Vocabulário cross-domain: keywords que aparecem naturalmente em domínios
# adjacentes e NÃO indicam troca de tópico. Ex: "SKU" em vendas é detalhe
# de pedido (order_items), não intenção de ir para estoque.
_CROSS_DOMAIN_VOCAB: dict[str, frozenset[str]] = {
    "vendas": frozenset({"sku", "aprovacao"}),      # order_items referenciam SKUs; desconto tem aprovação
    "estoque": frozenset({"cliente"}),                # reservas podem mencionar clientes
    "rh": frozenset({"vendas", "venda", "financeiro", "engenharia", "operacoes", "rh"}),  # nomes de departamento
    "financas": frozenset(),
}


def _has_strong_conflict(question: str, last_domain: str) -> bool:
    """Se a query tem keywords fortes de OUTRO domínio, há troca de tópico.

    Duas fontes de conflito:
    1. Keywords textuais (_DOMAIN_KEYWORDS do router — source of truth único).
       Exceção: keywords que pertencem ao vocabulário cross-domain do domínio
       atual (_CROSS_DOMAIN_VOCAB) — aparecem naturalmente sem indicar troca.
    2. Entidades estruturadas (SKU → estoque, pedido → vendas, CPF → RH).
    """
    normalized = _normalize(question)
    cross_vocab = _CROSS_DOMAIN_VOCAB.get(last_domain, frozenset())

    # 1. Keywords textuais (excluindo vocabulário cross-domain do domínio atual)
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if domain == last_domain:
            continue
        for kw in keywords:
            if kw in cross_vocab:
                continue
            if re.search(rf"(?<![a-z]){re.escape(kw)}", normalized):
                return True

    # 2. Entidades estruturadas → domínio implícito
    for label, pattern in _ENTITY_PATTERNS:
        if pattern.search(question) and label in _ENTITY_DOMAIN_MAP:
            implied_domain = _ENTITY_DOMAIN_MAP[label]
            if implied_domain != last_domain:
                return True

    return False


def gather_signals(state: dict[str, Any], *, spacy_enabled: bool = True) -> ContextSignal:
    """Extrai sinais de contexto do GraphState.

    Prioridade para _last_route (validado pelo classifier no turno anterior).
    Fallback: inferir domínio de keywords no history (menos confiável).
    """
    signals = ContextSignal()
    history = state.get("history") or []

    # Prioridade: usar route do turno anterior (já validado pelo classifier).
    last_route = state.get("_last_route")
    if last_route and last_route.get("domains"):
        signals.last_domain = last_route["domains"][0]

    # Fallback: inferir de keywords no último turno de user no history.
    if not signals.last_domain and history:
        user_msgs = [m for m in history if m.get("role") == "user"]
        if user_msgs:
            last_user = user_msgs[-1].get("content", "")
            normalized = _normalize(last_user)
            # 1. Keywords textuais
            for domain, keywords in _DOMAIN_KEYWORDS.items():
                if any(re.search(rf"(?<![a-z]){re.escape(kw)}", normalized) for kw in keywords):
                    signals.last_domain = domain
                    break
            # 2. Entidades estruturadas no history → domínio implícito
            if not signals.last_domain:
                for label, pattern in _ENTITY_PATTERNS:
                    if pattern.search(last_user) and label in _ENTITY_DOMAIN_MAP:
                        signals.last_domain = _ENTITY_DOMAIN_MAP[label]
                        break

    # Entidades: extrair da query sanitizada (já no state), não do history.
    sanitized = state.get("sanitized", "")
    signals.recent_entities = _regex_extract(sanitized)

    # spaCy fallback: só se regex não encontrou nada E feature habilitada.
    if not signals.recent_entities and spacy_enabled:
        spacy_entities = _spacy.extract(sanitized)
        if spacy_entities:
            signals.recent_entities = spacy_entities

    return signals


def enrich_query(question: str, signals: ContextSignal, kg: Any = None) -> tuple[str, bool]:
    """Monta query contextualizada com metadata estruturada.

    Retorna (query_enriquecida, foi_enriquecida).

    NUNCA concatena texto cru do history — só usa campos estruturados
    (domínio validado, entidades extraídas por regex/spaCy).

    `kg` (Semiose — Camada A+B, opt-in): se fornecido, injeta os vizinhos
    1-hop das entidades resolvidas no grafo, dando contexto relacional
    determinístico ao classificador. Graceful: KG fora → ignorado.
    """
    # Sem sinais úteis → retorna original.
    if not signals.last_domain and not signals.recent_entities:
        return question, False

    # Troca de tópico: query tem keywords fortes de outro domínio → dropar enriquecimento.
    if signals.last_domain and _has_strong_conflict(question, signals.last_domain):
        return question, False

    # Montar prefixo com metadata controlada.
    parts: list[str] = []
    if signals.last_domain:
        parts.append(f"domínio: {signals.last_domain}")
    if signals.recent_entities:
        parts.append(f"entidades: {', '.join(signals.recent_entities[:5])}")

    # Realimentação do KG: vizinhos 1-hop das entidades (determinístico, graceful).
    if kg is not None and signals.recent_entities:
        neighbors = kg_neighbors(signals.recent_entities, kg)
        if neighbors:
            parts.append(f"relacionado: {', '.join(neighbors)}")

    if not parts:
        return question, False

    prefix = f"[{'; '.join(parts)}] "
    return f"{prefix}{question}", True
