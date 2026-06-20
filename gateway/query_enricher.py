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

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from gateway.router import _DOMAIN_KEYWORDS, _normalize

# ---------------------------------------------------------------------------
# Regex de entidades estruturadas (patterns conhecidos dos 4 domínios)
# ---------------------------------------------------------------------------

_SKU_RE = re.compile(r"\b[A-Z]{2,5}-\d{2,5}\b")
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


def _has_strong_conflict(question: str, last_domain: str) -> bool:
    """Se a query tem keywords fortes de OUTRO domínio, há troca de tópico.

    Importa _DOMAIN_KEYWORDS do router (source of truth único) — compartilha
    dados, não pipeline. Zero duplicação.
    """
    normalized = _normalize(question)
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        if domain == last_domain:
            continue
        if any(re.search(rf"(?<![a-z]){re.escape(kw)}", normalized) for kw in keywords):
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
            for domain, keywords in _DOMAIN_KEYWORDS.items():
                if any(re.search(rf"(?<![a-z]){re.escape(kw)}", normalized) for kw in keywords):
                    signals.last_domain = domain
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


def enrich_query(question: str, signals: ContextSignal) -> tuple[str, bool]:
    """Monta query contextualizada com metadata estruturada.

    Retorna (query_enriquecida, foi_enriquecida).

    NUNCA concatena texto cru do history — só usa campos estruturados
    (domínio validado, entidades extraídas por regex/spaCy).
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

    if not parts:
        return question, False

    prefix = f"[{'; '.join(parts)}] "
    return f"{prefix}{question}", True
