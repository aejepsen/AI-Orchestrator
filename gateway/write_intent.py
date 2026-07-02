"""Detecção determinística de intenção de escrita (Harness antes de Model).

Gate do HITL: `interrupt()` só deve pausar o grafo para operações que MUDAM
estado (criar/pagar/reservar/cancelar...). Consulta nunca exige confirmação.
O léxico deriva das write ops reais dos 4 microsserviços (POST/PUT/DELETE).

Postura de erro: falso negativo auto-aprova — não é falha de segurança, a
regra de negócio vive na API e o system prompt exige campos obrigatórios;
HITL é camada de governança. Frases nominais de consulta que contêm verbo
de escrita ("contas a pagar", "pedidos a concluir") são excluídas antes do
match para não pausar leitura.
"""

from __future__ import annotations

import re
import unicodedata

# Infinitivo + imperativo/presente (formas mais comuns em pedidos PT-BR).
# financas: criar conta, pagar, quitar, liquidar, aprovar despesa, excluir.
# rh: admitir, contratar, demitir, conceder férias, reembolsar, solicitar.
# estoque: cadastrar produto, reservar, liberar reserva, ajustar saldo.
# vendas: criar pedido, cancelar, concluir, aplicar desconto.
_WRITE_VERBS = (
    "criar", "crie",
    "cadastrar", "cadastre",
    "incluir", "inclua", "adicionar", "adicione",
    "registrar", "registre", "lancar", "lance",
    "atualizar", "atualize", "alterar", "altere", "editar", "edite", "mudar", "mude",
    "corrigir", "corrija", "ajustar", "ajuste",
    "excluir", "exclua", "deletar", "delete", "remover", "remova", "apagar", "apague",
    "cancelar", "cancele",
    "pagar", "pague", "quitar", "quite", "liquidar", "liquide", "receba",
    "aprovar", "aprove", "autorizar", "autorize",
    "reservar", "reserve", "liberar", "libere",
    "admitir", "admita", "contratar", "contrate", "demitir", "demita",
    "conceder", "conceda", "agendar", "agende", "marcar", "marque",
    "reembolsar", "reembolse", "solicitar", "solicite",
    "concluir", "conclua", "finalizar", "finalize",
    "efetuar", "efetue", "processar", "processe",
    "aplicar", "aplique",
)

# Frases nominais de CONSULTA que contêm verbo de escrita — nunca são write.
_NOUN_PHRASE_RE = re.compile(r"\ba (?:pagar|receber|liquidar|aprovar|concluir|vencer)\b")

_VERB_RE = re.compile(r"\b(?:" + "|".join(_WRITE_VERBS) + r")\b")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def detect_write_intent(question: str) -> tuple[bool, list[str]]:
    """(há intenção de escrita?, termos que dispararam) — determinístico, sem LLM."""
    normalized = _NOUN_PHRASE_RE.sub(" ", _normalize(question))
    matches = _VERB_RE.findall(normalized)
    return (bool(matches), sorted(set(matches)))
