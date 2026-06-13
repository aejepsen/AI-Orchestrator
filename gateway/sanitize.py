"""Sanitização anti-injection no boundary do gateway.

A pergunta do usuário é input não confiável. O gateway monta prompts em
ChatML (Qwen/Ollama) e envolve a pergunta em tags <user_question>; este
módulo remove as fugas estruturais que permitiriam ao texto escapar desse
invólucro (padrão validado no AI-Tractor):

1. Tokens especiais de template ChatML (<|im_start|>, <|im_end|>,
   <|endoftext|>, qualquer <|...|>) — podem forjar um turno system/assistant.
2. As próprias tags de wrapper usadas na montagem dos prompts
   (</user_question>, <plan>, <agent_answers>...) — fechar a tag cedo
   permite que texto injetado se passe por instrução confiável.

Injeções semânticas ("ignore as instruções...") NÃO são reescritas:
mutilar keywords destrói perguntas legítimas; a defesa para essa classe é
o system prompt + isolamento por tag + least-privilege de tools.
"""

from __future__ import annotations

import re

# Qualquer token especial estilo ChatML: <|im_start|>, <|endoftext|>, etc.
_SPECIAL_TOKEN = re.compile(r"<\|[^|>]{0,32}\|>")

# Tags de wrapper usadas na montagem dos prompts do gateway, case-insensitive
# e tolerantes a espaços internos: </user_question>, < / PLAN >, <agent_answers>...
_WRAPPER_TAG = re.compile(
    r"<\s*/?\s*(user_question|user_input|plan|agent_answers|context)\s*>",
    re.IGNORECASE,
)


_INJECTION_PATTERNS = [
    re.compile(r"ignore.*instru[çc]", re.IGNORECASE),
    re.compile(r"ignore.*instruct", re.IGNORECASE),
    re.compile(r"ignore.*previous", re.IGNORECASE),
    re.compile(r"ignore.*above", re.IGNORECASE),
    re.compile(r"esque[çc]a.*tudo", re.IGNORECASE),
    re.compile(r"forget.*everything", re.IGNORECASE),
    re.compile(r"voc[eê] agora [eé]", re.IGNORECASE),
    re.compile(r"you are now", re.IGNORECASE),
    re.compile(r"act as", re.IGNORECASE),
    re.compile(r"atue como", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"system message", re.IGNORECASE),
    re.compile(r"revel.*instru[çc]", re.IGNORECASE),
    re.compile(r"reveal.*instruct", re.IGNORECASE),
]


def flag_injection(text: str) -> bool:
    """Retorna True se o texto contém padrão semântico de injection."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def sanitize_question(text: str) -> str:
    """Remove sequências de escape estrutural do texto não confiável."""
    text = _SPECIAL_TOKEN.sub(" ", text)
    text = _WRAPPER_TAG.sub(" ", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()
