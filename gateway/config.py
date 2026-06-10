"""Configuração central do gateway.

Tudo é sobrescrevível por variável de ambiente — em desenvolvimento os
serviços rodam em localhost:8101..8104; na Fase 3 (Docker) viram hostnames
da rede interna (ex.: SERVICE_URL_FINANCAS=http://financas:8000).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

DOMAINS: tuple[str, ...] = ("financas", "rh", "estoque", "vendas")

_DEFAULT_SERVICE_URLS: dict[str, str] = {
    "financas": "http://localhost:8101",
    "rh": "http://localhost:8102",
    "estoque": "http://localhost:8103",
    "vendas": "http://localhost:8104",
}


def _service_urls_from_env() -> dict[str, str]:
    return {
        domain: os.environ.get(f"SERVICE_URL_{domain.upper()}", default)
        for domain, default in _DEFAULT_SERVICE_URLS.items()
    }


@dataclass(frozen=True)
class Settings:
    """Configuração imutável carregada uma vez no boot."""

    ollama_url: str = field(default_factory=lambda: os.environ.get("OLLAMA_URL", "http://localhost:11435"))
    model: str = field(default_factory=lambda: os.environ.get("MODEL", "qwen3:30b-a3b"))
    service_urls: dict[str, str] = field(default_factory=_service_urls_from_env)
    llm_timeout_s: float = field(default_factory=lambda: float(os.environ.get("LLM_TIMEOUT_S", "300")))
    http_timeout_s: float = field(default_factory=lambda: float(os.environ.get("HTTP_TIMEOUT_S", "30")))
    agent_max_iters: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_ITERS", "6")))
    agent_deadline_s: float = field(default_factory=lambda: float(os.environ.get("AGENT_DEADLINE_S", "600")))
    internal_api_key: str | None = field(default_factory=lambda: os.environ.get("INTERNAL_API_KEY"))


def load_settings() -> Settings:
    """Carrega as configurações do ambiente (chamar uma vez no boot)."""
    return Settings()
