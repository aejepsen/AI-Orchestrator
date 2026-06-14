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
    model: str = field(default_factory=lambda: os.environ.get("MODEL", "qwen2.5:7b-instruct-q4_K_M"))
    service_urls: dict[str, str] = field(default_factory=_service_urls_from_env)
    llm_timeout_s: float = field(default_factory=lambda: float(os.environ.get("LLM_TIMEOUT_S", "300")))
    keep_alive: str = field(default_factory=lambda: os.environ.get("KEEP_ALIVE", "30m"))
    http_timeout_s: float = field(default_factory=lambda: float(os.environ.get("HTTP_TIMEOUT_S", "30")))
    agent_max_iters: int = field(default_factory=lambda: int(os.environ.get("AGENT_MAX_ITERS", "6")))
    agent_deadline_s: float = field(default_factory=lambda: float(os.environ.get("AGENT_DEADLINE_S", "600")))
    internal_api_key: str | None = field(default_factory=lambda: os.environ.get("INTERNAL_API_KEY"))
    # Semantic router (Qdrant + embeddings locais). Desabilita com SEMANTIC_ENABLED=0.
    qdrant_url: str = field(default_factory=lambda: os.environ.get("QDRANT_URL", "http://localhost:6333"))
    embed_model: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", "nomic-embed-text"))
    semantic_enabled: bool = field(
        default_factory=lambda: os.environ.get("SEMANTIC_ENABLED", "1") not in ("0", "false", "False")
    )
    semantic_threshold: float = field(default_factory=lambda: float(os.environ.get("SEMANTIC_THRESHOLD", "0.92")))
    semantic_top_k: int = field(default_factory=lambda: int(os.environ.get("SEMANTIC_TOP_K", "5")))
    routing_examples_path: str = field(
        default_factory=lambda: os.environ.get("ROUTING_EXAMPLES_PATH", "evals/golden_routing.jsonl")
    )
    # SBERT embeddings (CPU). Fallback para Ollama se sentence-transformers não instalado.
    sbert_model: str = field(
        default_factory=lambda: os.environ.get("SBERT_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    )
    sbert_cache_dir: str = field(default_factory=lambda: os.environ.get("SBERT_CACHE_DIR", "/app/models"))
    # Langfuse (observabilidade LLM). Desabilita com LANGFUSE_ENABLED=0.
    langfuse_enabled: bool = field(
        default_factory=lambda: os.environ.get("LANGFUSE_ENABLED", "1") not in ("0", "false", "False")
    )
    langfuse_host: str = field(default_factory=lambda: os.environ.get("LANGFUSE_HOST", "http://localhost:3100"))
    langfuse_public_key: str = field(default_factory=lambda: os.environ.get("LANGFUSE_PUBLIC_KEY", "pk-lf-local"))
    langfuse_secret_key: str = field(default_factory=lambda: os.environ.get("LANGFUSE_SECRET_KEY", "sk-lf-local"))
    # Estado conversacional (multi-turn)
    thread_db_path: str = field(default_factory=lambda: os.environ.get("THREAD_DB_PATH", "/tmp/threads.db"))
    # Injection Detector (BERTimbau fine-tunado). Desabilita com INJECTION_DETECTOR_ENABLED=0.
    injection_model: str = field(
        default_factory=lambda: os.environ.get("INJECTION_MODEL", "models/injection_classifier")
    )
    injection_threshold: float = field(
        default_factory=lambda: float(os.environ.get("INJECTION_THRESHOLD", "0.7"))
    )
    injection_detector_enabled: bool = field(
        default_factory=lambda: os.environ.get("INJECTION_DETECTOR_ENABLED", "1") not in ("0", "false", "False")
    )
    # Pool dedicado para execução do grafo (evita competir com asyncio default pool).
    max_graph_workers: int = field(default_factory=lambda: int(os.environ.get("MAX_GRAPH_WORKERS", "4")))
    # Timeout global de request SSE (segundos). Independente do LLM_TIMEOUT_S.
    request_timeout_s: float = field(default_factory=lambda: float(os.environ.get("REQUEST_TIMEOUT_S", "600")))


def load_settings() -> Settings:
    """Carrega as configurações do ambiente (chamar uma vez no boot)."""
    return Settings()
