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
    qdrant_api_key: str | None = field(default_factory=lambda: os.environ.get("QDRANT_API_KEY"))
    embed_model: str = field(default_factory=lambda: os.environ.get("EMBED_MODEL", "nomic-embed-text"))
    semantic_enabled: bool = field(
        default_factory=lambda: os.environ.get("SEMANTIC_ENABLED", "1") not in ("0", "false", "False")
    )
    semantic_threshold: float = field(default_factory=lambda: float(os.environ.get("SEMANTIC_THRESHOLD", "0.92")))
    semantic_top_k: int = field(default_factory=lambda: int(os.environ.get("SEMANTIC_TOP_K", "5")))
    # Semiose — Camada C: re-ranking contextual. Desabilita com RERANK_ENABLED=0.
    rerank_enabled: bool = field(
        default_factory=lambda: os.environ.get("RERANK_ENABLED", "1") not in ("0", "false", "False")
    )
    context_boost: float = field(default_factory=lambda: float(os.environ.get("CONTEXT_BOOST", "0.02")))
    # Semiose — Camada C Nível 2 (S3): cross-encoder reranker (opt-in, lazy via sentence-transformers).
    # Bi-encoder recupera; cross-encoder reordena top-K (Reimers & Gurevych, 2019; Gao et al., 2023).
    rerank_cross_encoder_enabled: bool = field(
        default_factory=lambda: os.environ.get("RERANK_CROSS_ENCODER_ENABLED", "0") not in ("0", "false", "False")
    )
    cross_encoder_model: str = field(
        default_factory=lambda: os.environ.get(
            "CROSS_ENCODER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
        )
    )
    # Semiose — Camada A+ (S1): Contextual Embeddings no índice do router (opt-in).
    # Prefixa cada exemplo com seu domínio antes de embedar (Anthropic, 2024).
    contextual_embeddings_enabled: bool = field(
        default_factory=lambda: os.environ.get("CONTEXTUAL_EMBEDDINGS_ENABLED", "0") not in ("0", "false", "False")
    )
    # Semiose — S2: retrieval híbrido no router — denso (Qdrant) + BM25 in-process
    # com fusão RRF (opt-in). Espelha Contextual Embeddings + Contextual BM25
    # (Anthropic, 2024). Gates de aceite continuam sobre o cosseno.
    hybrid_retrieval_enabled: bool = field(
        default_factory=lambda: os.environ.get("HYBRID_RETRIEVAL_ENABLED", "0") not in ("0", "false", "False")
    )
    rrf_k: int = field(default_factory=lambda: int(os.environ.get("RRF_K", "60")))
    # Semiose — S5: multi-query expansion no router (opt-in, modo Model).
    # No MISS do consenso, expande a pergunta em N variantes via LLM e tenta
    # cada uma pelos mesmos gates. Custo ~1 chamada LLM por miss.
    multi_query_enabled: bool = field(
        default_factory=lambda: os.environ.get("MULTI_QUERY_ENABLED", "0") not in ("0", "false", "False")
    )
    multi_query_n: int = field(default_factory=lambda: int(os.environ.get("MULTI_QUERY_N", "2")))
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
    # LangSmith — tracing cloud (ecossistema LangChain oficial).
    # Desabilita com LANGSMITH_ENABLED=0. Chave gratuita em smith.langchain.com.
    langsmith_enabled: bool = field(
        default_factory=lambda: os.environ.get("LANGSMITH_ENABLED", "0") not in ("0", "false", "False")
    )
    langsmith_api_key: str = field(default_factory=lambda: os.environ.get("LANGSMITH_API_KEY", ""))
    langsmith_project: str = field(default_factory=lambda: os.environ.get("LANGSMITH_PROJECT", "ai-orchestrator"))
    langsmith_otel_enabled: bool = field(
        default_factory=lambda: os.environ.get("LANGSMITH_OTEL_ENABLED", "0") not in ("0", "false", "False")
    )
    # OpenTelemetry — camada padrão de instrumentação (CNCF).
    # Fan-out para todos os backends via OTel Collector.
    otel_enabled: bool = field(
        default_factory=lambda: os.environ.get("OTEL_ENABLED", "0") not in ("0", "false", "False")
    )
    otel_service_name: str = field(default_factory=lambda: os.environ.get("OTEL_SERVICE_NAME", "ai-orchestrator"))
    otel_exporter_endpoint: str = field(
        default_factory=lambda: os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://otel-collector:4318")
    )
    # Phoenix — evaluation de LLM (Arize, Apache 2.0).
    # Desabilita com PHOENIX_ENABLED=0. Roda local na porta 6006.
    phoenix_enabled: bool = field(
        default_factory=lambda: os.environ.get("PHOENIX_ENABLED", "0") not in ("0", "false", "False")
    )
    phoenix_host: str = field(default_factory=lambda: os.environ.get("PHOENIX_HOST", "http://phoenix:6006"))
    phoenix_project: str = field(default_factory=lambda: os.environ.get("PHOENIX_PROJECT", "ai-orchestrator"))
    # Estado conversacional (multi-turn)
    thread_db_path: str = field(default_factory=lambda: os.environ.get("THREAD_DB_PATH", "/tmp/threads.db"))
    # Injection Detector (BERTimbau fine-tunado). Desabilita com INJECTION_DETECTOR_ENABLED=0.
    injection_model: str = field(
        default_factory=lambda: os.environ.get("INJECTION_MODEL", "/app/models/injection_classifier")
    )
    injection_threshold: float = field(
        default_factory=lambda: float(os.environ.get("INJECTION_THRESHOLD", "0.7"))
    )
    injection_detector_enabled: bool = field(
        default_factory=lambda: os.environ.get("INJECTION_DETECTOR_ENABLED", "1") not in ("0", "false", "False")
    )
    # OOD guard (log-only): resíduo de subespaço da query vs golden de routing
    # (SVD, numpy, CPU). 3º sinal de segurança — não bloqueia, só loga/tracia.
    ood_guard_enabled: bool = field(
        default_factory=lambda: os.environ.get("OOD_GUARD_ENABLED", "1") not in ("0", "false", "False")
    )
    # Calibrado em 2026-07-02 (P95 do LOO in-dist, AUC 0.980) — evals/eval_ood_guard.py.
    ood_threshold: float = field(default_factory=lambda: float(os.environ.get("OOD_THRESHOLD", "0.48")))
    # HITL: confirmação humana antes do dispatch de operações de ESCRITA
    # (write-intent determinístico em gateway/write_intent.py). Leitura nunca
    # pausa. Opt-in: HITL_ENABLED=1 religa o evento SSE `confirm` no /chat.
    hitl_enabled: bool = field(
        default_factory=lambda: os.environ.get("HITL_ENABLED", "0") not in ("0", "false", "False")
    )
    # Semiose — Camada A: enriquecimento contextual da query. Desabilita com ENRICHER_ENABLED=0.
    enricher_enabled: bool = field(
        default_factory=lambda: os.environ.get("ENRICHER_ENABLED", "1") not in ("0", "false", "False")
    )
    # spaCy NER fallback (lazy-load pt_core_news_sm). Desabilita com SPACY_ENABLED=0.
    spacy_enabled: bool = field(
        default_factory=lambda: os.environ.get("SPACY_ENABLED", "1") not in ("0", "false", "False")
    )
    # Semiose — Camada B: Knowledge Graph (Neo4j). Desabilita com NEO4J_ENABLED=0.
    neo4j_enabled: bool = field(
        default_factory=lambda: os.environ.get("NEO4J_ENABLED", "0") not in ("0", "false", "False")
    )
    neo4j_uri: str = field(default_factory=lambda: os.environ.get("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.environ.get("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.environ.get("NEO4J_PASSWORD", "changeme"))
    # Semiose — Camada A+B: realimenta o enricher com vizinhos 1-hop do KG na
    # query enviada ao classificador (opt-in). Requer NEO4J_ENABLED. Default off
    # até validação em eval (pode causar over-routing em queries single-domain).
    kg_enrich_enabled: bool = field(
        default_factory=lambda: os.environ.get("KG_ENRICH_ENABLED", "0") not in ("0", "false", "False")
    )
    # Pool dedicado para execução do grafo (evita competir com asyncio default pool).
    max_graph_workers: int = field(default_factory=lambda: int(os.environ.get("MAX_GRAPH_WORKERS", "4")))
    # Timeout global de request SSE (segundos). Independente do LLM_TIMEOUT_S.
    request_timeout_s: float = field(default_factory=lambda: float(os.environ.get("REQUEST_TIMEOUT_S", "600")))


def load_settings() -> Settings:
    """Carrega as configurações do ambiente (chamar uma vez no boot)."""
    return Settings()
