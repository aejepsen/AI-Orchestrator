# ROADMAP — AI-Orchestrator (2026-07-04)

**Estado atual:** Fases 0–8 concluídas (PoC + LoRA 9B + Fase 6 obs/HITL + Fase 7 BERT/security + Fase 8 Semiose A/B/C) + rodada 2026-07-04 de observabilidade (OTel GenAI semconv + Collector fan-out Phoenix/Prometheus, evals com fontes live/eval/estimate, tuning Ollama −16% makespan). Produção live em https://suasalada.com.br (`MODEL=qwen3.5-9b-orch`). Auditoria 2026-06-14: 0 CRITICO/ALTO/MEDIO. Git main sincronizado; 1 arquivo untracked (notas de safeguards, ver §5).

> **Regra de produção:** app deve permanecer ONLINE (links com entrevistadores). Deploy/restart só com aviso prévio ao usuário.

---

## Implementações programadas (localizadas)

### 1. Semiose — Trabalho Futuro (`PLANO_SEMIOSE.md` §Trabalho Futuro, tabela de sugestões priorizadas)

| # | Item | Arquivos-alvo | Esforço | Status |
|---|------|--------------|---------|--------|
| S2 | Retrieval híbrido (denso + BM25) com fusão RRF — combo Anthropic "Contextual Embeddings + Contextual BM25" | `gateway/bm25.py` + `gateway/semantic_router.py` | Médio | ✅ 2026-07-02 (opt-in, default off — ver nota no PLANO_SEMIOSE) |
| S5 | Multi-query expansion opt-in (modo LLM, flag `MULTI_QUERY_ENABLED`) | `gateway/query_enricher.py` | Médio (custo LLM) | ⏳ Pendente |
| S4 | GraphRAG global: comunidades Leiden + resumos pré-gerados, tool `summarize_community` | `scripts/seed_neo4j.py` + nova tool | Alto (experimental) | ⏳ Pendente |
| — | Re-ranking Nível 2 via LLM (`RERANK_LLM_ENABLED`) | `gateway/semantic_router.py` | — | Marcado como futuro |
| — | Golden de relações (derivado do seed) p/ Relation Validity real (hoje é proxy non-garbage) | `evals/eval_semiose.py` + seed | Baixo | Trabalho futuro |

Ordem do próprio plano: **S2 primeiro** (potencializa S1 já implementado); S5/S4 só se métricas justificarem.

### 2. Backlog da PoC (`README.md` §Backlog, resíduos das Fases 2/5)

- ~~**Reset de estado dos serviços entre runs de eval**~~ ✅ 2026-07-02 — `POST /admin/reset` (common.py + 4 mains) + reset automático no eval_domains.
- ~~**Registry anexar response schema à description da tool**~~ ✅ 2026-07-02 — `_response_summary()` em registry.py.
- ~~**Streaming token-a-token na síntese**~~ ✅ 2026-07-02 — `chat_stream` + `on_token` + evento SSE `token`; frontend com balão streaming.
- ~~**RAG sobre documentos não estruturados**~~ ✅ 2026-07-02 — políticas internas via `search_documents` (Recall@3 100%); não abriu 5º domínio: política é leitura cross-domain servida como tool aos 4 agentes.

### 3. HITL — write-intent detection (Fase 6, residual)

✅ **CONCLUÍDO 2026-07-02.** `gateway/write_intent.py` + gate no `_confirm_dispatch` (leitura auto-aprova) + flag `HITL_ENABLED` (main.py religa o evento SSE `confirm`). Evolução futura: confirmar no nível da tool call (interceptar POST/PUT/DELETE no executor) em vez do pré-dispatch.

### 4. Observabilidade — métricas faltantes (`docs/observability-plan.md`)

| Métrica | Status | Ação |
|---------|--------|------|
| Task Success Rate | ✅ 2026-07-02 | `task_success` medido no eval_domains (100%); fonte `eval` no dashboard |
| Tool Call Efficiency | ✅ 2026-07-02 | `tools_per_task` 1.3 média / 2 P95; fonte `eval` no dashboard |
| Faithfulness (RAG Triad) | ✅ 2026-07-02/04 | Juiz LLM local (desvio documentado do Phoenix evaluator): 39/40 = 97.5%; dashboard lê o eval real em vez de valor fixo |
| Correction Frequency | 🟡 | Endpoint `/feedback` → LangSmith AnnotationQueue (documentável; status não aceita mais key placeholder) |
| Human Takeover Rate | ✅ 2026-07-04 | Clarification Rate agora **live** (clarifications/traces), não mais do eval de semiose; evento SSE `clarification` distinto no /chat |
| Routing Failure Rate | ✅ 2026-07-04 | Agora **live** a partir dos traces |
| RAOI | 🟡 doc-only | Fórmula no README/dashboard (precisa de dados reais de operação) |

✅ **2026-07-04 — camada OTel GenAI completa** (`gateway/otel.py` + `otel-collector-config.yaml`): spans `gen_ai.*` + histogramas token.usage/operation.duration/TTFT via OTLP; Collector fan-out traces→Phoenix, métricas→Prometheus :8889 (fonte independente do Langfuse no `/metrics`). Causa-raiz do `tokens=0` corrigida: usage lido na fonte (Ollama `prompt_eval_count`/`eval_count`) + trace propagado em classify/agentes. Bônus perf: `OLLAMA_NUM_PARALLEL=3` + `OLLAMA_FLASH_ATTENTION=1` → fan-out 3 domínios 23.3s→19.5s (−16%), VRAM 6.3/12 GB.

### 4b. Derivados do parecer CliffordNet (`plano_cliffordNet_parecer.md`)

- ~~**Fase 1 — OOD guard (resíduo de subespaço)**~~ ✅ 2026-07-02 — `gateway/subspace_guard.py`, log-only no sanitize; AUC **0.9803** PASS (protocolo LOO), 27/30 OOD @ threshold 0.48; adversarial in-domain confirmado baixo (papel do BERTimbau). Gotchas: fit filtra casos de clarification (fora de domínio por design contaminavam o subespaço) e calibração LOO obrigatória (split 80/20 superestimava o threshold).
- ~~**Fase 2 — KG embeddings rotacionais (RotatE via PyKEEN)**~~ ✅ 2026-07-02 — `scripts/kg_link_prediction.py`; RotatE MRR 0.226/Hits@10 0.431 (gate 0.5 **FAIL declarado** — teste de ~30 triplas, variância ±0.16); gate comparativo PASS: **RotatE = 3× TransE em MRR** (rotação geométrica paga). CSV de candidatas type-safe p/ revisão humana em `evals/results/kg_suggestions_*.csv`; zero escrita no Neo4j.

### 5. Housekeeping

- Arquivo untracked na raiz: `"Para criar salvaguardas (safeguards) robustas"` (notas sobre defesas anti-injection, 5.4 KB) — mover para `docs/notes-safeguards.md` ou descartar.
- Notebooks Colab antigos no Drive a deletar (ids em memória de sessão).

---

## Sequência recomendada (ROI × esforço)

1. ~~**Quick wins de eval**~~ ✅ **CONCLUÍDO 2026-07-02** — `POST /admin/reset` nos 4 serviços (fora do OpenAPI, X-Internal-Key); response schema anexado à description das tools (`_response_summary` em registry.py); `task_success_rate` + `tools_per_task` (média/P95) no eval_domains; reset automático pré-run (`--no-reset` p/ pular). Bônus: fix de regressão no guard desconto→remove-estoque (`_AVAILABILITY_RE` em router.py) que quebrava 2 testes multi-domínio. 335 testes verdes. **Pendente: rebuild dos containers p/ ativar em prod (combinar horário).**
2. ~~**S2 retrieval híbrido (BM25+RRF)**~~ ✅ **CONCLUÍDO 2026-07-02** — `gateway/bm25.py` (Okapi stdlib) + `_rrf_fuse` no SemanticRouter (reordena pool denso 2×top_k; cosseno preservado nos gates); flag `HYBRID_RETRIEVAL_ENABLED` default off. Medição: no threshold 0.92 (prod) camada semântica não dispara em leave-one-out (efeito nulo); na banda 0.80 o híbrido corta falsos aceites (10→8 disparos) e recupera +1.3 pp (69.3%→70.6%), mas banda <0.92 segue pior que fallback LLM → default off; destrava com embedder melhor. Nota completa em PLANO_SEMIOSE.md. **Achado colateral: routing no golden expandido (153 casos) está em 72.5% (gate 90%) — dominado por sub-roteamento multi-domínio; README anuncia 90.5% do golden antigo (63).**
3. ~~**HITL write-intent**~~ ✅ **CONCLUÍDO 2026-07-02** — `gateway/write_intent.py` (léxico determinístico PT das write ops; frases nominais "contas a pagar" excluídas); `_confirm_dispatch` só pausa em escrita; `HITL_ENABLED=1` religa o evento SSE `confirm` no /chat (main.py). 24 testes novos. Ativação em prod requer rebuild + `HITL_ENABLED=1` no .env.
4. ~~**Faithfulness eval (judge local)**~~ ✅ **CONCLUÍDO 2026-07-02** — `evals/eval_faithfulness.py`: juiz LLM local (OllamaClient, format=json) avalia resposta vs observações (body agora capturado na tool_trace, truncado em 12k — 2k gerava falso INFIEL). **Medido: 39/40 = 97.5% PASS (gate 90%)**; único INFIEL é ruído do juiz (resposta correta, juiz confundiu contagem×quantidade). Desvio documentado: juiz direto em vez do Phoenix FaithfulnessEvaluator (espera OpenAI). Bônus da rodada: eval_domains 36/40 = 90% (4/4 gates), task_success 100%, tools/task 1.3 média / 2 P95.
5. ~~**Streaming token-a-token**~~ ✅ **CONCLUÍDO 2026-07-02** — `OllamaClient.chat_stream` (NDJSON), callback `on_token` thread-local no `_synthesize` (só multi-domínio; single-domain vem pronto do agente), evento SSE `token` + `final` mantido como fonte de verdade; frontend acumula em balão "streaming" com cursor e substitui no `final`. Bônus: fix de callbacks thread-local stale no `/resume` (threads do pool reutilizadas vazavam callbacks do request anterior). Verificação live pendente de rebuild.
6. **Experimentais:**
   - ~~**S5 multi-query expansion**~~ ✅ implementado e medido 2026-07-02 — **default OFF por medição**: no threshold de prod (0.92) acurácia idêntica (94.1%), camada semântica segue 0 disparos em leave-one-out, latência média 2× (1.09→2.32s: cada miss paga 1 chamada LLM de expansão antes do fallback, que já custava 1 chamada). Flag `MULTI_QUERY_ENABLED` fica disponível p/ quando o embedder for trocado.
   - ~~**S4 GraphRAG mínimo**~~ ✅ 2026-07-02 — Louvain offline (`scripts/build_kg_communities.py`, 11 comunidades, modularity 0.618) + resumos LLM grounded por comunidade + tool virtual `summarize_community` (opt-in `GRAPHRAG_ENABLED=1`; serve artefato pré-gerado, zero Neo4j/LLM no caminho da request). Artefato em `./models/kg_communities.json` (volume, fora do git).
   - ~~**RAG docs (políticas internas)**~~ ✅ 2026-07-02 — corpus `docs/policies/*.md` (5 políticas derivadas das regras REAIS dos serviços — não podem contradizer a API), chunking por seção + SBERT + Qdrant collection `documents` (`scripts/ingest_documents.py`, idempotente), tool virtual `search_documents` para os 4 domínios (opt-in `RAG_DOCS_ENABLED=1`). **Medido: Recall@3 = 12/12 = 100% PASS** (`evals/eval_docs.py`, gate 80%).
7. ~~**Routing no golden 153**~~ ✅ **CONCLUÍDO 2026-07-02** — 72.5% → **91.5% PASS** (gate 90%) via guards determinísticos (bug fornecedor re-add, conceito financeiro→+financas, quem-aprova→fin+rh, comprou→estoque+vendas, vendedor-região→vendas) + regras de decomposição no prompt (CUSTO É FINANÇAS c/ exceção produto, PERFIL DO FUNCIONÁRIO É RH, preferir rotear a clarificar). Injection reconfirmado 0/6. README atualizado. Resíduo (~8.5%): ruído de label do golden denso — **auditoria executada 2026-07-02**: critérios explícitos em `docs/golden_routing_criteria.md` (C1 comissão, C2 aprovação, C3 orçamento×departamento, C4 vendedor×região, C5 compra), 8 labels normalizados, 2 guards afinados → **94.1% PASS**, injection 0/6.
