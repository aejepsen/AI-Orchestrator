# ROADMAP — AI-Orchestrator (2026-07-02)

**Estado atual:** Fases 0–8 concluídas (PoC + LoRA 9B + Fase 6 obs/HITL + Fase 7 BERT/security + Fase 8 Semiose A/B/C). Produção live em https://suasalada.com.br (`MODEL=qwen3.5-9b-orch`). Auditoria 2026-06-14: 0 CRITICO/ALTO/MEDIO. Git main sincronizado; 1 arquivo untracked (notas de safeguards, ver §5).

> **Regra de produção:** app deve permanecer ONLINE (links com entrevistadores). Deploy/restart só com aviso prévio ao usuário.

---

## Implementações programadas (localizadas)

### 1. Semiose — Trabalho Futuro (`PLANO_SEMIOSE.md` §Trabalho Futuro, tabela de sugestões priorizadas)

| # | Item | Arquivos-alvo | Esforço | Status |
|---|------|--------------|---------|--------|
| S2 | Retrieval híbrido (denso + BM25) com fusão RRF — combo Anthropic "Contextual Embeddings + Contextual BM25" (−67% falhas c/ rerank) | `gateway/semantic_router.py` | Médio | ⏳ Pendente |
| S5 | Multi-query expansion opt-in (modo LLM, flag `MULTI_QUERY_ENABLED`) | `gateway/query_enricher.py` | Médio (custo LLM) | ⏳ Pendente |
| S4 | GraphRAG global: comunidades Leiden + resumos pré-gerados, tool `summarize_community` | `scripts/seed_neo4j.py` + nova tool | Alto (experimental) | ⏳ Pendente |
| — | Re-ranking Nível 2 via LLM (`RERANK_LLM_ENABLED`) | `gateway/semantic_router.py` | — | Marcado como futuro |
| — | Golden de relações (derivado do seed) p/ Relation Validity real (hoje é proxy non-garbage) | `evals/eval_semiose.py` + seed | Baixo | Trabalho futuro |

Ordem do próprio plano: **S2 primeiro** (potencializa S1 já implementado); S5/S4 só se métricas justificarem.

### 2. Backlog da PoC (`README.md` §Backlog, resíduos das Fases 2/5)

- ~~**Reset de estado dos serviços entre runs de eval**~~ ✅ 2026-07-02 — `POST /admin/reset` (common.py + 4 mains) + reset automático no eval_domains.
- ~~**Registry anexar response schema à description da tool**~~ ✅ 2026-07-02 — `_response_summary()` em registry.py.
- **Streaming token-a-token na síntese** — hoje SSE emite `route`/`agent`/`final` por etapa. Alvo: `gateway/llm.py` + nó synthesize + frontend.
- **RAG sobre documentos não estruturados** — extensão maior; abriria 5º domínio.

### 3. HITL — write-intent detection (Fase 6, residual)

HITL implementado (`interrupt()` no nó `confirm_dispatch`, `POST /chat/{thread_id}/resume`, `ConfirmCard.tsx`) mas **desabilitado por padrão**: sem detecção write vs read, dispara pra toda query. Pendente: classificar intent de escrita (regra/regex sobre tools chamadas ou flag no RoutePlan) → habilitar HITL só em writes.

### 4. Observabilidade — métricas faltantes (`docs/observability-plan.md`)

| Métrica | Status | Ação |
|---------|--------|------|
| Task Success Rate | ✅ plano | `task_success` no `eval_semiose.py` (`stop_reason != "answer"` → fail) |
| Tool Call Efficiency | ✅ plano | `tools_per_task` (média + P95) no `eval_semiose.py` |
| Faithfulness (RAG Triad) | 🟡 próximo passo | Phoenix `FaithfulnessEvaluator` adaptado a LLM local (judge via Ollama) |
| Correction Frequency | 🟡 | Endpoint `/feedback` → LangSmith AnnotationQueue (documentável) |
| Human Takeover Rate | 🔴 doc-only | `clarification_rate` derivada do RoutePlan |
| RAOI | 🟡 doc-only | Fórmula no README/dashboard (precisa de dados reais de operação) |

### 5. Housekeeping

- Arquivo untracked na raiz: `"Para criar salvaguardas (safeguards) robustas"` (notas sobre defesas anti-injection, 5.4 KB) — mover para `docs/notes-safeguards.md` ou descartar.
- Notebooks Colab antigos no Drive a deletar (ids em memória de sessão).

---

## Sequência recomendada (ROI × esforço)

1. ~~**Quick wins de eval**~~ ✅ **CONCLUÍDO 2026-07-02** — `POST /admin/reset` nos 4 serviços (fora do OpenAPI, X-Internal-Key); response schema anexado à description das tools (`_response_summary` em registry.py); `task_success_rate` + `tools_per_task` (média/P95) no eval_domains; reset automático pré-run (`--no-reset` p/ pular). Bônus: fix de regressão no guard desconto→remove-estoque (`_AVAILABILITY_RE` em router.py) que quebrava 2 testes multi-domínio. 335 testes verdes. **Pendente: rebuild dos containers p/ ativar em prod (combinar horário).**
2. **S2 retrieval híbrido (BM25+RRF)** — maior evidência de ganho (Anthropic), médio esforço, medível com Routing Failure Rate (S6 já implementado).
3. **HITL write-intent** — transforma feature desabilitada em diferencial de demo (governança).
4. **Faithfulness eval (Phoenix + judge local)** — completa o RAG Triad, forte sinal de portfólio.
5. **Streaming token-a-token** — UX da síntese; independente dos anteriores.
6. **S5 multi-query / S4 GraphRAG / RAG docs** — experimentais; só se métricas das etapas 1–2 justificarem.
