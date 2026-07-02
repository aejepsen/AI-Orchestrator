# ROADMAP — AI-Orchestrator (2026-07-02)

**Estado atual:** Fases 0–8 concluídas (PoC + LoRA 9B + Fase 6 obs/HITL + Fase 7 BERT/security + Fase 8 Semiose A/B/C). Produção live em https://suasalada.com.br (`MODEL=qwen3.5-9b-orch`). Auditoria 2026-06-14: 0 CRITICO/ALTO/MEDIO. Git main sincronizado; 1 arquivo untracked (notas de safeguards, ver §5).

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
- **Streaming token-a-token na síntese** — hoje SSE emite `route`/`agent`/`final` por etapa. Alvo: `gateway/llm.py` + nó synthesize + frontend.
- **RAG sobre documentos não estruturados** — extensão maior; abriria 5º domínio.

### 3. HITL — write-intent detection (Fase 6, residual)

✅ **CONCLUÍDO 2026-07-02.** `gateway/write_intent.py` + gate no `_confirm_dispatch` (leitura auto-aprova) + flag `HITL_ENABLED` (main.py religa o evento SSE `confirm`). Evolução futura: confirmar no nível da tool call (interceptar POST/PUT/DELETE no executor) em vez do pré-dispatch.

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
2. ~~**S2 retrieval híbrido (BM25+RRF)**~~ ✅ **CONCLUÍDO 2026-07-02** — `gateway/bm25.py` (Okapi stdlib) + `_rrf_fuse` no SemanticRouter (reordena pool denso 2×top_k; cosseno preservado nos gates); flag `HYBRID_RETRIEVAL_ENABLED` default off. Medição: no threshold 0.92 (prod) camada semântica não dispara em leave-one-out (efeito nulo); na banda 0.80 o híbrido corta falsos aceites (10→8 disparos) e recupera +1.3 pp (69.3%→70.6%), mas banda <0.92 segue pior que fallback LLM → default off; destrava com embedder melhor. Nota completa em PLANO_SEMIOSE.md. **Achado colateral: routing no golden expandido (153 casos) está em 72.5% (gate 90%) — dominado por sub-roteamento multi-domínio; README anuncia 90.5% do golden antigo (63).**
3. ~~**HITL write-intent**~~ ✅ **CONCLUÍDO 2026-07-02** — `gateway/write_intent.py` (léxico determinístico PT das write ops; frases nominais "contas a pagar" excluídas); `_confirm_dispatch` só pausa em escrita; `HITL_ENABLED=1` religa o evento SSE `confirm` no /chat (main.py). 24 testes novos. Ativação em prod requer rebuild + `HITL_ENABLED=1` no .env.
4. ~~**Faithfulness eval (judge local)**~~ ✅ **CONCLUÍDO 2026-07-02** — `evals/eval_faithfulness.py`: juiz LLM local (OllamaClient, format=json) avalia resposta vs observações (body agora capturado na tool_trace, truncado em 12k — 2k gerava falso INFIEL). **Medido: 39/40 = 97.5% PASS (gate 90%)**; único INFIEL é ruído do juiz (resposta correta, juiz confundiu contagem×quantidade). Desvio documentado: juiz direto em vez do Phoenix FaithfulnessEvaluator (espera OpenAI). Bônus da rodada: eval_domains 36/40 = 90% (4/4 gates), task_success 100%, tools/task 1.3 média / 2 P95.
5. **Streaming token-a-token** — UX da síntese; independente dos anteriores.
6. **S5 multi-query / S4 GraphRAG / RAG docs** — experimentais; só se métricas das etapas 1–2 justificarem.
