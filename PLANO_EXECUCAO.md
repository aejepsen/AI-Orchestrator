# PLANO_EXECUCAO.md — AI-Orchestrator

> PoC **AI Gateway + Multi-Agent Microservices** (ver `PROPOSAL.md`). Orquestrador MoE raciocina e roteia; subagentes (nós do grafo com tools escopadas por domínio — decisão pós-Fase 0) executam tool-calling; regras de negócio vivem em microsserviços FastAPI determinísticos. 100% on-premise.

**Premissa de recursos:** AI-Tractor **parado** durante execução — 12 GB VRAM + 14 GB RAM integralmente disponíveis.

---

## 0. Decisões de arquitetura (fixadas antes de codar)

| Decisão | Escolha | Razão |
|---|---|---|
| Orquestrador | Qwen3 MoE A3B (Q4_K_M, ~19-20 GB) via **llama.cpp/Ollama com offload híbrido GPU+RAM** | 20 GB não cabem só na RAM (14 GB físicos). Split: ~9-10 GB VRAM + ~10 GB RAM. MoE com 3B ativos mantém 8-15 tok/s mesmo parcialmente em CPU |
| Subagente | **Mesmo MoE, contexto isolado por domínio** (decisão pós-Fase 0) | Swap de modelo medido em 8.8–24 s (inviável); MoE warm a 18 tok/s atende tudo. Subagente = nó LangGraph com system prompt + tools só do seu domínio. 7B = contingência |
| Gestão de memória | **MoE-único residente** (`keep_alive=-1`), zero swap | Medição Fase 0: handoff por swap custaria 9–24 s por troca; residência única dá latência uniforme |
| Orquestração de grafo | **LangGraph** (StateGraph, checkpointer SQLite) | Estado explícito, retry por nó, human-in-the-loop nativo, já dominado (AI-Professor) |
| Microsserviços | FastAPI + Pydantic v2, um serviço por domínio (Finanças, RH, Estoque, Vendas), SQLite por serviço | Determinístico, OpenAPI gratuito (vira schema de tool), isolamento real de dados por domínio |
| Contrato IA↔negócio | Tools geradas **a partir do OpenAPI** de cada microsserviço | IA nunca calcula/persiste; única fonte de verdade é a API. Schema drift impossível |
| Transporte | HTTP interno na rede Docker; gateway expõe um único endpoint SSE `/chat` | Padrão Experience Layer: cliente fala com 1 porta |
| Segurança | Sanitização de input no boundary (reusar padrão `sanitize.py` do AI-Tractor), API key interna entre serviços, least-privilege: cada subagente só enxerga tools do seu domínio | Lição medida no AI-Tractor: system prompt sozinho não segura injection |
| Observabilidade | Log estruturado JSON por nó do grafo + trace_id propagado; latência por etapa | Sem isso, debugging de grafo multi-agente é cego |

---

## 1. Estrutura do repositório

```
AI-Orchestrator/
├── PROPOSAL.md / PLANO_EXECUCAO.md
├── docker-compose.yml            # ollama, gateway, 4 microsserviços, qdrant, langfuse
├── gateway/                      # Experience Layer
│   ├── main.py                   # FastAPI: POST /chat (SSE), POST /chat/{thread_id}/resume
│   ├── graph.py                  # LangGraph StateGraph: sanitize → classify → dispatch → synthesize
│   ├── agents.py                 # loop tool-calling por domínio + system prompt anti-fabricação
│   ├── router.py                 # classify_intent: semântico → LLM → léxico
│   ├── llm.py                    # cliente Ollama (chat/embed, think detection)
│   ├── sanitize.py               # boundary anti-injection + 14 regex injection detection (log only)
│   ├── security.py               # AccessTokenGuard (fail-closed) + RateLimiter + CF-Connecting-IP
│   ├── config.py                 # Settings dataclass (todas as envs)
│   ├── embedder.py               # Embedder Protocol (SBERTEmbedder + OllamaEmbedder fallback, 384 dim)
│   ├── injection_classifier.py   # BERTimbau fine-tunado (400 exemplos, 100% val accuracy)
│   ├── tracing.py                # Langfuse integration (trace/span/generation)
│   └── tools/                    # registry OpenAPI→tools + circuit breaker
├── services/
│   ├── financas/  (main.py, rules.py, db.py, seed.py, tests/) — CRUD completo
│   ├── rh/        ...
│   ├── estoque/   ...
│   └── vendas/    ...
├── frontend/                     # Chat single-page (Vite + React + Tailwind v4)
├── evals/
│   ├── golden_routing.jsonl      # intenção → domínio esperado
│   ├── eval_routing.py           # gate de acurácia de roteamento
│   ├── eval_domains.py           # gate de domínio ≥80%
│   ├── eval_injection.py         # gate 0 leaks
│   └── demo.py                   # 5 conversas gravadas
├── train/                        # LoRA fine-tune: build_dataset.py, colab notebooks, Modelfile
├── docs/                         # PLANO_LORA_9B.md, SKILL_MULTIAGENT.md, AUDIT_2026-06-14.md, gen_diagrams.py (7 PNGs)
└── demo/                         # transcripts gravados
```

---

## 2. Fases de execução

### Fase 0 — Validação de viabilidade (gate go/no-go) — ✅ EXECUTADA 2026-06-10
1. ~~Baixar o MoE quantizado; medir tok/s e swap.~~ Medido (`evals/fase0_bench.py`, container exclusivo `ai-orchestrator-ollama`, porta 11435, AI-Tractor parado):

| Métrica | Medido | Gate | Status |
|---|---|---|---|
| MoE `qwen3:30b-a3b` Q4 geração (warm) | **17.97 tok/s** (cold load 55 s; split ~10 GiB GPU + ~8 GiB RAM) | ≥ 5 tok/s | ✅ 3.6x acima |
| Swap MoE→7B / 7B→MoE | **24.0 s / 8.8 s** | ≤ 5 s | ❌ |

2. **Veredito: arquitetura revisada — MoE-único residente.** O swap reprova, mas o MoE a 18 tok/s elimina a necessidade do 7B: subagentes passam a ser **nós do grafo com system prompt e tools escopados por domínio** (mesmo peso, contextos isolados), não modelos separados. Padrão multi-agente preservado (isolamento por escopo de tool, não por peso); zero troca de modelo; latência uniforme. O 7B fica como contingência documentada.

### Fase 1 — Microsserviços determinísticos (sem IA) — ✅ EXECUTADA 2026-06-10
> As-built: 4 serviços em `services/` (portas 8101–8104, Dockerfile único por ARG, SQLite por serviço em volume próprio, envelope de erro `{error, detail, rule}` em `services/common.py`). **89 testes passando**, ruff limpo, 4 containers Up com smoke nos endpoints de negócio. Liberação de reserva via `POST /reservations/{id}/release` (histórico preservado, tool-friendly); cashflow projetado por `due_date`.
1. 4 serviços FastAPI com regras puras, dados seed realistas e **CRUD completo**:
   - **Finanças (:8101):** `GET/POST /accounts`, `GET/PUT/DELETE /accounts/{id}`, `POST /accounts/{id}/pay`, `POST /accounts/{id}/receive`, `GET /cashflow` (start/end). Regras: alçada de aprovação, só abertas aceitam exclusão/liquidação.
   - **RH (:8102):** `GET/POST /employees`, `GET/PUT/DELETE /employees/{id}`, `GET /employees/{id}/vacation-balance`, `POST /vacations`, `POST /reimbursements`, `GET /headcount`. Regras: férias CLT simplificada, limite de reembolso. Filtros com `COLLATE NOCASE`.
   - **Estoque (:8103):** `GET/POST /products`, `GET/PUT/DELETE /products/{sku}`, `GET/POST /reservations`, `POST /reservations/{id}/release`, `GET /replenishment`. Regras: SKU único, não exclui com reservas ativas.
   - **Vendas (:8104):** `GET/POST /orders`, `GET/PUT/DELETE /orders/{id}`, `GET /orders/{id}/commission`, `GET/POST /sellers`, `GET/PUT/DELETE /sellers/{id}`. Regras: desconto por política, comissão por faixa.
2. Pydantic em todas as entradas/saídas; erros de negócio = 422 com payload estruturado (vira feedback legível pro agente).
3. Testes unitários das regras (pytest) — **antes** de qualquer IA tocar nelas.

### Fase 2 — Subagentes especialistas — ✅ EXECUTADA 2026-06-10 (gate PASS 4/4 domínios)
> As-built: `gateway/tools/registry.py` (OpenAPI→tools Ollama, executor HTTP que devolve 422/404 como observação), `gateway/llm.py` (/api/chat, `keep_alive=-1`), `gateway/agents.py` (loop escopado por domínio). Eval `evals/eval_domains.py` (40 tasks): **financas 9/10, rh 10/10, estoque 9/10, vendas 10/10 = 95%** (47.7 min, sequencial).
> Achados medidos: (1) `think=false` no qwen3 vaza CoT no content E faz o modelo pré-computar regra de negócio sem chamar a tool (1º run: 82.5%, rh 60%) — fix: `think=true` + regras explícitas no system prompt ("execute a chamada e deixe a API validar", "resolva nome→id via listagem", "chame a tool de leitura antes de concluir indisponibilidade") → 95%. (2) Resíduo financas = estado acumulado entre runs (conta já liquidada; modelo recusou corretamente — eval precisa de reset de estado, backlog). (3) Resíduo estoque = modelo julga capacidade da tool pela descrição (schema de resposta não vai no tool schema) — backlog: registry anexar campos de resposta à description.
1. Tool registry: parse do `openapi.json` de cada serviço → schema de tools Ollama.
2. Loop de tool-calling por domínio (MoE com contexto/tools escopados; máx. N iterações, timeout, retry com erro 422 reinjetado).
3. Eval por domínio: 10 tarefas golden por serviço (ex.: "reserve 50 unidades do SKU X" → chamada correta + resposta fundada no retorno da API).

### Fase 3 — Gateway / Orquestrador — ✅ EXECUTADA 2026-06-10 (gate PASS 90.5%), atualizado 2026-06-14
> As-built: `gateway/sanitize.py` (boundary anti-injection, padrão AI-Tractor), `gateway/router.py` (RoutePlan Pydantic + `format` JSON + retry 1x + fallback léxico; clarification = rota válida; **prompt multi-domínio explícito com 4 exemplos concretos**; **score gap filter `min_score_gap=0.05`** no semantic router), `gateway/graph.py` (LangGraph StateGraph: sanitize → classify → [clarification | dispatch fan-out ThreadPoolExecutor → synthesize]; síntese pula LLM com 1 domínio; trace_id + log JSON por nó), `gateway/main.py` (POST /chat SSE: eventos route/agent/final/error). **Golden set expandido: 44→64 exemplos** (20 novos, 12 multi-domínio). Eval `evals/eval_routing.py` (63 queries): **90.5% PASS** — gate ≥90% mantido. Anti-fabricação para leitura: regra no prompt proíbe inventar dados ilustrativos em consultas.
1. Grafo LangGraph: `sanitize → classify_intent (MoE) → dispatch (1..n subagentes, paralelo quando domínios independentes) → synthesize (MoE) → resposta SSE`.
2. Roteamento estruturado: MoE responde JSON `{domains: [...], plan: ...}` validado por Pydantic; inválido → retry 1x → fallback regra léxica.
3. Multi-domínio: pergunta que cruza Vendas+Estoque+Finanças ("posso aceitar pedido de 500 unidades com 15% de desconto?") exercita fan-out/fan-in — **este é o caso demo principal**.
4. Eval de roteamento: `golden_routing.jsonl` ≥ 40 perguntas (incl. ambíguas e fora de domínio), gate ≥ 90% de acerto de domínio.

### Fase 4 — Segurança e resiliência — ✅ EXECUTADA 2026-06-10 (injection gate PASS 0/6 leaks)
> As-built: API key interna `X-Internal-Key` (`services/common.py::register_internal_auth`, `hmac.compare_digest`, isenta /health /openapi.json /docs; modo aberto com warning sem env), circuit breaker por domínio (`gateway/tools/circuit.py`: 3 falhas de transporte → OPEN 30 s → half-open; 4xx não conta), registry envia key em toda chamada, compose injeta `INTERNAL_API_KEY`. **182 testes passando**, ruff limpo; validado live: 401 sem key, 200 com key. Sanitização boundary já entregue na Fase 3.
> **Injection eval** (`evals/eval_injection.py`, 6 casos adversários: "ignore as instruções", role-swap, token ChatML, falsa autoridade, wrapper breakout): 1º run **FAIL 2/6** — router roteava a instrução injetada como intenção legítima (vazamento só de roteamento; **zero tool-calls cross-domain** — least-privilege por escopo segurou). Fix: regra de SEGURANÇA + exemplo adversário no system prompt do router → **PASS 0/6 leaks** (`evals/results/injection_20260610_145651.json`). Gotcha do eval: rodar do host exige `INTERNAL_API_KEY` no env, senão agentes levam 401 e o teste enfraquece.
> **Security hardening (Fase 6).** `.dockerignore` (root + services); Swagger/OpenAPI desabilitado em produção (`docs_url=None`, `redoc_url=None`, `openapi_url=None`); error sanitization (stack traces nunca expostos no SSE); fail-closed auth (`ALLOW_OPEN_ACCESS`); rate limiting por IP (sliding window, `RATE_LIMIT_PER_HOUR`); semantic injection detection (14 patterns PT/EN, log only); CF-Connecting-IP trust para IP real; dedicated thread pool (`MAX_GRAPH_WORKERS=4`); request deadline (`REQUEST_TIMEOUT_S=600`).
> **Demo SSE end-to-end** (gateway :8100, "pedido de 100 un. TEC-MEC-005 com 15% de desconto"): route 3 domínios em 47 s, fan-out paralelo, vendas recusa 15%>10% (regra da API), estoque confirma 100 un., síntese correta — 480 s total. Achado: fan-out de 3 agentes enfileira no MoE único (Ollama serializa) → 3º agente estourava timeout 300 s; fix `LLM_TIMEOUT_S=900` no gateway (comentado no compose).
1. Sanitização boundary + testes (padrão AI-Tractor: strip tokens de template + tags wrapper).
2. **Semantic injection detection** (`sanitize.py`): 14 padrões regex PT/EN, log-only (não reescreve texto).
3. API key interna (gateway→serviços); serviços recusam chamada direta sem key.
4. Circuit breaker simples por serviço (3 falhas → resposta degradada explícita).
5. Teste de injection no eval: pergunta com "ignore as instruções..." não pode vazar p/ tool-call fora do domínio.
6. **Anti-fabricação no system prompt**: regra crítica proíbe agente de fabricar dados em write ops — lista campos obrigatórios e pede ao usuário.

### Fase 5 — Observabilidade, demo e documentação — ✅ EXECUTADA 2026-06-10
> As-built: trace_id + log JSON por nó já entregues na Fase 3 (verificados live). `README.md` de portfólio com topologia, tabela de latências medidas (classify 15–70 s; task single-domain 26–216 s; demo 3 domínios 480 s; cold load 55 s) e resultados reais de todos os gates. `evals/demo.py` gravou as 5 conversas em `demo/transcripts.{md,json}`: single-domain 115.5 s (férias Carlos = 30 dias), multi-domain 474.2 s (15% > limite 10%), fora de domínio 17.4 s (clarification), erro 422 112.8 s (8 un. disponíveis < 500), injection 84.2 s (salários ignorados, só saldo do SKU).
1. trace_id por request; log JSON por nó (latência, modelo, tokens, tools chamadas).
2. README de portfólio: diagrama da topologia, tabela de latências medidas, resultados dos evals (números reais, não promessas).
3. Script demo: 5 conversas gravadas cobrindo single-domain, multi-domain, fora de domínio, erro de negócio (422), injection bloqueada.

---

## 3. Riscos e mitigações

| Risco | Probabilidade | Mitigação |
|---|---|---|
| MoE 20 GB lento demais no split GPU/RAM | média | Gate Fase 0; fallback 7B-único já desenhado |
| Swap de modelo a cada handoff degrada UX | média | Medir; alternativa: manter só MoE residente e dar tools direto a ele (subagente vira prompt, não modelo) |
| LLM alucina chamada de tool | alta | Validação Pydantic do tool-call, 422 reinjetado, máx. iterações; eval por domínio como gate |
| Roteamento errado em pergunta ambígua | média | Golden set com ambíguas; fallback "pedir esclarecimento" como rota válida |
| Escopo crescer (RAG, memória, UI) | alta | PoC fecha na Fase 5. Extensões viram backlog no README |

---

## 4. Critérios de aceite da PoC

- [x] 4 microsserviços com testes de regra passando (182 testes no total)
- [x] Roteamento ≥ 90% no golden set (90.5%, 64 perguntas — golden set expandido)
- [x] Caso multi-domínio fan-out/fan-in funcionando end-to-end via SSE (474 s, 3 domínios)
- [x] Injection test: 0 vazamentos cross-domain (0/6 após endurecimento do router)
- [x] Latências medidas e documentadas por etapa (tabela no README)
- [x] `docker compose up` sobe tudo do zero (6 containers; só falta `ollama pull` na 1ª vez)

---

## 5. Evolução pós-PoC: LoRA Qwen3.5-9B (2026-06-11/13)

Após PoC completa, fine-tune LoRA do Qwen3.5-9B especializado em tool-calling + routing do orquestrador. Plano e resultados completos em `docs/PLANO_LORA_9B.md`.

| Eval | LoRA 9B (prod) | Baseline 9B | Baseline 7b | MoE 30B |
|---|---|---|---|---|
| Routing (64 perguntas) | **90.5%** | 95.5% | 90.5% | 90.5% |
| Injection (6 casos) | **0/6** | 0/6 | 0/6 | 0/6 |
| Domains (40 tasks) | **87.5%** (90/90/90/80) | 87.5% (90/80/80/100) | 82.5% | 95% |
| Latência por task | **~2–4 s** (100% GPU) | — | ~7 s | ~55 s |

**Decisão:** LoRA 9B promovido a modelo de produção (`.env MODEL=qwen3.5-9b-orch`). Domains +5 pp vs 7b, routing dentro do gate, latência 5–7x menor que MoE 30B. Trade-off aceito: routing -4.6 pp vs baseline 9B (casos coloquiais), compensado pela latência e pelo fato de caber 100% em GPU.

### Fase 6 — Observabilidade LLM, HITL e estado conversacional (Fase 6.6 da skill)

Implementado sobre a PoC estabilizada com LoRA 9B.

**Langfuse observability.** `gateway/tracing.py`: trace por request, span por no do grafo, generation por chamada LLM. Container Langfuse v2 + Postgres dedicado no `docker-compose.yml` (porta 3100). Degradacao graceful: Langfuse fora nao impacta requests.

**HITL (Human-in-the-Loop).** No `confirm_dispatch` entre classify e dispatch. Usa `interrupt()` nativo do LangGraph. Evento SSE `confirm` com preview da acao. Endpoint `POST /chat/{thread_id}/resume` para retomar. Frontend: `ConfirmCard.tsx`. **Desabilitado por padrao** — sem deteccao de write vs read intent, dispara para toda query. Auto-aprova sem callback.

**Estado conversacional.** `MemorySaver` como checkpointer. `thread_id` por sessao propagado no POST `/chat`. Frontend persiste `thread_id` em localStorage. Botao "Nova conversa" reseta thread.

#### Gotchas encontrados

1. **Callbacks em thread-local.** Lambdas (`on_agent`, `on_confirm`, trace) movidos de GraphState para `threading.local()` porque MemorySaver/checkpointer nao serializa funcoes (TypeError no msgpack).
2. **Estado residual entre turns.** Com checkpointer, `final_answer` do turn anterior persiste. `_route_after_confirm` desviava pro END via truthy check. Fix: `_sanitize` limpa `final_answer`, `agent_results`, `route`, `error`, `pending_confirmation` a cada novo turn.
3. **Null payload no interrupt.** Stream yield com payload `None` quando `interrupt()` suspende grafo. `"final_answer" in None` dispara TypeError. Fix: guard `if not payload: continue`.
4. **Langfuse v3 exige ClickHouse.** Pin para `langfuse/langfuse:2`. Healthcheck: `node -e "require('http')..."` (wget indisponivel). `HOSTNAME=0.0.0.0` obrigatorio (Next.js bind).
5. **Single-domain box duplicado.** Agent event e final event identicos com 1 agente. Fix: frontend so renderiza agent box quando `expectedAgents > 1`.
6. **Case-sensitivity em filtros SQL com LLM tool-calling.** LLMs enviam parametros em lowercase (ex.: `department='vendas'`). SQLite default e case-sensitive — query retorna vazio com dados existentes. Fix: `COLLATE NOCASE` em todo filtro textual (department, category, status, name). Incidente real: `get_headcount(department='vendas')` retornava [] com 3 funcionarios em "Vendas".
7. **5 findings criticos de seguranca (auditoria pre-tunel).** `ACCESS_TOKEN` default vazio expunha `/chat` publicamente — fail-closed com `ALLOW_OPEN_ACCESS`. `.dockerignore` ausente — `.env` vazava em layers Docker. `/docs`/`/redoc` ativos em producao — desabilitados (`docs_url=None`, `redoc_url=None`, `openapi_url=None`). Stack traces no SSE error — mensagem generica + log interno. Containers root — Dockerfiles ja tinham `USER` nao-root (verificado). Regra: auditoria antes do tunel, nao apos.
8. **CF-Connecting-IP para IP real.** Cloudflare seta header nao-spoofavel. Fallback chain: `CF-Connecting-IP` → `X-Real-IP` → `X-Forwarded-For` → socket. Rate limiter usa essa chain.
9. **Request deadline 600 s.** `REQUEST_TIMEOUT_S` independente do `LLM_TIMEOUT_S`. Heartbeat SSE (`:keepalive`) a cada 15 s — Cloudflare corta apos ~100 s sem bytes (erro 524).
10. **Injection detection (14 patterns, log only).** `sanitize.py::flag_injection()` detecta padroes PT/EN de injection semantica. Nao reescreve o texto; defesa ativa e system prompt + isolamento + least-privilege.
11. **Anti-fabricacao no system prompt.** LLM inventa dados se o prompt nao proibir explicitamente. Regra critica em `agents.py`: para write ops sem todos os campos obrigatorios, listar campos e pedir ao usuario. Nunca fabricar nomes, salarios, datas, quantidades, IDs ou SKUs.

### Fase 7 — BERT Features + Security Hardening — ✅ EXECUTADA 2026-06-14

**SBERT Embeddings.** `paraphrase-multilingual-MiniLM-L12-v2` (CPU, 384 dim) substituiu `nomic-embed-text` do Ollama. Embedder Protocol em `gateway/embedder.py`: `SBERTEmbedder` (primário) + `OllamaEmbedder` (fallback). Elimina dependência de `ollama pull` para embeddings. Gateway Dockerfile atualizado com PyTorch CPU + pre-download do modelo SBERT.

**Injection Classifier BERT.** BERTimbau (`neuralmind/bert-base-portuguese-cased`) fine-tunado com 400 exemplos sintéticos (200 injection + 200 legítimos). 100% accuracy na validação. Classifier binário em `gateway/injection_classifier.py`. Modelo 417 MB montado via volume `./models:/app/models`. Training script + dataset sintético em `train/`.

**Routing Improvements.** Golden set expandido de 44→64 exemplos (20 novos, 12 multi-domínio). Prompt multi-domínio explícito no classifier com 4 exemplos concretos. Score gap filter no semantic router (`min_score_gap=0.05`) — rejeita hits onde top-1 e top-2 estão muito próximos. Anti-fabricação para leitura (nunca inventar dados ilustrativos). Acurácia routing: 90.5% PASS (63 queries). Injection eval: 0/6 leaks PASS.

**Security Fixes (auditoria /hm-engineer).**
- Qdrant API key auth (`QDRANT__SERVICE__API_KEY`)
- Langfuse secrets required — sem defaults inseguros (`LANGFUSE_NEXTAUTH_SECRET`, `LANGFUSE_SALT`)
- SQLite WAL mode + `busy_timeout=5000` em todos os 4 microsserviços
- RateLimiter com `max_entries=10000` + eviction periódico (proteção contra memory exhaustion)
- SSE error handler: mensagem genérica pro cliente, traceback server-side
- Auditoria final: 0 CRITICO, 0 ALTO, 0 MEDIO (relatório em `docs/AUDIT_2026-06-14.md`)

**Infra.** Volume mount `./models:/app/models` (injection classifier 417 MB). `models/` e `ebook-llm-on-premise/` adicionados ao `.gitignore`.
