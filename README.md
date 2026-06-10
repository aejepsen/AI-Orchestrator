##### AI-Orchestrator

**AI Gateway multi-agente 100% on-premise**: um orquestrador MoE roteia perguntas de negócio para subagentes especialistas que executam tool-calling contra microsserviços FastAPI determinísticos. A regra de negócio vive na API — nunca no modelo.

PoC de portfólio com **números medidos em hardware consumer** (RTX 3060 12 GB + 14 GB RAM): todos os gates abaixo foram executados, não estimados.

---

Resultados (medidos)

| Gate                             | Resultado                                                                   | Critério       | Evidência                                        |
| -------------------------------- | --------------------------------------------------------------------------- | --------------- | ------------------------------------------------- |
| Geração MoE warm (Fase 0)      | **17.97 tok/s**                                                       | ≥ 5 tok/s      | `evals/fase0_bench.py`                          |
| Subagentes por domínio (Fase 2) | **38/40 = 95%** (financas 9/10, rh 10/10, estoque 9/10, vendas 10/10) | ≥ 80%/domínio | `evals/eval_domains.py`, 40 tasks golden        |
| Roteamento (Fase 3)              | **38/42 = 90.5%** (reconfirmado pós-endurecimento anti-injection)    | ≥ 90%          | `evals/eval_routing.py`, 42 perguntas           |
| Injection (Fase 4)               | **0/6 vazamentos cross-domain**                                       | 0 leaks         | `evals/eval_injection.py`, 6 casos adversários |
| Testes determinísticos          | **182 passando** (regras de negócio + gateway)                       | 100%            | `pytest services gateway/tests`                 |
| Demo multi-domínio SSE          | **end-to-end OK** (3 domínios, fan-out/fan-in, 480 s)                | funcional       | `evals/demo.py` → `demo/transcripts.md`      |

---

Arquitetura

```
                         ┌──────────────────────────────────────────────┐
                         │  Gateway (FastAPI, porta 8100)               │
 cliente ── POST /chat ─▶│  LangGraph: sanitize → classify ─┬─ clarif.  │
            (SSE)        │            dispatch (fan-out) ───┤           │
                         │            synthesize (fan-in) ──┘           │
                         └───────┬──────────────────────┬───────────────┘
                                 │ tools (OpenAPI)      │ /api/chat
                  X-Internal-Key │                      ▼
        ┌──────────┬─────────────┼──────────┐   ┌───────────────────┐
        ▼          ▼             ▼          ▼   │ Ollama (11435)    │
   ┌─────────┐┌─────────┐ ┌─────────┐┌─────────┐│ qwen3:30b-a3b MoE │
   │financas ││   rh    │ │ estoque ││ vendas  ││ residente         │
   │  :8101  ││  :8102  │ │  :8103  ││  :8104  ││ keep_alive=-1     │
   └─────────┘└─────────┘ └─────────┘└─────────┘│ ~10GiB GPU+8GiB RAM│
    FastAPI + Pydantic v2 + SQLite por serviço  └───────────────────┘
    regras de negócio determinísticas, erro 422 {error, detail, rule}
```

**Decisões medidas, não assumidas:**

| Decisão                                                       | Por quê (medição)                                                                                                                                                                                                           |
| -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **MoE-único residente** em vez de swap orquestrador↔7B | Swap medido em 8.8–24 s por troca (gate era ≤ 5 s). MoE warm a 18 tok/s atende tudo; subagente virou**nó do grafo com system prompt + tools escopados por domínio** — isolamento por escopo de tool, não por peso. |
| `think=true` no qwen3                                        | Com `think=false` o modelo vaza CoT no `content` E pré-computa regra de negócio sem chamar a tool (eval caiu pra 82.5%, rh 60%). Com `think=true` + regras explícitas no prompt → 95%.                               |
| Tools geradas do `openapi.json`                              | Única fonte de verdade é a API; schema drift impossível. 422/404 voltam como **observação** pro loop do agente, nunca exceção.                                                                                   |
| `LLM_TIMEOUT_S=900` no gateway                               | Fan-out de 3 agentes enfileira no MoE único (Ollama serializa): o 3º agente espera ~2 loops inteiros — 300 s estourava (medido na demo).                                                                                    |

Latências medidas (warm, RTX 3060 + split CPU)

| Etapa                                                                    | Latência típica         |
| ------------------------------------------------------------------------ | ------------------------- |
| Classify (roteamento)                                                    | 15–70 s                  |
| Task single-domain (loop 2–4 iterações de tool-calling)               | 26–216 s (mediana ~55 s) |
| Caso demo 3 domínios end-to-end (route → fan-out paralelo → síntese) | **480 s**           |
| Cold load do MoE (uma vez por boot)                                      | 55 s                      |

> Hardware consumer + modelo 30B parcialmente em CPU = latência de demo, não de produção. Em produção o mesmo desenho roda com endpoint dedicado; nada na arquitetura depende da velocidade do modelo.

Segurança

- **Sanitização no boundary** (`gateway/sanitize.py`): strip de tokens especiais ChatML (`<|...|>`) e wrapper tags antes de qualquer LLM.
- **Router endurecido**: instruções injetadas ("ignore as instruções...", "agora você é...", falsa autoridade) não adicionam domínios à rota. 1º eval: 2/6 vazamentos de roteamento (zero tool-calls cross-domain — least-privilege segurou) → regra de segurança + exemplo adversário no prompt → **0/6**.
- **Least-privilege por escopo**: cada subagente só enxerga as tools do seu domínio — mesmo "convencido", não tem como chamar outra API.
- **API key interna** (`X-Internal-Key`, `hmac.compare_digest`): serviços recusam chamada que não venha do gateway (401).
- **Circuit breaker por domínio**: 3 falhas de transporte → OPEN 30 s → half-open. 4xx (regra de negócio) não conta como falha.

Observabilidade

`trace_id` por request propagado pelo grafo; log JSON estruturado por nó (nó, latência, domínios). Eventos SSE em tempo real: `route` → `agent` (um por subagente concluído) → `final`.

Como rodar

```bash
docker compose up -d --build          # ollama + gateway + 4 microsserviços
docker exec ai-orchestrator-ollama ollama pull qwen3:30b-a3b

# pergunta multi-domínio via SSE
curl -N -X POST localhost:8100/chat -H 'content-type: application/json' \
  -d '{"question": "Posso aceitar um pedido de 100 unidades do SKU TEC-MEC-005 com 15% de desconto?"}'
```

Evals e testes:

```bash
python -m venv .venv && .venv/bin/pip install -r services/requirements.txt
.venv/bin/python -m pytest services gateway/tests -q        # 182 testes, sem LLM
INTERNAL_API_KEY=dev-internal-key .venv/bin/python evals/eval_domains.py    # 40 tasks
.venv/bin/python evals/eval_routing.py                       # 42 perguntas
INTERNAL_API_KEY=dev-internal-key .venv/bin/python evals/eval_injection.py  # 6 casos
INTERNAL_API_KEY=dev-internal-key .venv/bin/python evals/demo.py            # 5 conversas
```

Estrutura

```
gateway/            # Experience Layer: graph (LangGraph), router, agents, sanitize, SSE
gateway/tools/      # registry OpenAPI→tools + circuit breaker
services/           # 4 microsserviços FastAPI (financas, rh, estoque, vendas)
evals/              # golden sets + gates (fase0_bench, domains, routing, injection, demo)
demo/               # transcripts gravados das 5 conversas
PLANO_EXECUCAO.md   # plano por fase com as-built e números medidos
PROPOSAL.md         # visão do padrão AI Gateway
```

Backlog (fora do escopo da PoC)

- Reset de estado dos serviços entre runs de eval (resíduo financas-09: conta já liquidada de run anterior).
- Registry anexar campos do schema de resposta à description da tool (resíduo estoque-03: modelo julga capacidade da tool só pela descrição).
- Streaming token-a-token na síntese; RAG sobre documentos; UI.
