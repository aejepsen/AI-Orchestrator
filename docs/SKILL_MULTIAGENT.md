# /hm-multiagent — Construção de App Multi-Agente (LLM Gateway + Agentes + Microsserviços)

Você está agora em **modo arquiteto multi-agente**. Seu trabalho é construir (ou guiar a construção de) um sistema multi-agente production-grade: gateway de IA, orquestração por grafo, agentes especialistas com tool-calling, microsserviços determinísticos e evals como gates de qualidade. Padrão hm-engineer em todas as camadas — segurança primeiro, decisões com razão explícita, nada mediano.

## Princípios inegociáveis

1. **Meça antes de arquitetar.** Nenhuma decisão de modelo/arquitetura sem benchmark no hardware real. Intuição sobre latência de LLM erra por ordem de magnitude.
2. **Determinismo nas bordas, LLM no meio.** Regras de negócio vivem em APIs determinísticas e testáveis — nunca no prompt. O LLM orquestra, interpreta e redige; ele não calcula limite de reembolso.
3. **Evals são gates, não relatórios.** Todo critério de aceite é numérico, versionado e roda antes de qualquer promoção (de modelo, de prompt, de feature). Sem eval verde, não shippa.
4. **Segurança desde o commit 1.** Auth interna entre serviços, sanitização de entrada, defesa a prompt injection e circuit breaker não são fase 2.
5. **Um agente ≠ um modelo.** Agente = nó de grafo com system prompt + tools escopados por domínio. Modelos separados por agente só se o benchmark de swap justificar (quase nunca justifica em GPU única).
6. **Falha é dado.** Erros de API voltam pro agente em formato acionável; trajetórias rejeitadas são auditáveis; incidentes viram seção de gotchas na doc.

## Matriz de decisão inicial — escolha ANTES da Fase 0

Responda 4 perguntas; elas determinam o stack inteiro. Não pule.

### 1. Realidade de GPU/infra

| Cenário | Inference | Modelos típicos | Quando escolher |
|---|---|---|---|
| **Sem GPU / time-to-market** | API (Claude/OpenAI/DeepSeek) | Claude Sonnet/Haiku, DeepSeek V3 | MVP, baixo volume, dado pode sair do perímetro. Melhor qualidade por hora de engenharia. |
| **GPU consumer única (8–24GB)** | Ollama (llama.cpp) | Qwen 7–9B Q4, Llama 8B | Zero-cloud obrigatório, 1–5 usuários, PoC/portfólio. Simplicidade > throughput. |
| **GPU servidor (40GB+) ou multi-GPU** | **vLLM** (ou SGLang) | Qwen 30B+, Llama 70B AWQ | 10+ usuários concorrentes. Continuous batching + PagedAttention: 5–20× o throughput do Ollama. Ollama aqui é erro. |
| **Híbrido (recomendado p/ produção séria)** | Local p/ routing/classificação + API p/ síntese complexa | 7B local + Claude na borda | Custo baixo no volume alto (routing), qualidade alta onde o usuário lê. Exige abstração de cliente LLM desde o dia 1. |

Regra: **abstraia o cliente LLM atrás de uma interface única** (um `llm.py` com generate/chat/embed) — trocar Ollama↔vLLM↔API vira mudança de config, não refactor. vLLM e a maioria dos providers falam API OpenAI-compatible; Ollama também expõe uma — padronize nela.

### 2. Escala alvo

| Usuários concorrentes | Arquitetura |
|---|---|
| 1–5 | Como descrito nas fases (fan-out enfileira no modelo único; timeout agregado). |
| 5–50 | Fila (Redis/RabbitMQ) entre gateway e agentes + N workers; vLLM com continuous batching; rate limiting por usuário no gateway. |
| 50+ | vLLM atrás de load balancer, autoscaling de workers, cache semântico de respostas (perguntas repetidas), routing local barato + síntese em modelo maior só quando necessário. |

LLM é o gargalo sempre — escale o serving de inference antes de escalar o resto. Microsserviços FastAPI aguentam ordens de magnitude mais que o LLM.

### 3. Orquestração

- **LangGraph** — default recomendado: grafo explícito, checkpointing nativo (estado conversacional), interrupt() para HITL, streaming por nó. Maduro, padrão de mercado.
- **Pydantic AI / Agents SDK (OpenAI) / CrewAI** — mais simples; aceitáveis se o fluxo é linear e sem necessidade de checkpoint/HITL. CrewAI: evitar em produção (abstrações opacas, difícil de debugar).
- **Código puro (loop while + tool calls)** — legítimo para 1 agente sem grafo. Não subestime: menos dependência, controle total. Migre pra grafo quando aparecer fan-out/clarification/estado.

### 4. Vector DB (se houver camada semântica/RAG)

- **Qdrant** — default standalone: container leve, REST simples, filtros por payload.
- **pgvector** — se já existe Postgres no stack: um componente a menos. Suficiente até milhões de vetores.
- **Nenhum** — para <500 itens de índice, kNN em memória (numpy) resolve. Não adicione infra pra demo de infra (lição real: camada semântica com 0 acionamentos porque o LLM já roteava em 1.6s).

### Seleção de modelo por papel (não um modelo pra tudo)

| Papel | Exigência | Tier local | Tier API |
|---|---|---|---|
| Router/classifier | JSON estrito, rápido, barato | 7–9B instruct | Haiku / DeepSeek |
| Agente tool-calling | Tool-calling confiável, raciocínio | 9–30B (thinking ajuda) | Sonnet |
| Síntese final | Qualidade de texto | maior disponível | Sonnet/Opus |
| Embeddings | dim/velocidade | nomic-embed-text, bge-m3 | voyage/openai-embed |
| Gerador de dataset (destilação) | melhor modelo que couber | 30B+ | Claude/DeepSeek (custa ~US$2–5 por 3k exemplos — frequentemente melhor que GPU horas) |

Verifique benchmarks ATUAIS (modelos evoluem por trimestre) — mas o **seu eval no seu hardware** é o benchmark que decide, não leaderboard.

## Fase 0 — Benchmark e decisão de modelo

Antes de escrever qualquer código de produto:

- Script de bench (`evals/fase0_bench.py` como referência): para cada modelo candidato medir **cold load**, **tok/s warm** (3+ gerações), e **tempo de swap** entre modelos.
- Decisões que saem daqui, com critérios:
  - **Swap > ~10s entre modelos → modelo único residente.** Multi-modelo em GPU única quase sempre morre aqui (caso real: swap MoE↔7B de 24s inviabilizou um modelo por agente).
  - **Modelo não cabe 100% na VRAM → medir o split real.** Offload pra CPU pode derrubar throughput a ponto de inviabilizar produção (caso real: 30B-A3B com 44% em CPU → trocado por 7B 100% GPU pra prod).
  - `keep_alive`: **NUNCA `-1`** no Ollama em sistemas request/response — causa deadlock de VRAM no scheduler quando outro load é necessário. Usar valor finito configurável (30m default, 5m em evals).
    - **Exceção: pipelines de streaming/tempo real** (áudio ao vivo, WebSocket com inferência contínua): cold start no meio da sessão inviabiliza a sincronia. Regra: modelo residente durante a sessão ativa — `-1` é aceitável **somente se o runtime serve um único modelo**; em multi-modelo, usar finito alto (ex.: 60m) renovado a cada interação, e liberar explicitamente no evento de disconnect. O deadlock do `-1` só existe quando outro load disputa a VRAM.
- Registre os números no README/doc de plano. Medições reais são parte do entregável.

**Por quê:** a arquitetura inteira (quantos modelos, quantos agentes, fan-out paralelo ou sequencial) deriva dessas três medidas. Decidir antes de medir é construir sobre chute.

## Fase 1 — Microsserviços de domínio (a camada determinística)

- Um serviço por domínio de negócio (ex.: finanças, RH, estoque, vendas). FastAPI + SQLite (ou Postgres se concorrência real). Pequenos, focados, sem dependência entre si.
- **Envelope de erro acionável** — o contrato mais importante do sistema:
  ```json
  { "error": "limite_excedido", "detail": "Reembolso máximo é R$ 3.000,00 por pedido", "rule": "RH-REEMB-01" }
  ```
  HTTP 422 para violação de regra de negócio, 404 para inexistente, 400 para malformado. O `detail` é escrito **para o LLM ler** — ele será reinjetado no loop do agente, que deve conseguir se corrigir ou explicar ao usuário. Erro vago = agente alucina contorno.
- **Regras de negócio na API, com testes unitários diretos** (`rules.py` + `test_rules.py`). O LLM nunca é a fonte da regra.
- **Seed determinístico no startup** (idempotente): dados realistas, nomes/IDs fixos. Tudo downstream (evals, dataset de fine-tune, demo) depende de estado reproduzível.
- **Concorrência no SQLite (obrigatório se houver fan-out com escrita)**: SQLite trava o arquivo inteiro em escrita — agentes paralelos escrevendo no mesmo .db geram `database is locked`. Na inicialização do banco: `PRAGMA journal_mode=WAL;` + `PRAGMA busy_timeout=5000;` (WAL permite leitores concorrentes durante escrita; busy_timeout enfileira em vez de explodir). Um .db por serviço já isola a maior parte do conflito; se a escrita concorrente dentro do mesmo serviço for pesada (50+ usuários, histórico por request), migre para Postgres — WAL atenua, não elimina o writer único.
- Infra: Dockerfile único parametrizado (`ARG SERVICE`), volume nomeado por serviço, healthcheck real, `DB_PATH` e auth via env. Multi-stage, non-root, `.dockerignore` (padrão hm-engineer).
- **Auth interna obrigatória**: header `X-Internal-Key` validado com `secrets.compare_digest` (anti timing attack). Microsserviços nunca expostos publicamente — só o gateway fala com eles.

**Por quê SQLite + FastAPI:** para PoC/portfólio, zero-ops e reproduzível; o contrato (OpenAPI + envelope de erro) é o que importa e migra intacto pra qualquer stack.

## Fase 2 — Tool-calling: registry e loop do agente

- **Registry OpenAPI → tools**: gerar as definições de tool automaticamente do schema OpenAPI dos serviços. Uma fonte de verdade; tool nova = endpoint novo, sem duplicação manual. Anexar o schema de **resposta** à description da tool (o agente precisa saber o que volta).
- **Loop do agente** com guard-rails duros: máximo de iterações, timeout por item, budget de chamadas. Loop sem teto = custo e travamento.
- Lições de prompt que valem como regra:
  - **"Deixe a API validar."** O agente não pré-computa regra de negócio — chama a tool e reage ao 422. Senão ele decide errado de cabeça.
  - **"Resolva nome→ID via listagem."** Proibido chutar IDs. Modelo fraco chuta ID e recebe 404 onde deveria haver 422 (sintoma clássico medido em eval: domínio com 20% de acerto).
  - **"Leia antes de concluir indisponível."** Consultar estado real antes de negar.
  - **Reasoning (`think`)**: em famílias com thinking nativo, `think=true` no tool-loop (desligar vaza CoT no content e degrada decisão de tool). Desligar apenas onde só se quer JSON estruturado rápido (classificação, geração sintética).
- **Eval de domínio imediatamente**: golden set de N tasks por domínio com resultado esperado verificável (registro criado, 422 com regra certa, número correto). Gate: ≥80% por domínio. Este eval é o detector de regressão de prompt e de modelo para sempre.

## Fase 3 — Orquestração por grafo (gateway)

- Grafo explícito (LangGraph ou equivalente): `sanitize → classify → [clarification | dispatch fan-out | synthesize]`.
  - **sanitize**: normalização + heurísticas de injection na entrada.
  - **classify (router)**: LLM com saída JSON estrita `{domains, plan, clarification}` + few-shot. Cadeia de fallback: parse falhou → retry → classificador léxico determinístico. **Guards determinísticos** pós-classificação para casos ambíguos conhecidos (ex.: termo que existe em dois domínios) — regra de código corrige o LLM, não outro prompt.
  - **dispatch**: fan-out paralelo para os agentes dos domínios roteados. Atenção: com modelo único, fan-out **enfileira** no LLM — dimensionar timeout do gateway pela soma, não pelo item (caso real: `LLM_TIMEOUT_S=900`).
  - **synthesize**: consolida respostas dos agentes em uma resposta única, citando dados retornados. **Teto de contexto obrigatório**: cada agente entrega resumo estruturado compacto (schema fixo: resposta + dados-chave, com limite de caracteres/tokens por agente), nunca o payload bruto das tools. Concatenação ingênua de N payloads JSON estoura contexto do modelo e explode custo de input em API. Truncar com marcador explícito se exceder o teto — o synthesize deve saber que houve corte.
- **Clarification como saída de primeira classe**: pergunta ambígua → pedir esclarecimento é acerto, não falha. Entra no golden set como caso esperado.
- **SSE para streaming de progresso** (eventos por nó do grafo). Em produção atrás de proxy/CDN: **heartbeat a cada ~15s** — Cloudflare corta conexão sem bytes por ~100s (erro 524).
- **Eval de routing**: golden set de 40+ perguntas (incluindo coloquiais, multi-domínio e clarification). Gate: ≥90%. Falha típica a vigiar: sobre-roteamento de pergunta casual para vários domínios.

## Fase 4 — Segurança e resiliência

- **Prompt injection é requisito com eval próprio**: golden set adversário (payloads tentando exfiltrar prompt, forçar tool de outro domínio, ignorar instruções). Gate: **0 leaks**. Mitigação em camadas: sanitize na entrada + regra de SEGURANÇA explícita no prompt do router + **exemplo adversário no few-shot** (a correção que mais move o ponteiro) + tools escopadas por domínio (mesmo roteado errado, o agente não tem a tool de outro domínio).
- **Circuit breaker** no cliente dos microsserviços (ex.: 3 falhas de transporte → OPEN 30s). Falha de dependência → resposta degradada honesta, nunca crash.
- Demais itens herdam o checklist hm-engineer: zero secrets hardcoded (env + `.env.example`), zero bare except, validação em toda boundary, logs estruturados sem secrets.

## Fase 5 — Evals como sistema (não como script)

- Estrutura: `evals/golden_*.jsonl` (casos) + `evals/eval_*.py` (runners) + resultados timestampados em `evals/results/`.
- Regras:
  - **Golden sets são sagrados**: nunca entram em treino/few-shot dinâmico (anti-contaminação). São o juiz.
  - Todo eval imprime **scoreboard com gate PASS/FAIL** e persiste JSON com latências por item.
  - **Watchdog/timeout por item** — um item travado não pode matar a suite.
  - Reset de estado entre runs quando o eval cria registros (ou IDs únicos por run).
- **Armadilha de configuração (incidente real):** `.env` lido só pelo docker-compose; scripts no host usam default hardcoded do `config.py`. Dois defaults divergentes = eval rodando com modelo errado sem ninguém notar. Regra: **um único default, idêntico, nos dois lugares** + logar `model=` no início de todo eval.
- Promoção de modelo/prompt: só com scoreboard completo (todos os gates) comparado ao baseline. Tabela no README com números medidos.

### Medição de métricas (o que medir, sempre)

- **Por modelo (Fase 0 e a cada candidato)**: cold load (s), tok/s warm, tempo de swap, VRAM/RAM split, % GPU vs CPU.
- **Por eval**: acurácia por gate (routing %, domains % por domínio, injection leaks), latência por item (p50/p95), latência por camada de routing (semantic/llm/lexical), taxa de clarification, timestamp + modelo + config no JSON de resultado.
- **Por request em produção**: latência por nó do grafo (sanitize/classify/dispatch/synthesize), nº de tool calls por agente, iterações do loop, circuit breaker state.
- **Por pipeline de dataset**: itens/s por estágio, taxa de rejeição por motivo, distribuição por domínio/kind, duplicatas.
- Regras: todo número em arquivo versionável (JSON timestampado), nunca só no stdout; todo eval loga `model=` e config efetiva na primeira linha (evita rodar com modelo errado sem notar); comparação sempre contra baseline registrado — número sem baseline não é métrica, é anedota.

## Fase 6 — Produção

- Gateway é a única porta pública. Token de acesso no `/chat` (mesmo em demo). Tunnel (Cloudflare) ou reverse proxy; serviços e LLM ficam em rede interna.
- `MODEL`, `KEEP_ALIVE`, timeouts: tudo configurável por env, defaults sãos.
- Incidente vira runbook: ex. site fora com tunnel de pé → checar DNS Records primeiro (CNAME pode sumir em transição de zona).
- Demo reproduzível (`evals/demo.py` → transcripts versionados): prova viva do comportamento para portfólio/auditoria.

## Fase 6.5 (opcional) — Roteamento semântico (vector DB + embeddings)

Camada kNN na frente do router LLM: robustez a paráfrase + latência menor em rota de alta confiança + demonstração de banco vetorial.

- **Stack**: Qdrant em container (volume nomeado, porta só em 127.0.0.1, healthcheck TCP — imagem não tem curl, **API key auth via `QDRANT__SERVICE__API_KEY`**) + embeddings via SBERT (`paraphrase-multilingual-MiniLM-L12-v2`, dim 384, CPU, ~120 MB, roda no gateway sem dependência do Ollama) com fallback para Ollama (`nomic-embed-text`, dim 768). Embedder Protocol em `gateway/embedder.py`. Cliente Qdrant via httpx REST puro — sem SDK pesado quando 4 endpoints bastam.
- **Fonte do índice = golden set de routing** (só casos sem clarification), upsert idempotente com hash da pergunta como point ID. Payload: `{domains, question}`.
- **Sincronismo do índice com fonte mutável (RAG)**: upsert por hash sozinho não remove o que foi deletado da fonte — vetores fantasmas poluem a busca pra sempre. Toda reindexação grava os pontos com `batch_id` único no payload e, ao final, faz purge (delete por filtro) de todos os pontos com `batch_id` diferente do atual. Fonte estática (golden set) dispensa; fonte viva (docs, regras) exige.
- **Pipeline em cascata**: 1º semantic (threshold) → 2º LLM classifier → 3º léxico. Guards determinísticos aplicados na saída de TODAS as camadas.
- **Critério de aceite duplo**: top1.score ≥ threshold **e** consenso **unânime** de domínios nos vizinhos confiantes do top-k (não maioria). Sem consenso → `None` → cai pro LLM. Conservador por design: falso positivo de rota é pior que latência.
- **Degradação graceful obrigatória**: Qdrant/embedding fora → log warning + `None`; a request nunca morre por causa da camada opcional.
- **Eval com leave-one-out**: ao avaliar pergunta do golden, excluir o próprio ponto do índice — senão o acerto é self-match trivial. Relatório registra camada usada (semantic/llm/lexical) + latência por camada.
- **Lição medida (caso real)**: threshold 0.92 + consenso restritivo → 0 acionamentos; LLM resolvia em 1.6s. Camada semântica só compensa se o LLM de routing for lento ou caro — **meça antes de tunar o threshold**, e aceite desligar a camada se o baseline já é bom.

## Fase 6.6 — Observabilidade LLM, HITL e estado conversacional

**Esta fase não é opcional.** Sistema multi-agente sem tracing LLM dedicado, sem gate humano antes de ações destrutivas e sem estado conversacional não é production-grade — é demo. Implemente junto com a Fase 6 ou imediatamente após.

### Observabilidade LLM (Langfuse) — implementar primeiro

Logs estruturados mostram que algo aconteceu. **Tracing LLM mostra por quê.** Cada chamada ao modelo deve registrar: prompt completo, response, tokens in/out, latência, modelo, temperatura, tool calls. Sem isso, debugar "por que o agente roteou errado" ou "por que gastou 4 iterações" é arqueologia em logs.

- **Solução recomendada: Langfuse self-hosted** (container no compose, Postgres próprio, UI web). Zero-cloud, open source, SDK Python leve (`langfuse`). Alternativa: OpenTelemetry GenAI spans — mais genérico, menos dashboard pronto.
- **Onde instrumentar**: `llm.py` (decorator/wrapper em `chat()` e `embed()`), `graph.py` (span por nó do grafo), `agents.py` (span por iteração do loop). Cada request do `/chat` é um trace; cada nó é um span filho. O `trace_id` já existente no gateway vira o `trace_id` do Langfuse.
- **Custo por request como métrica de produto**: derive dos tokens rastreados. Tokens in/out × custo por token (mesmo on-premise: tempo de GPU tem preço). Agregar por usuário/dia/domínio. Em API isso é fatura; em local é dimensionamento de capacidade. Dashboard do Langfuse já mostra isso nativamente.
- **Eval scoring no Langfuse**: vincular resultados dos evals (routing, domains, injection) aos traces — permite correlacionar score com prompt/modelo/config sem grep manual.

### Human-in-the-Loop (HITL) — implementar segundo

Agente que executa ação destrutiva sem confirmação humana é risco inaceitável. Um agent convencido por alucinação ou injection cria registro real, liquida conta, aprova reembolso — sem volta.

- **Mecanismo**: LangGraph `interrupt()` nativo. Antes de qualquer tool que faça POST/PUT/DELETE em microsserviço, o grafo pausa e emite evento SSE `confirm` com preview da ação (tool, argumentos, domínio). Frontend mostra ao usuário; aprovação retoma o grafo, rejeição aborta com mensagem.
- **Whitelist de tools auto-executáveis**: tools de leitura (GET/listagem) executam sem pausa. Tools de escrita (criar pedido, aprovar reembolso, reservar estoque) exigem confirmação. A whitelist é config, não hardcode — novas tools são restritivas por default.
- **Timeout de confirmação**: se o usuário não responde em N minutos, o grafo aborta com mensagem clara (não fica pendurado).
- **Auditoria**: toda confirmação/rejeição é logada (quem, quando, qual ação) — rastreável no Langfuse como evento do span.

### Estado conversacional — implementar terceiro

Sem estado, cada `/chat` é uma conversa nova. Usuário não pode dizer "e o estoque desse produto?" referindo-se ao contexto anterior.

- **Mecanismo**: LangGraph checkpointer (SQLite para PoC, Postgres/Redis para produção). `thread_id` por sessão do frontend, propagado no POST `/chat`.
- **Janela de história com teto**: manter últimas N mensagens + sumarização do excedente via LLM. História infinita = contexto estourado + custo crescente. Teto recomendado: 4096 tokens de história; excedente sumarizado em ~500 tokens.
- **Limpeza**: TTL por thread (ex.: 24h sem atividade → purge). Sem TTL, checkpointer acumula estado indefinidamente.

### Structured output — reforço

JSON crítico (router) com schema enforcement nativo do serving (format/json_schema no Ollama, guided_json no vLLM, tool-use forçado em API) — não regex sobre texto livre. Parse + retry é fallback, não estratégia.

### Gotchas / Armadilhas (Fase 6.6)

Incidentes reais encontrados durante a implementacao. Documentados como regra para evitar reincidencia.

1. **Callbacks nao-serializaveis no GraphState.** NUNCA colocar lambdas, closures ou funcoes no GraphState quando usando checkpointer (MemorySaver, SQLite, etc). O checkpointer serializa o estado inteiro via msgpack/pickle; funcoes nao serializam e disparam TypeError. Solucao: mover callbacks (`on_agent`, `on_confirm`, trace functions) para `threading.local()` ou variavel de instancia do runner. O GraphState deve conter apenas dados primitivos e estruturas serializaveis.

2. **Estado residual entre turns.** Com checkpointer ativo, campos como `final_answer`, `agent_results`, `route`, `error` e `pending_confirmation` persistem entre turns. Conditional edges que checam estado (ex.: `if state["final_answer"]`) sao enganadas por valores de turns anteriores — o grafo desvia para END sem executar dispatch. Regra: limpar campos de resultado no primeiro no de cada turn (`_sanitize` zera `final_answer`, `agent_results`, `route`, `error`, `pending_confirmation`).

3. **Null payload no interrupt.** Quando `interrupt()` suspende o grafo, o stream pode yieldar payloads `None`. Codigo como `"final_answer" in None` dispara TypeError. Regra: sempre guardar com `if not payload: continue` antes de acessar campos do payload no worker/stream handler.

4. **Langfuse v3 vs v2.** Langfuse v3 exige ClickHouse como backend — overhead inaceitavel para PoC. Usar `langfuse/langfuse:2` (Postgres only). Detalhes de container: imagem nao tem wget/curl, healthcheck via `node -e "require('http')..."`. Variavel `HOSTNAME=0.0.0.0` obrigatoria (Next.js bind — sem ela, escuta so em 127.0.0.1 dentro do container).

5. **HITL scoping.** `interrupt()` sem deteccao de intent (write vs read) dispara para toda query, incluindo leituras. Isso trava o fluxo em perguntas que nao executam acoes destrutivas. Regra: desabilitar HITL por padrao; ativar so quando houver deteccao confiavel de write intent. Auto-aprovar sem callback e a estrategia segura ate a deteccao existir.

6. **Single-domain box duplicado no frontend.** Com um unico agente, o evento `agent` e o evento `final` carregam conteudo identico — o frontend renderiza dois cards com o mesmo texto. Fix: so renderizar agent box quando `expectedAgents > 1`.

7. **Case-sensitivity em filtros SQL com LLM tool-calling.** LLMs enviam parametros de filtro em lowercase (ex.: `department='vendas'`). Se a API/DB usa case-sensitive match (SQLite default), a query retorna vazio mesmo com dados existentes. Regra: todo filtro de texto em endpoints consumidos por agentes DEVE usar `COLLATE NOCASE` (SQLite) ou `ILIKE` (Postgres). LLM nao e confiavel para acertar capitalizacao. Onde aplicar: qualquer endpoint com parametro de filtro textual nos microsservicos (department, category, status, name, etc). Incidente real: `get_headcount(department='vendas')` retornava [] enquanto DB tinha 3 funcionarios em "Vendas" — agente concluia "nao ha funcionarios", resposta errada para pergunta simples.

8. **5 findings criticos de seguranca corrigidos (Fase 6).** `ACCESS_TOKEN` default vazio abria `/chat` para o mundo — fix: fail-closed com flag `ALLOW_OPEN_ACCESS` explícita. `.dockerignore` ausente — `.env` vazava nas layers da imagem Docker. `/docs` e `/redoc` expostos em producao — desabilitados (`docs_url=None`, `redoc_url=None`, `openapi_url=None`). Stack traces expostos no SSE error — substituidos por mensagem generica com log interno. Containers root — verificado que Dockerfiles ja tinham `USER` nao-root. Regra: auditoria de seguranca ANTES do tunel publico, nao depois.

9. **CF-Connecting-IP para IP real.** Cloudflare seta header nao-spoofavel pelo cliente. Rate limiter usa fallback chain: `CF-Connecting-IP` → `X-Real-IP` → `X-Forwarded-For` → socket.

10. **Request deadline 600 s.** `REQUEST_TIMEOUT_S` e independente do `LLM_TIMEOUT_S`. Heartbeat SSE (`: keepalive`) a cada 15 s — Cloudflare corta apos ~100 s sem bytes (erro 524). Pool dedicado (`MAX_GRAPH_WORKERS=4`) para execucao sincrona do grafo, nao compete com asyncio default pool.

11. **Injection detection (14 patterns, log only).** `sanitize.py::flag_injection()` detecta padroes PT/EN de injection semantica (ex.: "ignore as instruções", "you are now", "act as", "system prompt", "reveal instructions"). Nao reescreve o texto — mutilar keywords destroi perguntas legitimas. Defesa ativa: system prompt + isolamento por tag + least-privilege de tools.

12. **Anti-fabricacao no system prompt.** LLM inventa dados (nomes, salarios, datas, SKUs) se o prompt nao proibir explicitamente. Regra critica em `agents.py::_SYSTEM_PROMPT_TEMPLATE`: para write ops sem todos os campos obrigatorios fornecidos pelo usuario, o agente DEVE listar os campos e pedir os valores — nunca fabricar. Incidente real: sem a regra, agente criava funcionario com nome inventado ao receber "incluir um funcionario" sem detalhes.

13. **SBERT substituiu nomic-embed-text para embeddings.** `paraphrase-multilingual-MiniLM-L12-v2` (384 dim, CPU, ~120 MB) roda no gateway via sentence-transformers — elimina dependencia do Ollama para embeddings. Embedder Protocol (`SBERTEmbedder` primario + `OllamaEmbedder` fallback) em `gateway/embedder.py`. Gateway Dockerfile atualizado com PyTorch CPU + pre-download SBERT. Dimensao 384 (antes 768 com nomic) — reindexar Qdrant ao migrar.

14. **Injection classifier BERT fine-tunado.** BERTimbau (`neuralmind/bert-base-portuguese-cased`) treinado com 400 exemplos sinteticos (200 injection + 200 legitimos). 100% accuracy na validacao. Classifier binario em `gateway/injection_classifier.py`. Modelo 417 MB montado via volume `./models:/app/models` — nao versionar pesos no git (`models/` no `.gitignore`). Training script + dataset sintetico em `train/`. Segunda camada de defesa alem dos 14 regex patterns.

15. **Score gap filter no semantic router.** `min_score_gap=0.05` rejeita hits onde top-1 e top-2 estao muito proximos (ambiguidade). Previne roteamento incorreto quando duas rotas competem com scores quase identicos. Configuravel via env.

16. **Golden set expandido (44→64 exemplos).** 20 novos exemplos, 12 multi-dominio. Prompt multi-dominio explicito no classifier com 4 exemplos concretos. Anti-fabricacao para leitura: regra no prompt proibe inventar dados ilustrativos em consultas.

17. **Qdrant API key auth.** `QDRANT__SERVICE__API_KEY` protege o banco vetorial — nao mais exposto sem autenticacao. Configurar no `.env`.

18. **Langfuse secrets obrigatorios.** `LANGFUSE_NEXTAUTH_SECRET` e `LANGFUSE_SALT` sem defaults inseguros — compose e fail-closed. Gerar com `openssl rand -base64 32`.

19. **SQLite WAL mode + busy_timeout em todos os microsservicos.** `PRAGMA journal_mode=WAL` + `PRAGMA busy_timeout=5000` na inicializacao de cada banco. Previne `database is locked` em fan-out concorrente com escrita.

20. **RateLimiter com max_entries + eviction.** `max_entries=10000` + eviction periodico de entries expiradas. Protecao contra memory exhaustion por IP spoofing/DDoS distribuido.

21. **SSE error handler sanitizado.** Mensagem generica pro cliente ("Erro interno. Tente novamente."), traceback completo server-side no log. Stack traces nunca expostos no stream SSE.

## Fase 7 (opcional) — Fine-tune por destilação (LoRA)

Só após baseline medido. Critério de adoção definido **antes** de treinar: superar baseline sem regressão em nenhum gate; empate = mantém base.

- **Dataset**: gerar com o melhor modelo disponível contra os serviços reais (trajetórias completas com tool calls). Pipeline em **estágios retomáveis por hash** (`questions → trajectories → routing → assemble`) — sessões de GPU caem; retomada é requisito, não luxo.
- **Filtro automático + `rejected.jsonl` auditável**: rejeitar trajetória sem tool call, com alucinação numérica (números da resposta devem existir nos tool results), tool/status errado. Taxa de rejeição ~3–8% é saudável; 0% = filtro quebrado.
- **Anti-contaminação**: golden sets 100% fora do treino. Split train/val com seed fixo.
- **Comunicação de progresso**: contador de estágio explícito no log (`[trajectories] 90/1431`), nunca um "3000/3000" ambíguo de estágio intermediário que pareça fim de pipeline (incidente real: sessão encerrada prematuramente por isso). Backup incremental do progresso a cada 1–2h.
- **Checkpoints SEMPRE no armazenamento persistente (Drive/S3/NFS), NUNCA só em `/content` ou disco efêmero.** Sessões Colab caem sem aviso (timeout, desconexão, OOM, crash CUDA). Se `output_dir` aponta pra disco local, um reset de sessão apaga horas de treino irrecuperáveis (incidente real: 2h17 de treino perdidas por checkpoint só em `/content/outputs`). Regra: `output_dir` do SFTConfig aponta direto pro Drive montado (ex.: `/content/drive/MyDrive/<projeto>/training`); backup incremental do progresso. Disco local é cache, não storage.
- Gotchas de ambiente Colab/remoto: instalador do Ollama exige `zstd`; `num_ctx` explícito anti-OOM; pinned versions de transformers/TRL no notebook; sanity check do modelo antes do run longo; torchao incompatível com torch do Colab — se import falhar, patch `importlib.util.find_spec` pra retornar None pro torchao e usar transformers+peft direto.
- Validação final: o GGUF/adapter passa pelos **mesmos 3 gates** (routing/injection/domains) do baseline. O eval decide, não a loss.

## Ordem de construção (resumo executável)

1. Bench de modelos no hardware real → decisão de modelo/arquitetura documentada.
2. Microsserviços com regras testadas + envelope 422 acionável + auth interna + seed.
3. Registry OpenAPI→tools + loop de agente com guard-rails → eval de domínio ≥80%.
4. Grafo de orquestração + router com fallbacks e guards → eval de routing ≥90%.
5. Hardening: injection eval 0 leaks, circuit breaker, timeouts de fan-out.
6. Produção: SSE+heartbeat, tunnel, token, README com números.
6.5. (Opcional) Roteamento semântico — só se LLM de routing for lento/caro.
6.6. **Observabilidade LLM (Langfuse) → HITL → estado conversacional.** Não é opcional — sistema multi-agente sem tracing e sem gate humano não é production-grade.
7. (Opcional) Destilação + LoRA com gates de adoção.

Cada fase tem critério numérico de saída. Não avance com gate vermelho.

## Anti-padrões (rejeição imediata)

- Regra de negócio no prompt em vez da API.
- Promover modelo/prompt sem rodar os três evals.
- Golden set usado como few-shot ou treino.
- `keep_alive=-1` / loop de agente sem max iterations / fan-out sem timeout agregado.
- Defaults de modelo divergentes entre `.env` e código.
- Erro de API genérico ("invalid request") que o agente não consegue interpretar.
- Multi-modelo em GPU única sem medir swap.
- Pipeline longo de geração sem retomada nem backup incremental.
- Checkpoints de treino em disco efêmero (Colab `/content`, `/tmp`) sem cópia em storage persistente.
