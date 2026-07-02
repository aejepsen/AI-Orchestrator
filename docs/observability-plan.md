# Plano de Observabilidade — 4 Frameworks + 4 Pilares

**Data**: 2026-06-24
**Objetivo**: Portfólio demonstrando prontidão enterprise em LLM ops

---

## Arquitetura de telemetria

```
                        AI-Orchestrator (LangGraph)
                                 │
                    OpenTelemetry SDK (única instrumentação)
                                 │
                         OTel Collector (fan-out)
                        ┌────────┼────────┐
                        ▼        ▼        ▼
                   LangSmith   Langfuse   Phoenix
                   (cloud)    (local)    (local)
                        │                  │
                        └──── OTel ────────┘
```

**LangSmith** faz papel duplo: trace destination **e** OTel endpoint. As outras duas recebem via Collector.

---

## Matriz de métricas → framework × viabilidade

Cada métrica do seu texto classificada como:

| Símbolo | Significado |
|---------|-------------|
| ✅ | Implementável agora, local, sem custo |
| 🟡 | Implementável com adaptação (precisa de LLM externo como judge) |
| 🔴 | Aspiracional — documentável no portfólio como "pronto para produção" |

---

## 1. Impacto de Negócio & Financeiro (C-Level)

| Métrica | Status | Framework | Como implementar | Sinal de portfólio |
|---------|--------|-----------|------------------|-------------------|
| **Cost-per-Task** | ✅ | Langfuse + LangSmith | Token counting via `token_usage` nas spans do LangGraph. Langfuse calcula custo automaticamente se configurar `price_per_1k_tokens`. LangSmith idem pelo `Cost` tab. Fórmula: `(prompt_tokens × $/1k + completion_tokens × $/1k) / total_tasks` | Sabe calcular TCO de LLM, mesmo com Ollama local (usa preço de GPU equivalente) |
| **RAOI (Return on AI Investment)** | 🟡 | Todos (dashboard customizado) | Métrica composta: `(horas_manual_evitadas × custo_hora_humano - custo_infra_gpu - custo_tokens) / custo_total`. Documentável como fórmula no README e no eval dashboard. Não automatizada (precisa de dados reais de operação). | Pensamento C-level: não é só engenharia, é negócio |

**O que dá pra fazer agora**: Cost-per-Task é 100% viável. RAOI como cálculo documentado no `PLANO_SEMIOSE.md`.

---

## 2. Engenharia & Eficiência (Devs & Data Engineers)

| Métrica | Status | Framework | Como implementar | Sinal de portfólio |
|---------|--------|-----------|------------------|-------------------|
| **Task Success Rate** | ✅ | LangSmith | LangGraph traces já mostram `error` vs `ok` por node. LangSmith agrega `run.status`. Adicionar `task_success` como métrica no `eval_semiose.py`. Threshold: se `stop_reason != "answer"` → fail. | Sabe medir qualidade de agentes, não só latência |
| **Tool Call Efficiency** | ✅ | Langfuse + Phoenix | Langfuse: ratio `tool_calls / total_spans`. Phoenix: latência por tool no trace waterfall. Métrica nova: `tools_per_task` (média e P95). | Sabe que tool calling caro é problema real |
| **Semantic Drift** | ✅ (já existe!) | Já implementado (`eval_semiose.py`) | `contextual_drift_score`: cosseno entre embedding da query original e embedding após enriquecimento. Gate calibrado em 0.12. Phoenix pode visualizar a distribuição de drift ao longo do tempo. | Já tinham métrica avançada de drift antes de ser moda |

**O que já existe e o que falta**: Drift score já está implementado e passando. Task Success e Tool Efficiency precisam de 2-3 novas métricas no `eval_semiose.py`.

---

## 3. Alinhamento & Riscos (Governança)

| Métrica | Status | Framework | Como implementar | Sinal de portfólio |
|---------|--------|-----------|------------------|-------------------|
| **Faithfulness (RAG Triad)** | ✅ (2026-07-02) | Juiz local (`evals/eval_faithfulness.py`) | Implementado com juiz direto via OllamaClient (`format=json`) em vez do `FaithfulnessEvaluator` do Phoenix (que espera `gpt-4o`) — zero-cloud, padrão in-house dos evals. Contexto = `body` das tool calls (capturado na trace, truncado 12k). **Medido: 39/40 = 97.5% PASS (gate 90%)**; 1 INFIEL = ruído do juiz. Phoenix segue como visualização opcional. | Governança de IA: o modelo não pode alucinar sobre dados corporativos |
| **Jailbreak Resistance** | ✅ (já existe!) | Langfuse | `routing_failure_rate` quando tem injection. Langfuse filtra traces com `injection=true`. Métrica nova: `injection_block_rate` (quantos prompts com payload malicioso foram corretamente roteados sem vazar o domínio errado). | Security-first: já pensa em proteção desde o design |

**O que já existe**: Injection resistance está implementado no router e nos 320 exemplos de injection do dataset v2. Faithfulness é o próximo passo.

---

## 4. Sinergia Humano-IA (Experiência)

| Métrica | Status | Framework | Como implementar | Sinal de portfólio |
|---------|--------|-----------|------------------|-------------------|
| **Human Takeover Rate** | 🔴 | LangSmith (documentado) | Não implementável sem interface de usuário real. Documentável no portfólio: quando `clarification != null` no RoutePlan, é um "takeover" implícito (o sistema pede mais informação em vez de adivinhar). Métrica: `clarification_rate`. | Sabe que IA não substitui humano — complementa |
| **Correction Frequency** | 🟡 | LangSmith (human annotation) | LangSmith tem `AnnotationQueue` para feedback humano. Implementável como endpoint `/feedback` que recebe correções do usuário e envia para LangSmith via SDK. Documentável como arquitetura, mesmo sem frontend real. | Fecha o loop: feedback → retrain → melhora |

---

## Resumo executivo

| Pilar | Métricas implementáveis agora | Nível de esforço |
|-------|------------------------------|------------------|
| Negócio | 1 de 2 (Cost-per-Task) | Baixo (token counting já existe) |
| Engenharia | 3 de 3 (todas) | Médio (adicionar 2-3 métricas ao eval) |
| Governança | 2 de 2 (Jailbreak já existe, Faithfulness novo) | Médio (Phoenix eval pipeline) |
| Humano-IA | 1 de 2 (Clarification rate) | Baixo (já existe no RoutePlan) |

### O que implementar agora

1. **Adicionar LangSmith** ao projeto (3 env vars + `langsmith` SDK)
2. **Adicionar Phoenix** ao `docker-compose.yml` (1 container)
3. **Adicionar OpenTelemetry** SDK + Collector como camada única de instrumentação
4. **Novas métricas** no `eval_semiose.py`:
   - `cost_per_task` (R$ estimado por inferência local)
   - `task_success_rate` (% de agent runs que terminam com `stop_reason=answer`)
   - `tool_call_efficiency` (tools por task, P50/P95)
   - `injection_block_rate` (injection prompts corretamente roteados)
   - `clarification_rate` (takeover implícito)
5. **Documentar** RAOI e Correction Frequency como arquitetura pronta para produção

### O que NÃO implementar (e por quê)

| Métrica | Razão |
|---------|-------|
| RAOI automatizado | Precisa de dados reais de operação (horas salvas, custo hora). Documenta a fórmula. |
| Human Takeover Rate real | Precisa de UI com botão de takeover. Já temos `clarification_rate` como proxy. |
| Correction Frequency real | Precisa de annotation loop com humanos reais. Documenta o pipeline. |
