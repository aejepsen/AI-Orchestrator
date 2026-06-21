# Avaliação das Métricas Propostas + Métricas para Semiose

## Veredito do framework original (4 categorias)

Sólido como taxonomia NLP genérica. Problema: assume pipeline generativo (texto→texto). Semiose é pipeline de **transformação determinística** (query→signals→re-rank→routing). A avaliação corretamente identifica esse gap e adapta.

## Análise por camada

### Camada A — Query Enricher

**Contextual Drift Score** — excelente métrica. Captura o risco real: enricher puxar embedding pra longe do significado original. Threshold `< 0.10` conservador e correto pro nosso caso (prefixo curto, não reescrita).

**Entity Propagation F1** — métrica certa pro problema certo. Multi-turn é onde enricher agrega valor. Observação: precisa definir golden set de propagação (pares de turnos onde entidade X deveria propagar). Sugestão: derivar do `golden_routing.jsonl` existente adicionando pares multi-turn.

**Token-Level Boundary F1 para regex** — útil mas baixa prioridade. Regex patterns são determinísticos e testáveis com unit tests (já cobertos nos 120 testes). F1 sobre spans faz mais sentido pra spaCy NER onde boundary é probabilístico.

**Métrica ausente — False Enrichment Rate (FER):**

```
FER = queries onde enricher adicionou contexto E roteamento piorou
    / total de queries onde enricher adicionou contexto
```

Captura o caso mais perigoso: enricher **ativo mas prejudicial**. Drift Score mede deslocamento vetorial mas não mede impacto no roteamento. FER conecta causa (enrichment) ao efeito (routing accuracy). FER > 0.05 = alarme.

### Camada B — Knowledge Graph

**Graph Expansion Utility (GEU)** — métrica correta e prática. Mede utilidade real (informação apareceu na resposta). GEU < 0.5 = tool gera ruído — concordo.

**Relation Validity@5** — implementada como _non-garbage rate_: fração das relações retornadas (top-5 por chamada) cujo domínio pertence ao conjunto conhecido (`estoque`, `vendas`, `financas`, `rh`). Não é precisão contra um golden de relações (um golden derivado do seed script é trabalho futuro). Target ≥ 0.80 adequado para o proxy de validade.

**Métrica ausente — Graph Latency Budget:**

```
Graph Latency Budget = p95 latency com expand_context / p95 latency sem expand_context
```

Target: < 1.3 (máximo 30% overhead). KG é opt-in justamente porque Neo4j adiciona latência. Se `expand_context` dobrar o tempo de resposta, valor não justifica. Langfuse já captura traces — métrica derivável sem instrumentação nova.

**Métrica ausente — Cross-Domain Resolution Rate:**

```
CDRR = queries onde expand_context trouxe entidade de OUTRO domínio que apareceu na resposta
     / total de chamadas expand_context
```

É o GEU filtrado por cross-domain. Se CDRR ≈ 0, o KG não está agregando nada que o agente do domínio já não soubesse. Essa é a razão de existir do KG — se não resolve cross-domain, não vale o custo.

### Camada C — Re-ranking Contextual

**Contextual Gain Ratio (CGR)** — formulação elegante. Normaliza pelo erro residual, isola contribuição real. `CGR ≥ 0.30` = 30% do erro residual corrigido — ambicioso mas alcançável com golden bem calibrado.

**Métrica ausente — Boost Precision:**

```
Boost Precision = queries onde boost aditivo mudou top-1 E a mudança foi correta
               / total de queries onde boost mudou top-1
```

O boost `min(score + 0.05, 1.0)` é conservador por design. Mas quando ele muda o ranking (top-1 flip), precisa estar certo. Se Boost Precision < 0.90, o threshold de 0.05 precisa ser ajustado ou o mecanismo de topic-switch (`_has_strong_conflict`) está falhando.

### End-to-end

**Exact-Match Routing (+3pp)** — igualdade exata do conjunto de domínios previstos vs. esperados (não micro-F1 sobre labels individuais; nomeado assim para refletir o cálculo real). Target mínimo razoável. Baseline já deve estar alto (semantic router + lexical fallback); 3pp sobre baseline alto é significativo.

**Enrichment Cosine Preservation** — cosseno SBERT entre query original e enriquecida (não BERTScore token-level; nomeado assim para refletir a implementação). É o complemento do Contextual Drift Score (preservation ≈ 1 − drift). Sanity check: valor baixo indica que o enriquecimento distorce o significado. BERTScore real fica como trabalho futuro (caro em CPU, rodar offline em batch).

**Métrica ausente — Topic Switch Accuracy:**

```
TSA = queries com mudança de domínio onde enricher corretamente NÃO propagou contexto anterior
    / total de queries com mudança de domínio
```

`_has_strong_conflict()` é a defesa contra propagação indevida. Se TSA < 0.95, contexto está vazando entre turnos de domínios diferentes — o pior failure mode da Semiose.

## Tabela-resumo consolidada

| Camada | Métrica | Baseline | Alvo | Prioridade |
|--------|---------|----------|------|------------|
| A | Entity Propagation F1 | 0.0 | ≥ 0.70 | Alta |
| A | Contextual Drift Score | — | < 0.10 | Alta |
| A | **False Enrichment Rate** | — | < 0.05 | **Crítica** |
| A | **Topic Switch Accuracy** | — | ≥ 0.95 | **Crítica** |
| B | Graph Expansion Utility | 0.0 | ≥ 0.60 | Alta |
| B | Relation Validity@5 | — | ≥ 0.80 | Média |
| B | **Cross-Domain Resolution Rate** | — | ≥ 0.40 | Alta |
| B | **Graph Latency Budget** | 1.0 | < 1.30 | Média |
| C | Contextual Gain Ratio | 0.0 | ≥ 0.30 | Alta |
| C | **Boost Precision** | — | ≥ 0.90 | Alta |
| E2E | Exact-Match Routing | Baseline eval | +3pp | Alta |
| E2E | Enrichment Cosine Preservation | 0 | alto (≈1−drift) | Média |

## Observações arquiteturais

1. **Instrumentação**: 8 das 12 métricas são deriváveis dos traces Langfuse existentes (já captura tool calls, scores, latency). Não precisa de infraestrutura nova.

2. **Golden set multi-turn**: Falta. O `golden_routing.jsonl` atual é single-turn. Para Entity Propagation F1, TSA e CGR, precisa de pares `(turno_anterior, turno_atual, domínio_esperado, entidades_esperadas)`. Proposta: criar `golden_semiose.jsonl` com ~30 pares.

3. **Ordem de implementação das métricas**: FER e TSA primeiro (detectam danos). Depois CGR e GEU (medem valor). Por último Enrichment Cosine Preservation (sanity check; BERTScore real fica para depois).

4. **LLM-as-a-Judge**: Proposta no texto original é válida mas cara (requer SOTA externo). Para PoC, CGR + FER + TSA cobrem o mesmo ground de forma determinística e reprodutível. Reservar LLM-as-Judge para validação pré-produção.
