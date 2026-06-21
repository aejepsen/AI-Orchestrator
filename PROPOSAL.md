# PROPOSAL.md — PoC: AI Gateway & Multi-Agent Microservices Architecture

## 1. Visão Geral da Arquitetura
Esta Prova de Conceito (PoC) implementa o padrão **AI Gateway (Experience Layer)** operando sobre um ecossistema de **Subagentes Especialistas (Process Layer)**. O princípio fundamental do projeto é o desacoplamento entre o raciocínio cognitivo da IA e as regras de negócio corporativas: a IA atua puramente na orquestração e intenção, enquanto todo o cálculo, validação e persistência de dados ocorrem em microsserviços determinísticos via APIs RESTful.

O ecossistema é projetado para rodar **100% On-Premise**, garantindo custo zero de inferência e privacidade absoluta de dados sensíveis corporativos.

---

## 2. Hardware de Referência & Alocação de Memória

A topologia de execução distribui os modelos aproveitando a arquitetura híbrida de *Mixture of Experts* (MoE) e o paralelismo do hardware local:

| Componente | Papel Arquitetural | Alocação de Hardware | Perfil de Performance |
|-----------|--------------------|----------------------|-----------------------|
| **Qwen 3.5-9B LoRA** (Q4_K_M, 5.4 GB) | AI Gateway + Subagentes (modelo único) | **VRAM da GPU** (100% GPU, 5.4 de 12 GB) | ~2–4 s/task, routing 90.9%, domains 87.5% |
| **SBERT** `paraphrase-multilingual-MiniLM-L12-v2` (384d, ~120 MB) | Embeddings p/ rota semântica (Qdrant) | **CPU** (libera a GPU) | ~0.02 s/query |
| **FastAPI Microservices** | APIs de Finanças, RH, Estoque e Vendas | **CPU (Ryzen 5 5600)** | Execução instantânea e determinística baseada em regras puras |
| **LangGraph Orchestrator** | Gerenciamento de Estado e Ciclo do Grafo | **CPU (Ryzen 5 5600)** | Orquestração assíncrona orientada a eventos |

> **Evolução:** arquitetura original usava Qwen 3 30B-A3B (MoE, 18 GB, split GPU+CPU ~15 s/task) com 7B como contingência. Fine-tune LoRA (3.050 exemplos, Unsloth bf16, Colab A100) do Qwen 3.5-9B produziu modelo especializado que roda 100% GPU com qualidade equivalente e latência 5–7x menor. Detalhes em `docs/PLANO_LORA_9B.md`.

---

## 3. Topologia do Grafo (AI Gateway Pattern)

O orquestrador é um **StateGraph LangGraph** (`gateway/graph.py`) com transições explícitas, observáveis e testadas. Cada request `POST /chat` percorre:

```
sanitize → enrich → classify → ┬─ respond_clarification → END
                               └─ dispatch (fan-out) → synthesize → END
```

- **sanitize** — boundary anti-injection (strip ChatML + 14 regex PT/EN + classificador BERTimbau).
- **enrich** *(Semiose Camada A)* — reconstrói a query com contexto estruturado (domínio do turno anterior + entidades via regex/spaCy), sem LLM; realimenta opcionalmente com vizinhos do Knowledge Graph (`KG_ENRICH_ENABLED`).
- **classify** — três camadas em cascata: semântico (Qdrant kNN + boost contextual + desempate por cross-encoder, *Semiose Camada C*) → LLM (RoutePlan JSON validado por Pydantic, com decomposição multi-domínio) → fallback léxico determinístico. Pergunta fora de escopo vira `clarification` (rota válida).
- **dispatch** — fan-out paralelo (ThreadPoolExecutor) por domínio; cada subagente roda um loop de tool-calling least-privilege contra seu microserviço, podendo consultar o Knowledge Graph (*Semiose Camada B*, tool virtual `expand_context`).
- **synthesize** — consolida as respostas dos agentes em uma resposta única (passthrough com 1 domínio).

Progresso transmitido por **SSE** em tempo real: `route` → `agent` (um por subagente concluído) → `final`. A camada de compreensão contextual (Semiose, Camadas A/B/C) é descrita em `PLANO_SEMIOSE.md` e ilustrada em `docs/semiose-flow.png`.

---

## 4. Referências de implementação

- `PLANO_EXECUCAO.md` — plano por fase com as-built e números medidos (Fases 0–8).
- `PLANO_SEMIOSE.md` — Camadas A/B/C, desvios arquiteturais e resultados de eval.
- `docs/PLANO_LORA_9B.md` — fine-tune LoRA do Qwen 3.5-9B (treino + resultados).
- `README.md` — visão geral, segurança, latências medidas e como rodar.