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
| **nomic-embed-text** (274 MB) | Embeddings p/ rota semântica (Qdrant) | **VRAM da GPU** | ~0.1 s/query |
| **FastAPI Microservices** | APIs de Finanças, RH, Estoque e Vendas | **CPU (Ryzen 5 5600)** | Execução instantânea e determinística baseada em regras puras |
| **LangGraph Orchestrator** | Gerenciamento de Estado e Ciclo do Grafo | **CPU (Ryzen 5 5600)** | Orquestração assíncrona orientada a eventos |

> **Evolução:** arquitetura original usava Qwen 3 30B-A3B (MoE, 18 GB, split GPU+CPU ~15 s/task) com 7B como contingência. Fine-tune LoRA (3.050 exemplos, Unsloth bf16, Colab A100) do Qwen 3.5-9B produziu modelo especializado que roda 100% GPU com qualidade equivalente e latência 5–7x menor. Detalhes em `docs/PLANO_LORA_9B.md`.

---

## 3. Topologia do Grafo (AI Gateway Pattern)