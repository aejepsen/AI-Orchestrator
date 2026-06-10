# PROPOSAL.md — PoC: AI Gateway & Multi-Agent Microservices Architecture

## 1. Visão Geral da Arquitetura
Esta Prova de Conceito (PoC) implementa o padrão **AI Gateway (Experience Layer)** operando sobre um ecossistema de **Subagentes Especialistas (Process Layer)**. O princípio fundamental do projeto é o desacoplamento entre o raciocínio cognitivo da IA e as regras de negócio corporativas: a IA atua puramente na orquestração e intenção, enquanto todo o cálculo, validação e persistência de dados ocorrem em microsserviços determinísticos via APIs RESTful.

O ecossistema é projetado para rodar **100% On-Premise**, garantindo custo zero de inferência e privacidade absoluta de dados sensíveis corporativos.

---

## 2. Hardware de Referência & Alocação de Memória

A topologia de execução distribui os modelos aproveitando a arquitetura híbrida de *Mixture of Experts* (MoE) e o paralelismo do hardware local:

| Componente | Papel Arquitetural | Alocação de Hardware | Perfil de Performance |
|-----------|--------------------|----------------------|-----------------------|
| **Qwen 3.6 35B-A3B** (Q4_K_M) | AI Gateway / Orquestrador Central | **RAM do Sistema** (~20 GB ocupados) | ~3 a 5 tokens/s (Baixa geração de texto, alto raciocínio lógico/roteamento) |
| **Qwen 2.5 7B Instruct** (Q4_K_M) | Subagentes Especialistas de Domínio | **VRAM da GPU** (~4.8 GB ocupados) | ~35+ tokens/s (Resposta rápida para chamadas de ferramentas e geração de código) |
| **FastAPI Microservices** | APIs de Finanças, RH, Estoque e Vendas | **CPU (Ryzen 5 5600)** | Execução instantânea e determinística baseada em regras puras |
| **LangGraph Orchestrator** | Gerenciamento de Estado e Ciclo do Grafo | **CPU (Ryzen 5 5600)** | Orquestração assíncrona orientada a eventos |

---

## 3. Topologia do Grafo (AI Gateway Pattern)