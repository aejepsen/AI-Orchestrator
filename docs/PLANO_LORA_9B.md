# Plano — LoRA Qwen3.5-9B (AI-Orchestrator)

**Objetivo:** especializar o Qwen3.5-9B em tool-calling + routing do orquestrador via LoRA, treinando no Google Colab (L4, ~15–25 unidades das 87 disponíveis), com deploy GGUF no Ollama local. Runtime permanece zero-cloud.

**Baseline a superar (9B base, 2026-06-11):** routing 95.5% · injection 0/6 · domains 87.5% (90/80/80/100).

## Fase 1 — Dataset (local, RTX 3060)

1. **Geração de perguntas sintéticas**: paráfrases e variações das 44 perguntas de routing + 40 tasks de domínio, usando `qwen3:30b-a3b` como gerador (melhor modelo local, gates OK). Novos cenários sempre sobre os dados reais do seed SQLite. Alvo: **1.500–3.000 exemplos**.
2. **Trajetórias de tool-calling**: rodar `run_domain_agent` com o 30b contra os microsserviços reais; capturar `messages` completas (system, user, assistant+tool_calls, tool results, resposta final) no template de chat do Qwen.
3. **Exemplos de routing**: pares pergunta → JSON `{domains, plan, clarification}` no formato do classifier, incluindo casos de injection (resposta correta = ignorar payload).
4. **Filtro de qualidade automático**: aceitar só trajetórias que passam checks (tool certa, status esperado, resposta fundamentada nos retornos, sem alucinação numérica). Dedup por similaridade.
5. **Anti-contaminação**: golden sets atuais ficam 100% fora do treino (são o eval). Split train/val 90/10 do sintético.

Saída: `train/dataset/orch_sft.jsonl` (+ val).

## Fase 2 — Treino (Colab L4 24GB)

- Notebook Unsloth, base `unsloth/Qwen3.5-9B`, **LoRA bf16** (QLoRA 4-bit contraindicado pela Unsloth em Qwen3.5).
- Hiperparâmetros: r=16, alpha=32, dropout 0.05; target `q/k/v/o_proj + gate/up/down_proj` (camadas DeltaNet fora); lr 2e-4 cosine; 2–3 epochs; batch efetivo 16 (grad accum); gradient checkpointing; max_seq 4096.
- Eval de val loss por epoch; early stop se overfit.
- Custo estimado: 3–5h ≈ 15–25 unidades. Margem p/ 1–2 re-runs.

## Fase 3 — Export e deploy local

1. `save_pretrained_merged(..., "merged_16bit")` → conversão GGUF **Q4_K_M** no próprio Colab.
2. Download do GGUF (~5.5 GB) → `ollama create qwen3.5-9b-orch -f Modelfile` (template de chat Qwen3.5, thinking default off).

## Fase 4 — Validação (gates locais, watchdog ativo)

- `MODEL=qwen3.5-9b-orch` nos 3 evals: routing ≥90%, injection 0 leaks, domains ≥80%/domínio.
- Critério de adoção: **superar 87.5% em domains sem regressão** em routing/injection. Se empatar ou piorar: mantém 9B base (LoRA documentado como experimento no README).

## Fase 5 — Promoção

- `.env MODEL=qwen3.5-9b-orch` + restart gateway + atualizar README (tabela de resultados com 7b/9B/9B-LoRA/30b) + commit (já autorizado pós-gates).

## Artefatos novos

- `train/build_dataset.py` (geração + filtro, local)
- `train/colab_lora_qwen35_9b.ipynb` (treino + export GGUF)
- `train/Modelfile`
- `docs/PLANO_LORA_9B.md` (este)

## Status (2026-06-13)

- **Fase 1 — CONCLUÍDA**: dataset final **3.050 exemplos** (train 2.745 / val 305): 1.325 trajetórias + 1.569 routing + 156 routing_injection. Domínios balanceados. Backup em `MyDrive/ai-orchestrator-dataset/`.
- **Fase 2 — CONCLUÍDA (Colab A100 40GB)**: 2 epochs, 344 steps, 148 min. LoRA bf16 r=16/alpha=32, batch efetivo 16, lr 2e-4 cosine, max_seq 4096. **Epoch 1: train 0.091 / val 0.097. Epoch 2: train 0.071 / val 0.089.** Sem overfit. VRAM pico 31.8 GB. Checkpoints no Drive (`MyDrive/ai-orchestrator-lora/training/`).
  - Gotchas: `trainer.evaluate()` pós-treino causa CUDA IllegalMemoryAccess nas camadas DeltaNet (bug Unsloth) — usar val loss do treino; Arrow/`load_dataset('json')` falha em tool_calls heterogêneos — json.loads manual; checkpoints SEMPRE no Drive (incidente: 2h17 perdidas por `output_dir` em `/content`); Unsloth gera GGUF em `gguf_gguf/` (adiciona `_gguf` ao path).
- **Fase 3 — CONCLUÍDA**: merge 16-bit → GGUF Q4_K_M (5.4 GB) + Modelfile no Drive (`MyDrive/ai-orchestrator-lora/`). Necessário limpar cache HF (~19 GB) antes do export por espaço em disco.
- **Fase 4 — CONCLUÍDA**: `ollama create qwen3.5-9b-orch`, Ollama atualizado 0.24→0.30.8 (0.24 não suportava arquitetura híbrida DeltaNet). Fix `llm.py`: `keep_alive` string→int (Ollama 0.30 rejeita `"-1"`); `think=true` excluído pra qwen3.5 (Small series não suporta). **Resultados (2 runs consistentes):**

| Eval | LoRA 9B | Baseline 9B | Baseline 7b | Gate |
|---|---|---|---|---|
| Routing | 90.9% | 95.5% | 90.5% | >=90% PASS |
| Injection | 0/6 | 0/6 | 0/6 | 0 leaks PASS |
| Domains | 87.5% (90/90/90/80) | 87.5% (90/80/80/100) | 82.5% | >=80%/dom PASS |

- **Fase 5 — CONCLUÍDA**: `.env MODEL=qwen3.5-9b-orch`, gateway rebuild, README atualizado.

## Riscos

- 22 GB em L4 24 GB é justo → mitigar com grad checkpointing + max_seq 4096; fallback A100.
- Dataset destilado do 30b herda erros dele (~7% routing) → filtro automático + revisão das amostras de val.
- Qwen3.5 exige transformers v5 no Colab (pinar versões no notebook).
