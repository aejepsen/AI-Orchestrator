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

## Status (2026-06-12)

- **Fase 1 — EM EXECUÇÃO no Colab A100** (notebook `train/colab_generate_dataset.ipynb` v3; projeto baixado de suasalada.com.br/ai-orchestrator.zip).
  - `questions`: **3000/3000 ✓** (1431 domain + 1569 routing, 0 duplicatas exatas).
  - `trajectories`: em andamento — ~15,4s/item, ETA ~5h45 a partir de 90/1431. Rejeição ~4% (`rejected.jsonl` auditável; motivos: sem tool call, alucinação numérica, tool/status errado).
  - Auditoria parcial (26 trajetórias): 26/26 com tool call, 0 alucinação numérica; amostra exemplar (recusa fundamentada por limite de R$ 3.000 do serviço RH).
  - Incidente: sessão Colab encerrada após `questions` (2026-06-11 18:47) — confundiu "3000/3000" do estágio 1 com fim do pipeline. Retomado por hash após restore do backup do Drive (`MyDrive/ai-orchestrator-dataset/`). Lição: célula 6 (monitor + backup incremental) deve rodar a cada 1–2h.
  - Gotchas Colab: instalador do Ollama exige `zstd` (`apt-get install zstd` antes); `think=False` na geração de perguntas; `num_ctx=4096` no tool-loop.
- **Fase 2 — notebook pronto**: `train/colab_train_lora.ipynb` (Unsloth, transformers v5 pinado, LoRA bf16 r=16/alpha=32, export GGUF Q4_K_M + Modelfile). Falta subir ao Drive e executar.
- **Créditos Colab**: 83,82 unidades restantes (de 87,45); projeção: ~30 p/ terminar Fase 1 + 16–21 p/ treino → margem p/ re-runs.
- **Fases 3–5**: pendentes.

## Riscos

- 22 GB em L4 24 GB é justo → mitigar com grad checkpointing + max_seq 4096; fallback A100.
- Dataset destilado do 30b herda erros dele (~7% routing) → filtro automático + revisão das amostras de val.
- Qwen3.5 exige transformers v5 no Colab (pinar versões no notebook).
