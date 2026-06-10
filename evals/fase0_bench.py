"""Fase 0 — gate de viabilidade do AI-Orchestrator.

Mede, contra o Ollama local (porta 11434):
1. tok/s de geração do orquestrador MoE (qwen3:30b-a3b) com split GPU/RAM automático.
2. Tempo de swap orquestrador <-> subagente (qwen2.5:7b-instruct-q4_K_M),
   simulando o handoff do grafo com keep_alive=0.

Gate (PLANO_EXECUCAO.md): orquestrador >= 5 tok/s E swap <= 5 s.

Uso (AI-Tractor parado, só o container ollama up):
    python3 evals/fase0_bench.py
"""

import json
import statistics
import time
import urllib.request

OLLAMA = "http://localhost:11435"  # container exclusivo ai-orchestrator-ollama
MOE = "qwen3:30b-a3b"
SUB = "qwen2.5:7b-instruct-q4_K_M"

GATE_TOKS = 5.0
GATE_SWAP_S = 5.0

PROMPTS = [
    "Classifique a intenção e liste os domínios envolvidos (financas, rh, estoque, vendas), "
    "respondendo só JSON: 'Posso aceitar um pedido de 500 unidades do SKU A12 com 15% de desconto?'",
    "Explique em 3 frases o trade-off entre consistência e disponibilidade em microsserviços.",
    "Um cliente pediu reembolso de viagem de R$ 2.340. Qual domínio trata isso e que dados faltam? Responda curto.",
]


def generate(model: str, prompt: str, keep_alive: str = "0", num_predict: int = 200) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": keep_alive,
        "options": {"num_predict": num_predict, "temperature": 0.3},
    }
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as resp:
        data = json.loads(resp.read())
    data["wall_s"] = time.perf_counter() - t0
    return data


def report(label: str, r: dict) -> tuple[float, float]:
    toks = r.get("eval_count", 0) / max(r.get("eval_duration", 1), 1) * 1e9
    load_s = r.get("load_duration", 0) / 1e9
    print(
        f"  {label}: load {load_s:6.2f}s | prompt_eval {r.get('prompt_eval_count', 0):4d} tok "
        f"| gen {r.get('eval_count', 0):4d} tok @ {toks:5.2f} tok/s | wall {r['wall_s']:6.1f}s"
    )
    return toks, load_s


def main() -> None:
    print(f"== Fase 0 bench ==\nMoE: {MOE}\nSub: {SUB}\n")

    print("[1] MoE — cold load + 3 gerações (keep_alive 5m durante a série)")
    rates = []
    for i, p in enumerate(PROMPTS):
        r = generate(MOE, p, keep_alive="5m")
        toks, load_s = report(f"run {i} ({'cold' if i == 0 else 'warm'})", r)
        if i > 0:  # warm runs only
            rates.append(toks)
    moe_toks = statistics.mean(rates)

    print("\n[2] Swap MoE -> 7B (handoff gateway->subagente)")
    generate(MOE, "ok", keep_alive="0", num_predict=5)  # descarrega MoE
    r = generate(SUB, PROMPTS[0], keep_alive="0", num_predict=120)
    _, swap_to_sub = report("7B cold após MoE", r)

    print("\n[3] Swap 7B -> MoE (volta pra síntese)")
    r = generate(MOE, "Resuma em 1 frase: pedido aprovado com ressalva de estoque.", keep_alive="0", num_predict=60)
    _, swap_to_moe = report("MoE cold após 7B", r)

    swap_worst = max(swap_to_sub, swap_to_moe)
    print("\n== Resultado ==")
    print(f"MoE geração (warm): {moe_toks:.2f} tok/s  (gate >= {GATE_TOKS})")
    print(f"Swap pior caso:     {swap_worst:.2f} s    (gate <= {GATE_SWAP_S})")
    ok = moe_toks >= GATE_TOKS and swap_worst <= GATE_SWAP_S
    print(f"Gate Fase 0: {'PASS — arquitetura da proposta' if ok else 'FAIL — avaliar fallback 7B-único'}")


if __name__ == "__main__":
    main()
