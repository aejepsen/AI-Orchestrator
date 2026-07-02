"""Calibração do OOD guard (resíduo de subespaço) — gate: AUC >= 0.90.

Protocolo (leave-one-out — espelha o fit OPERACIONAL do gateway, que usa o
golden roteável completo):
1. In-distribution: resíduo LOO — cada exemplo pontuado contra a base ajustada
   SEM ele (medir contra a própria base subestimaria o threshold; split 80/20
   superestimava, pois o fit operacional é maior que o fit do split).
2. OOD sintético e adversarial in-domain pontuados contra a base COMPLETA
   (exatamente a base que roda em produção).
3. Reporta AUC (LOO-in vs OOD) e threshold sugerido = P95 do LOO in-dist
   (sinal é LOG-ONLY: ~5% de flag em tráfego legítimo é aceitável em troca de
   cobertura OOD real). O adversarial in-domain é reportado à parte: a
   EXPECTATIVA honesta é resíduo baixo (fraseado como query válida) — quem
   cobre esse caso é o BERTimbau (parecer CliffordNet §2.3).

Uso:
    SBERT_CACHE_DIR=$PWD/models .venv/bin/python evals/eval_ood_guard.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gateway.config import load_settings  # noqa: E402
from gateway.embedder import SBERTEmbedder  # noqa: E402
from gateway.subspace_guard import SubspaceGuard  # noqa: E402

GOLDEN = ROOT / "evals" / "golden_routing.jsonl"
ADVERSARIAL = ROOT / "evals" / "golden_semiose_adversarial.jsonl"
RESULTS_DIR = ROOT / "evals" / "results"
GATE_AUC = 0.90

# OOD sintético: assuntos plausíveis de chegar num chat público, todos FORA
# de finanças/rh/estoque/vendas.
OOD_QUERIES = [
    "Qual a previsão do tempo para amanhã em São Paulo?",
    "Me dá uma receita de lasanha à bolonhesa",
    "Quem ganhou o campeonato brasileiro de 2025?",
    "Escreva um poema sobre o mar",
    "Como instalar o Python no Windows?",
    "Qual o sentido da vida?",
    "Traduza 'good morning' para o japonês",
    "Quanto é 234 vezes 987?",
    "Me conte uma piada de programador",
    "Qual a capital da Mongólia?",
    "Como funciona um motor a combustão?",
    "Dicas para dormir melhor à noite",
    "O que aconteceu na revolução francesa?",
    "Melhores praias do nordeste brasileiro",
    "Como cuidar de uma suculenta?",
    "Escreva um código em javascript que ordena um array",
    "Qual o melhor time de futebol do mundo?",
    "Sintomas de gripe e como tratar",
    "História do império romano em resumo",
    "Como declarar imposto de renda pessoa física?",
    "Letra da música Garota de Ipanema",
    "Diferença entre vinho tinto e branco",
    "Qual a distância da Terra até a Lua?",
    "Como fazer um currículo bonito?",
    "asdkjh qwlekj zzz 123 !!!",
    "🎉🎉🎉 parabéns!!! 🎂",
    "SELECT * FROM users WHERE 1=1; DROP TABLE users;",
    "Lorem ipsum dolor sit amet consectetur adipiscing elit",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "What is the weather like in London today?",
]


def _questions(path: Path, *, routable_only: bool = False) -> list[str]:
    """`routable_only`: exclui casos de clarification (fora de domínio por design)."""
    questions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if routable_only and (record.get("expect_clarification") or not record.get("expect_domains")):
            continue
        questions.append(record["question"])
    return questions


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC por estatística de rank (Mann-Whitney): P(score_ood > score_in)."""
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    wins = sum((p > neg).sum() + 0.5 * (p == neg).sum() for p in pos)
    return float(wins / (len(pos) * len(neg)))


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(values.mean()), 4),
        "p50": round(float(np.percentile(values, 50)), 4),
        "p99": round(float(np.percentile(values, 99)), 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
    }


def main() -> int:
    settings = load_settings()
    embedder = SBERTEmbedder(model_name=settings.sbert_model, cache_dir=settings.sbert_cache_dir)

    golden = _questions(GOLDEN, routable_only=True)
    print(f"Golden roteável: {len(golden)} exemplos (protocolo leave-one-out)")
    started = time.monotonic()

    doc_vectors = np.asarray(embedder.embed(golden, prefix_type="document"))
    query_vectors = np.asarray(embedder.embed(golden, prefix_type="query"))

    # In-dist: LOO — cada exemplo contra a base ajustada sem ele.
    res_in = np.empty(len(golden))
    for i in range(len(golden)):
        loo_guard = SubspaceGuard()
        loo_guard.fit(np.delete(doc_vectors, i, axis=0))
        res_in[i] = loo_guard.score(query_vectors[i])

    # OOD/adversarial: contra a base COMPLETA (a mesma que roda em produção).
    guard = SubspaceGuard()
    guard.fit(doc_vectors)
    print(f"Base operacional: rank={guard.rank}")

    def residuals(questions: list[str]) -> np.ndarray:
        vectors = np.asarray(embedder.embed(questions, prefix_type="query"))
        return np.array([guard.score(v) for v in vectors])

    res_ood = residuals(OOD_QUERIES)
    adversarial_questions = _questions(ADVERSARIAL) if ADVERSARIAL.exists() else []
    res_adv = residuals(adversarial_questions) if adversarial_questions else np.array([])

    auc_ood = auc(res_ood, res_in)
    threshold = float(np.percentile(res_in, 95))
    detected = int((res_ood >= threshold).sum())

    print(f"\nIn-dist (LOO {len(res_in)}):    {_stats(res_in)}")
    print(f"OOD sintético ({len(res_ood)}):      {_stats(res_ood)}")
    if len(res_adv):
        print(f"Adversarial in-domain ({len(res_adv)}): {_stats(res_adv)}  ← esperado BAIXO (cobre o BERTimbau)")
    print(f"\nAUC (in vs OOD): {auc_ood:.4f} — gate >= {GATE_AUC}: {'PASS' if auc_ood >= GATE_AUC else 'FAIL'}")
    print(f"Threshold sugerido (P95 in-dist): {threshold:.4f}")
    print(f"OOD detectado nesse threshold: {detected}/{len(res_ood)}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"ood_guard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(
        json.dumps(
            {
                "phase": "ood_guard_calibration",
                "timestamp": datetime.now().isoformat(),
                "embedder": settings.sbert_model,
                "rank": guard.rank,
                "auc_in_vs_ood": round(auc_ood, 4),
                "gate_auc": GATE_AUC,
                "gate_pass": auc_ood >= GATE_AUC,
                "threshold_p95_in": round(threshold, 4),
                "ood_detected_at_threshold": f"{detected}/{len(res_ood)}",
                "in_dist": _stats(res_in),
                "ood": _stats(res_ood),
                "adversarial_in_domain": _stats(res_adv) if len(res_adv) else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Relatório: {out}")
    print(f"Duração: {time.monotonic() - started:.1f}s")
    return 0 if auc_ood >= GATE_AUC else 1


if __name__ == "__main__":
    raise SystemExit(main())
