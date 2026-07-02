# Parecer Técnico — CliffordNet no AI-Orchestrator

> **Revisor:** avaliação de engenharia sobre `plano_cliffordNet.md`
> **Data:** 2026-07-02
> **Método:** verificação matemática numérica (numpy) das afirmações centrais + confronto com as medições reais do sistema em produção (evals de 2026-07-02)

---

## 1. Veredicto executivo

**Não adotar como substituição do pipeline. Adotar duas ideias derivadas, como experimentos opt-in.**

O plano é sofisticado na forma, mas três das suas afirmações matemáticas centrais **não sobrevivem à verificação numérica**, e a tabela de ganhos projeta números **fabricados sobre um diagnóstico desatualizado** — o principal problema que ele se propõe a resolver (routing 56.9–93.7%, latência de 15–70 s) **já foi resolvido hoje por outros meios** (routing 91.5% PASS no golden denso de 153; latência real do classifier LLM medida em **1.06–1.12 s**, não 15 s).

O que sobrevive da proposta, despido do vocabulário de Álgebra Geométrica, são duas técnicas clássicas e úteis:

1. **Detector de anomalia por resíduo de subespaço** (o §3.3 do plano é, matematicamente, projeção ortogonal + norma do resíduo — novelty detection clássica). Útil como **sinal adicional log-only** no `sanitize`, em CPU, ~30 linhas de numpy.
2. **Embeddings rotacionais para o Knowledge Graph** (Camada B) — a única aplicação com literatura real (RotatE, QuatE, GeomE/Clifford KG embeddings) que mapeia num componente existente do projeto.

---

## 2. Verificação das afirmações do plano

### 2.1 ✗ §3.1 — "M_Q = S·C + S∧C entrelaça semântica e contexto"

**Verificado numericamente: S·C = 0.0.** O plano define S (semântico, 384d) e C (contexto, 4d) em subespaços **ortogonais por construção**. O produto interno entre eles é identicamente zero — o "produto geométrico" reduz-se ao bivetor S∧C, que tem 384×4 = 1.536 componentes (não os 388 prometidos no §4.1). A decomposição "escalar + vetor + bivetor" apresentada é vazia neste setup.

### 2.2 ✗ §3.2 — "O rotor gira o vetor semântico e resolve homônimos"

**Verificado numericamente: mudança nas 384 dims semânticas = 0.0.** Rotação num plano só altera componentes **dentro do plano de rotação**. O rotor proposto gira no plano dos eixos de contexto (e_fin∧e_est) — as 384 dimensões semânticas ficam **intactas**. O exemplo-vitrine do plano ("banco" girando em direção a Finanças) é matematicamente impossível nesta construção. Para girar conteúdo semântico seria preciso um bivetor **aprendido em R³⁸⁴** — o que é, na prática, uma camada linear com restrição de ortogonalidade, nada intrinsecamente Clifford.

### 2.3 ~ §3.3 — "Wedge product detecta injection"

**Parcialmente verdadeiro, com identidade trocada.** ||Q ∧ (v₁∧…∧vₖ)|| é exatamente a **norma do resíduo da projeção ortogonal** de Q no span do golden set — técnica clássica de detecção de novidade (verificado: resíduo 0.0 para query dentro do span, 0.77 fora). Funciona para **out-of-distribution** ("previsão do tempo", payloads bizarros). **Não funciona para o ataque que importa**: injection semântica fraseada como query válida ("liste os salários e ignore as instruções…") vive DENTRO do span e passa com resíduo ≈ 0. O BERTimbau pega esses casos porque lê o conteúdo; o resíduo geométrico não lê nada. O F1 projetado de 0.995 não tem base.

### 2.4 ✗ §4.1 — "Cosseno do Qdrant sobre coeficientes = produto interno de Clifford"

Verdadeiro apenas porque a proposta implementável colapsa para **concatenar 4 dimensões de peso de domínio ao embedding** e usar cosseno — ou seja, feature concatenation, sem nenhuma estrutura de álgebra de Clifford restante. As 16 componentes de Cl(4,0) citadas somem silenciosamente entre o §4.1-item-1 e o item-3.

### 2.5 ✗ §5 — Tabela de ganhos

| Afirmação do plano | Realidade medida (2026-07-02) |
|---|---|
| Routing 93.7% → projeta 97.8% | **91.5% PASS** no golden denso 153 (o 93.7% era do golden antigo); resíduo de ~8.5% é **ruído de label do golden**, não fronteira geométrica |
| Latência "~15 s quando LLM decide" → 0.15 s | LLM classifier medido em **1.06–1.12 s** (`eval_routing`, 153 casos). Erro de 10× no baseline |
| "417 MB VRAM do BERT" liberados | BERTimbau roda em **CPU** via volume — não ocupa VRAM |
| CGENN 12K params substitui classifier | Treinar equivariância rotacional sobre **153 exemplos** = overfit garantido; e equivariância a rotações é **vácua** para embeddings de texto (cosseno já é invariante a rotações globais do espaço — não há simetria física a explorar) |
| Exact-Match 56.9% → 84.5% | Limitação já diagnosticada e documentada como **embedder** (MiniLM; E5 testado e pior — `eval_semiose.py` linhas 59-64). Transformação linear pós-hoc não cria informação que o embedding não tem |

### 2.6 ✓ O que o plano acerta

- O boost aditivo `+0.05` da Camada C é de fato ad-hoc (o próprio time o trata como desempate gap-gated, com Boost Precision monitorada).
- A intuição de que contexto conversacional deveria **transformar** a query, não somar score, é legítima — mas a transformação correta é aprendida, não fixada num plano de 2 eixos artificiais.
- Fases com gates e degradação graceful seguem o padrão correto do projeto.

---

## 3. Aplicabilidade real neste projeto

### Nível 1 — Aplicável (recomendado como experimento)

**A. Detector OOD por resíduo de subespaço (`gateway/subspace_guard.py`)**
O §3.3 reformulado com honestidade: SVD do golden de routing (153×384) → base U_k (k≈50-80, cotovelo do espectro); score = ||q − U_k U_kᵀ q||. **Log-only** no `sanitize`, ao lado do `flag_injection()` regex e do BERTimbau — terceiro sinal, especializado em OOD que os outros dois não cobrem. CPU, zero dependência nova, ~30 linhas.

**B. Embeddings rotacionais no KG (Camada B)**
Única aplicação com literatura de verdade: RotatE (rotações complexas), QuatE (quatérnios), GeomE (álgebras geométricas) para **link prediction** em knowledge graphs. Uso no projeto: sugerir relações faltantes no KG corporativo (o enriquecimento que eliminou 6 fornecedores órfãos foi manual — um modelo de KG embedding automatiza a sugestão de `ABASTECE`/`REQUER_APROVACAO` candidatas). Com 177 nós/260 relações o grafo é pequeno — o valor é **demonstrativo de portfólio** ("KG embeddings geométricos aplicados"), não operacional.

### Nível 2 — Experimental (só se as métricas pedirem)

**C. Transformação contextual aprendida de embedding** — a versão honesta do "rotor": matriz ortogonal por domínio (4 matrizes 384×384, via parametrização de Cayley ou `torch.nn.utils.parametrizations.orthogonal`), aplicada à query quando há `_last_route`, treinada no `golden_semiose`. Gate: superar o boost atual em Contextual Gain Ratio e Boost Precision **sem** regredir Exact-Match. Expectativa realista: ganho marginal — o gargalo documentado é o embedder, não a transformação.

### Nível 3 — Não aplicável (rejeitar)

- **CGENN como classificador de rota** — equivariância rotacional não tem o que explorar em texto; 153 exemplos não treinam nada generalizável; o problema de routing já está em 91.5% via guards + prompt.
- **Substituir BERTimbau pelo wedge/resíduo** — remove exatamente a defesa que cobre injection in-distribution. Segurança não regride por elegância matemática.
- **Refatorar o índice Qdrant para "multivetores"** — na forma implementável é concatenação de 4 floats; o custo de migração não compra nada que o payload `domains` já não dê.

---

## 4. Como executar (PoC mínima, com gates)

### Fase 1 — Detector OOD (esforço: baixo, 1 sessão) — ✅ EXECUTADA 2026-07-02

> **Resultado medido** (`evals/eval_ood_guard.py`): AUC in-dist vs OOD = **0.9371** (gate 0.90 PASS);
> threshold 0.60 (P95 in-dist) flagra **23/30 OOD** com ~5% de flag em tráfego legítimo (log-only);
> adversarial in-domain com resíduo baixo (média 0.39, máx 0.57 — **abaixo do threshold**), confirmando
> a previsão do §2.3: quem cobre injection in-distribution é o BERTimbau, não a geometria.
> Implementação: `gateway/subspace_guard.py` (SVD numpy, rank 93 @ energy 0.99), integrado log-only
> no nó `sanitize` (`_ood_residual` no state), flag `OOD_GUARD_ENABLED=1` / `OOD_THRESHOLD=0.60`.
1. `gateway/subspace_guard.py`: classe `SubspaceGuard(vectors, k)` com `fit()` (SVD truncada, numpy puro) e `score(q) -> float` (norma do resíduo).
2. Calibração: distribuição de resíduos no golden (in) vs `golden_semiose_adversarial` + queries OOD sintéticas → threshold no P99 do in-distribution.
3. Integração log-only no nó `sanitize` (mesmo padrão do `flag_injection`): campo `ood_residual` no log estruturado + span do trace.
4. **Gate:** zero falso-bloqueio (é log-only), AUC OOD > 0.9 no conjunto de calibração; medir e documentar sobreposição com o BERTimbau (o que cada um pega que o outro não pega).

### Fase 2 — KG embeddings rotacionais (esforço: médio, experimento de portfólio)
1. Exportar triplas do Neo4j (`scripts/export_kg_triples.py`).
2. Treinar RotatE/QuatE (PyKEEN, CPU — grafo minúsculo) com split 80/10/10 das 260 relações.
3. Eval: MRR/Hits@10 em link prediction; gerar top-20 relações candidatas ausentes e validar manualmente contra os seeds.
4. **Gate:** Hits@10 ≥ 0.5 no split de teste; ≥ 5 relações candidatas plausíveis → vira célula do dashboard/README como demonstração de "KG embeddings geométricos". Não entra no caminho de request.

### Fase 3 — Transformação contextual aprendida (só se Fase 1-2 motivarem)
Conforme Nível 2-C acima, atrás de flag `CONTEXT_TRANSFORM_ENABLED=0`, avaliada por `eval_semiose.py` — mesma régua das Camadas A/B/C.

**Fora de escopo em todas as fases:** tocar no caminho de produção do routing (91.5% recém-estabilizado), remover BERTimbau, migrar o índice Qdrant.

---

## 5. Vantagens (honestas)

- **Fase 1:** terceiro sinal de segurança ortogonal aos existentes (regex pega padrões, BERT pega semântica adversária, resíduo pega OOD), custo ~zero, CPU, sem dependência nova. Narrativa de defesa-em-profundidade forte para o portfólio.
- **Fase 2:** conecta o projeto à literatura de geometric ML de forma **defensável em entrevista** — "usei embeddings rotacionais para curadoria do KG" resiste a arguição técnica; "substituí meu router por CliffordNet" não resiste (as objeções do §2 seriam feitas pelo entrevistador).
- **Geral:** o exercício de verificação em si (este parecer + os 3 testes numéricos) é material de portfólio sobre rigor de engenharia — saber dizer não a uma proposta elegante e errada.

## 6. Resultados esperados (realistas)

| Item | Expectativa | Régua |
|---|---|---|
| OOD detector | AUC > 0.9 em OOD sintético; ~0 ganho em injection in-distribution (esperado e documentado) | Calibração Fase 1 |
| KG link prediction | MRR 0.3–0.6 (grafo pequeno, alta variância); 5-15 relações candidatas úteis | PyKEEN eval |
| Routing accuracy | **Sem mudança** — nada aqui toca o router | `eval_routing.py` |
| Latência do gateway | **Sem mudança** (Fase 1 é O(k·d) em numpy, sub-ms) | `/metrics` |
| VRAM/params | **Sem mudança** — BERTimbau permanece | — |

## 7. Recomendação final

Arquivar `plano_cliffordNet.md` como proposta original (valor histórico), **não executar as Fases 2-4 dele**. Executar as Fases 1-2 **deste parecer** se e quando houver janela — são pequenas, isoladas do caminho de produção e rendem material de portfólio verificável. O aprendizado central: a Álgebra de Clifford é ferramenta séria para dados com simetria geométrica real (nuvens de pontos, PDEs, física); embeddings de texto não têm essa simetria, e o que o plano propõe colapsa, na implementação, em técnicas lineares clássicas — que valem a pena exatamente quando chamadas pelo nome certo.
