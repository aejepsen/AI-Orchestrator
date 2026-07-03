# Critérios de rotulagem — `golden_routing.jsonl`

> Auditoria de 2026-07-02: o golden expandido (63→153) tinha ruído de label —
> a mesma pergunta-shape aparecia com conjuntos de domínios diferentes (ex.:
> "comissão que X gerou" ora `[vendas]`, ora `[financas,vendas]`, ora
> `[financas,rh,vendas]`). 8 labels normalizados pelos critérios abaixo.
> Regra de ouro: o critério decide, nunca o output do modelo.

## C1 — Comissão
Consulta de comissão (valor gerado por vendedor) = **vendas** puro — o serviço
de vendas calcula comissão (2%/3.5% sobre net_total). Inclui outros domínios
apenas com sinal EXPLÍCITO: "vamos pagar / cabe no caixa" → `+financas`;
"pesa na folha" → `+financas,+rh`.

## C2 — Aprovação de despesa/pagamento
- "QUEM aprova / quem precisa aprovar" (pergunta pela pessoa/cargo) →
  `financas + rh` (alçada é regra de finanças; aprovador é cargo no RH).
- Formas genéricas ("foi aprovado?", "precisam de aprovação", "total de
  despesas aprovadas") → `financas` puro.
- "Crédito aprovado" de cliente → `financas + vendas` (não é alçada).

## C3 — Orçamento por departamento
Orçamento é finanças. Departamento como mero QUALIFICADOR ("orçamento de
Engenharia/RH/Operações") **não** adiciona rh. `+rh` apenas quando a resposta
exige enumerar/agregar departamentos ("liste os departamentoS que estouraram")
ou dados de pessoas (folha, headcount, custo de equipe).

## C4 — Vendedor × região
Carteira/atuação comercial ("vendedores por região", "quem atende a região X")
= **vendas**. `+rh` apenas com sinal de staffing ("representante DEDICADO",
contratado/alocado).

## C5 — Compra de cliente
"Comprou / produtos comprados / pedidos com produtos" cruza pedidos (vendas)
com cadastro de produtos (estoque) → `vendas + estoque`.

## Processo
Mudanças de label exigem: (1) enquadrar num critério deste arquivo (ou criar
critério novo aqui), (2) commit citando o critério. Golden é o eval — nunca
ajustar label para "passar" um caso específico.
