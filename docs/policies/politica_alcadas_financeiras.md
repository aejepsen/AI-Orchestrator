# Política de Alçadas Financeiras

## Aprovação de despesas

Toda despesa (conta a pagar) segue a alçada por valor:

- Até R$ 5.000,00: aprovação automática — nenhum aprovador é exigido.
- De R$ 5.000,01 até R$ 50.000,00: exige aprovação de **gerente**.
- Acima de R$ 50.000,00: exige aprovação de **diretor**.

O papel do aprovador é registrado na própria conta (`approver_role`). Despesas
criadas sem o aprovador exigido pela faixa de valor são recusadas pelo sistema.

## Liquidação

- Contas do tipo **pagar** são liquidadas pela ação de pagamento; contas do tipo
  **receber**, pela ação de recebimento. As ações não são intercambiáveis.
- Apenas contas com status **aberta** podem ser liquidadas ou excluídas.
- A data de liquidação, quando não informada, é a data corrente.

## Fluxo de caixa

O fluxo de caixa projetado considera o vencimento (`due_date`) das contas
abertas no período consultado: entradas (receber) menos saídas (pagar).
