# Política de Estoque e Reservas

## Disponibilidade

O saldo **disponível** de um SKU é o estoque físico (`on_hand`) menos as
reservas **ativas**. Reservas liberadas não contam.

## Reservas

- Uma reserva só é aceita se a quantidade solicitada couber no disponível.
- Reservas ativas podem ser **liberadas**, devolvendo a quantidade ao
  disponível.

## Reposição

Cada SKU tem um **ponto de reposição** (`reorder_point`). Quando o disponível
fica igual ou abaixo do ponto de reposição, o sistema sugere a quantidade de
reposição necessária para recompor o estoque. Compras de reposição seguem a
política de alçadas financeiras para aprovação da despesa.
