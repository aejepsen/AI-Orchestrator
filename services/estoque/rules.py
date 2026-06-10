"""Regras de negócio puras do domínio Estoque (sem I/O)."""

from common import RuleViolation


def available(on_hand: int, reserved: int) -> int:
    """Quantidade disponível para reserva: saldo físico menos reservas ativas."""
    return on_hand - reserved


def validate_reservation(sku: str, on_hand: int, reserved: int, quantity: int) -> None:
    """Reserva não pode exceder o disponível (on_hand - reserved)."""
    free = available(on_hand, reserved)
    if quantity > free:
        raise RuleViolation(
            error="insufficient_stock",
            detail=(
                f"Não é possível reservar {quantity} unidade(s) do SKU {sku}: há {on_hand} em estoque, "
                f"{reserved} já reservada(s), restando {free} disponível(is). "
                f"Reserve no máximo {free} unidade(s) ou aguarde reposição."
            ),
            rule="reserva limitada ao disponível (estoque físico menos reservas ativas)",
        )


def replenishment_suggestion(on_hand: int, reserved: int, reorder_point: int) -> int | None:
    """Sugere reposição quando o disponível cai abaixo do ponto de reposição.

    Quantidade sugerida: reorder_point * 2 - disponível. Retorna None se não precisa repor.
    """
    free = available(on_hand, reserved)
    if free < reorder_point:
        return reorder_point * 2 - free
    return None
