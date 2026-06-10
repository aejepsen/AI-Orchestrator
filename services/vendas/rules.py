"""Regras de negócio puras do domínio Vendas (sem I/O)."""

from common import RuleViolation

DISCOUNT_LIMIT_BY_ROLE = {"vendedor": 10.0, "gerente": 20.0}
MAX_DISCOUNT_PCT = 20.0

COMMISSION_THRESHOLD = 10_000.00
COMMISSION_RATE_BASE = 0.02
COMMISSION_RATE_HIGH = 0.035


def validate_discount(discount_pct: float, role: str) -> None:
    """Desconto até 10% para 'vendedor', até 20% para 'gerente'; acima de 20% é sempre recusado."""
    if discount_pct > MAX_DISCOUNT_PCT:
        raise RuleViolation(
            error="discount_above_policy_maximum",
            detail=(
                f"Desconto de {discount_pct:.1f}% excede o teto absoluto de {MAX_DISCOUNT_PCT:.0f}% da política "
                f"comercial e não pode ser aprovado por nenhum papel. Reduza o desconto."
            ),
            rule=f"nenhum desconto pode exceder {MAX_DISCOUNT_PCT:.0f}%",
        )
    limit = DISCOUNT_LIMIT_BY_ROLE[role]
    if discount_pct > limit:
        raise RuleViolation(
            error="discount_above_role_limit",
            detail=(
                f"Desconto de {discount_pct:.1f}% excede o limite de {limit:.0f}% do papel '{role}'. "
                f"Reduza o desconto ou submeta com role='gerente' (limite de "
                f"{DISCOUNT_LIMIT_BY_ROLE['gerente']:.0f}%)."
            ),
            rule="desconto até 10% para 'vendedor'; até 20% para 'gerente'",
        )


def order_totals(items: list[dict], discount_pct: float) -> dict:
    """Total bruto (soma de quantity * unit_price) e líquido após o desconto percentual."""
    gross = round(sum(item["quantity"] * item["unit_price"] for item in items), 2)
    net = round(gross * (1 - discount_pct / 100), 2)
    return {"gross_total": gross, "net_total": net}


def compute_commission(net_total: float) -> dict:
    """Comissão sobre o total líquido: 2% até R$10.000; 3,5% acima."""
    rate = COMMISSION_RATE_BASE if net_total <= COMMISSION_THRESHOLD else COMMISSION_RATE_HIGH
    return {"rate": rate, "commission": round(net_total * rate, 2)}
