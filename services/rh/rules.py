"""Regras de negócio puras do domínio RH (sem I/O, sem relógio).

Férias seguem CLT simplificada: 30 dias por ano aquisitivo, elegível após
12 meses de admissão, fracionável em até 3 períodos (um ≥14 dias, demais ≥5).
"""

from datetime import date

from common import RuleViolation

VACATION_DAYS_PER_YEAR = 30
MAX_VACATION_PERIODS = 3
MIN_PERIOD_DAYS = 5
MIN_LONGEST_PERIOD_DAYS = 14

# Limites de reembolso por categoria (regra documentada na descrição da rota).
TRAVEL_LIMIT = 3_000.00
MEAL_LIMIT_PER_DAY = 100.00
HOME_OFFICE_LIMIT_PER_MONTH = 500.00


def add_months(base: date, months: int) -> date:
    """Soma meses preservando o dia quando possível (recua para o último dia do mês se necessário)."""
    total = base.month - 1 + months
    year, month = base.year + total // 12, total % 12 + 1
    for day in (base.day, 28):  # 28 cobre todos os meses como fallback
        try:
            return date(year, month, day)
        except ValueError:
            continue
    raise AssertionError("unreachable")


def vacation_balance(taken_days: list[int]) -> int:
    return VACATION_DAYS_PER_YEAR - sum(taken_days)


def validate_vacation_request(hire_date: date, start_date: date, days: int, taken_days: list[int]) -> None:
    """Valida um pedido de férias contra elegibilidade, fracionamento e saldo do ano aquisitivo.

    `taken_days` são as durações dos períodos já concedidos no mesmo ano aquisitivo.
    """
    eligible_from = add_months(hire_date, 12)
    if start_date < eligible_from:
        raise RuleViolation(
            error="vacation_not_yet_entitled",
            detail=(
                f"Funcionário admitido em {hire_date.isoformat()} só adquire direito a férias em "
                f"{eligible_from.isoformat()} (12 meses de admissão). Solicite início a partir dessa data."
            ),
            rule="direito a 30 dias de férias após 12 meses de admissão",
        )

    if days < MIN_PERIOD_DAYS:
        raise RuleViolation(
            error="vacation_period_too_short",
            detail=(
                f"Período de {days} dias é inválido: nenhum período de férias pode ter menos de "
                f"{MIN_PERIOD_DAYS} dias. Aumente a duração do período."
            ),
            rule=f"todo período de férias deve ter no mínimo {MIN_PERIOD_DAYS} dias",
        )

    if len(taken_days) >= MAX_VACATION_PERIODS:
        raise RuleViolation(
            error="vacation_too_many_periods",
            detail=(
                f"O ano aquisitivo já tem {len(taken_days)} períodos concedidos, o máximo permitido. "
                f"Não é possível fracionar as férias em mais de {MAX_VACATION_PERIODS} períodos."
            ),
            rule=f"férias fracionáveis em no máximo {MAX_VACATION_PERIODS} períodos por ano aquisitivo",
        )

    balance = vacation_balance(taken_days)
    if days > balance:
        raise RuleViolation(
            error="vacation_insufficient_balance",
            detail=(
                f"Pedido de {days} dias excede o saldo de {balance} dias do ano aquisitivo "
                f"(direito de {VACATION_DAYS_PER_YEAR} dias, {sum(taken_days)} já usados). "
                f"Solicite no máximo {balance} dias."
            ),
            rule=f"saldo de férias limitado a {VACATION_DAYS_PER_YEAR} dias por ano aquisitivo",
        )

    # Garante que ainda será possível ter um período ≥14 dias no ano aquisitivo.
    periods = [*taken_days, days]
    remaining = VACATION_DAYS_PER_YEAR - sum(periods)
    if max(periods) < MIN_LONGEST_PERIOD_DAYS and remaining < MIN_LONGEST_PERIOD_DAYS:
        raise RuleViolation(
            error="vacation_no_long_period_possible",
            detail=(
                f"Com esse pedido, nenhum período do ano aquisitivo teria {MIN_LONGEST_PERIOD_DAYS} dias ou mais "
                f"e o saldo restante ({remaining} dias) não permitiria um. Um dos períodos deve ter no mínimo "
                f"{MIN_LONGEST_PERIOD_DAYS} dias; ajuste a duração."
            ),
            rule=f"ao fracionar férias, um dos períodos deve ter no mínimo {MIN_LONGEST_PERIOD_DAYS} dias",
        )


def validate_reimbursement(category: str, amount: float, days: int | None, month_total: float) -> None:
    """Valida limites de reembolso: viagem R$3.000/pedido, refeição R$100/dia, home_office R$500/mês.

    `month_total` é o total já reembolsado na categoria home_office no mês de referência.
    """
    if category == "viagem":
        if amount > TRAVEL_LIMIT:
            raise RuleViolation(
                error="reimbursement_over_travel_limit",
                detail=(
                    f"Reembolso de viagem de R$ {amount:,.2f} excede o limite de R$ {TRAVEL_LIMIT:,.2f} por pedido. "
                    f"Divida em pedidos menores ou solicite exceção fora do sistema."
                ),
                rule=f"reembolso de viagem limitado a R$ {TRAVEL_LIMIT:,.2f} por pedido",
            )
    elif category == "refeicao":
        if not days or days < 1:
            raise RuleViolation(
                error="reimbursement_days_required",
                detail="Reembolso de refeição exige o campo 'days' (quantidade de dias, mínimo 1).",
                rule="reembolso de refeição é calculado por dia",
            )
        limit = MEAL_LIMIT_PER_DAY * days
        if amount > limit:
            raise RuleViolation(
                error="reimbursement_over_meal_limit",
                detail=(
                    f"Reembolso de refeição de R$ {amount:,.2f} para {days} dia(s) excede o limite de "
                    f"R$ {limit:,.2f} (R$ {MEAL_LIMIT_PER_DAY:,.2f}/dia). Reduza o valor ou informe mais dias."
                ),
                rule=f"reembolso de refeição limitado a R$ {MEAL_LIMIT_PER_DAY:,.2f} por dia",
            )
    elif category == "home_office":
        if month_total + amount > HOME_OFFICE_LIMIT_PER_MONTH:
            available = HOME_OFFICE_LIMIT_PER_MONTH - month_total
            raise RuleViolation(
                error="reimbursement_over_home_office_limit",
                detail=(
                    f"Reembolso de home office de R$ {amount:,.2f} excede o limite mensal de "
                    f"R$ {HOME_OFFICE_LIMIT_PER_MONTH:,.2f}: já foram reembolsados R$ {month_total:,.2f} no mês, "
                    f"restando R$ {available:,.2f}. Solicite no máximo o valor restante."
                ),
                rule=f"reembolso de home office limitado a R$ {HOME_OFFICE_LIMIT_PER_MONTH:,.2f} por mês",
            )
