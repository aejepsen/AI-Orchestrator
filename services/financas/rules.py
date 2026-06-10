"""Regras de negócio puras do domínio Finanças (sem I/O, sem relógio)."""

from datetime import date

from common import RuleViolation

AUTO_APPROVAL_LIMIT = 5_000.00
MANAGER_APPROVAL_LIMIT = 50_000.00

_ROLE_RANK = {"gerente": 1, "diretor": 2}


def required_approver_role(amount: float) -> str | None:
    """Alçada mínima exigida para aprovar uma despesa, ou None se auto-aprovada."""
    if amount <= AUTO_APPROVAL_LIMIT:
        return None
    if amount <= MANAGER_APPROVAL_LIMIT:
        return "gerente"
    return "diretor"


def validate_expense_approval(amount: float, approver_role: str | None) -> None:
    """Despesa ≤ R$5.000 é auto-aprovada; ≤ R$50.000 exige 'gerente'; acima exige 'diretor'."""
    required = required_approver_role(amount)
    if required is None:
        return
    if approver_role is None or _ROLE_RANK.get(approver_role, 0) < _ROLE_RANK[required]:
        raise RuleViolation(
            error="insufficient_approval_authority",
            detail=(
                f"Despesa de R$ {amount:,.2f} exige aprovação com alçada de '{required}', "
                f"mas o aprovador informado foi '{approver_role or 'nenhum'}'. "
                f"Reenvie a despesa com approver_role='{required}' ou superior."
            ),
            rule=(
                "alçada de aprovação: ≤ R$5.000 auto-aprovada; "
                "≤ R$50.000 exige 'gerente'; acima de R$50.000 exige 'diretor'"
            ),
        )


def validate_settlement(account_type: str, status: str, action: str) -> None:
    """'pay' só se aplica a conta 'pagar' aberta; 'receive' só a conta 'receber' aberta."""
    expected_type = "pagar" if action == "pay" else "receber"
    if account_type != expected_type:
        raise RuleViolation(
            error="wrong_account_type",
            detail=(
                f"A ação '{action}' só se aplica a contas do tipo '{expected_type}', "
                f"mas esta conta é do tipo '{account_type}'. "
                f"Use a ação correta para o tipo da conta."
            ),
            rule="pagar só liquida contas a pagar; receber só liquida contas a receber",
        )
    if status != "aberta":
        raise RuleViolation(
            error="account_already_settled",
            detail=(
                f"Esta conta já está com status '{status}' e não pode ser liquidada novamente. "
                f"Apenas contas com status 'aberta' aceitam liquidação."
            ),
            rule="só contas abertas podem ser pagas ou recebidas",
        )


def compute_cashflow(accounts: list[dict], start: date, end: date) -> dict:
    """Fluxo de caixa projetado por vencimento: entradas (receber), saídas (pagar), saldo."""
    in_period = [a for a in accounts if start <= a["due_date"] <= end]
    entradas = round(sum(a["amount"] for a in in_period if a["type"] == "receber"), 2)
    saidas = round(sum(a["amount"] for a in in_period if a["type"] == "pagar"), 2)
    return {"entradas": entradas, "saidas": saidas, "saldo": round(entradas - saidas, 2)}
