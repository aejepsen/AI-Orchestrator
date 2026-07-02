"""Serviço Finanças: contas a pagar/receber, fluxo de caixa e aprovação por alçada."""

from contextlib import asynccontextmanager
from datetime import date
from typing import AsyncIterator, Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from common import (
    NotFound,
    RuleViolation,
    register_admin_reset,
    register_error_handlers,
    register_internal_auth,
)
from financas import db, rules, seed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    with db.connect() as conn:
        db.init_schema(conn)
        seed.seed(conn)
    yield


app = FastAPI(
    title="Serviço de Finanças",
    description=(
        "Contas a pagar e a receber, fluxo de caixa projetado e aprovação de despesas por alçada. "
        "Despesas acima de R$5.000 exigem aprovador 'gerente'; acima de R$50.000, 'diretor'."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
register_error_handlers(app)
register_internal_auth(app)
register_admin_reset(app, service="financas", connect=db.connect, init_schema=db.init_schema, seed=seed.seed)

AccountType = Literal["pagar", "receber"]
ApproverRole = Literal["gerente", "diretor"]


class AccountCreate(BaseModel):
    type: AccountType = Field(description="Tipo da conta: 'pagar' (despesa) ou 'receber' (receita)")
    description: str = Field(min_length=3, description="Descrição da conta")
    counterparty: str = Field(min_length=2, description="Fornecedor ou cliente")
    amount: float = Field(gt=0, description="Valor em reais")
    due_date: date = Field(description="Data de vencimento (YYYY-MM-DD)")
    approver_role: ApproverRole | None = Field(
        default=None, description="Papel de quem aprovou a despesa, quando exigido pela alçada"
    )


class Account(AccountCreate):
    id: int
    status: Literal["aberta", "paga", "recebida"]
    settled_date: date | None


class AccountUpdate(BaseModel):
    description: str | None = Field(default=None, min_length=3, description="Nova descrição da conta")
    amount: float | None = Field(default=None, gt=0, description="Novo valor em reais")
    due_date: date | None = Field(default=None, description="Nova data de vencimento (YYYY-MM-DD)")


class SettleRequest(BaseModel):
    settled_date: date | None = Field(default=None, description="Data da liquidação; default = hoje")


class Cashflow(BaseModel):
    start: date
    end: date
    entradas: float
    saidas: float
    saldo: float


class Health(BaseModel):
    status: str
    service: str


def _row_to_account(row) -> Account:
    return Account(
        id=row["id"],
        type=row["type"],
        description=row["description"],
        counterparty=row["counterparty"],
        amount=row["amount"],
        due_date=date.fromisoformat(row["due_date"]),
        status=row["status"],
        approver_role=row["approver_role"],
        settled_date=date.fromisoformat(row["settled_date"]) if row["settled_date"] else None,
    )


def _get_account_row(conn, account_id: int):
    row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
    if row is None:
        raise NotFound(
            error="account_not_found",
            detail=f"Conta com id {account_id} não existe. Liste as contas em GET /accounts para ver os ids válidos.",
        )
    return row


@app.get("/health", operation_id="health_check", summary="Verifica se o serviço está no ar")
def health() -> Health:
    return Health(status="ok", service="financas")


@app.get(
    "/accounts",
    operation_id="list_accounts",
    summary="Lista contas a pagar e a receber",
    description="Retorna todas as contas, com filtros opcionais por tipo ('pagar'/'receber') e status.",
)
def list_accounts(
    type: AccountType | None = Query(default=None, description="Filtra por tipo"),
    status: Literal["aberta", "paga", "recebida"] | None = Query(default=None, description="Filtra por status"),
) -> list[Account]:
    query, params = "SELECT * FROM accounts", []
    clauses = []
    if type is not None:
        clauses.append("type = ?")
        params.append(type)
    if status is not None:
        clauses.append("status = ?")
        params.append(status)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    with db.connect() as conn:
        return [_row_to_account(r) for r in conn.execute(query + " ORDER BY due_date", params)]


@app.get(
    "/accounts/{account_id}",
    operation_id="get_account",
    summary="Detalha uma conta pelo id",
)
def get_account(account_id: int) -> Account:
    with db.connect() as conn:
        return _row_to_account(_get_account_row(conn, account_id))


@app.post(
    "/accounts",
    operation_id="create_account",
    summary="Cria conta a pagar ou a receber",
    description=(
        "Cria uma conta. Despesas (type='pagar') passam pela alçada de aprovação: "
        "≤ R$5.000 auto-aprovada; ≤ R$50.000 exige approver_role='gerente'; acima exige 'diretor'."
    ),
    status_code=201,
)
def create_account(body: AccountCreate) -> Account:
    if body.type == "pagar":
        rules.validate_expense_approval(body.amount, body.approver_role)
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO accounts (type, description, counterparty, amount, due_date, approver_role)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (body.type, body.description, body.counterparty, body.amount, body.due_date.isoformat(),
             body.approver_role),
        )
        conn.commit()
        return _row_to_account(_get_account_row(conn, cur.lastrowid))


@app.put(
    "/accounts/{account_id}",
    operation_id="update_account",
    summary="Atualiza uma conta existente (description, amount, due_date)",
    description="Atualiza campos opcionais de uma conta aberta. Contas já liquidadas não podem ser alteradas.",
)
def update_account(account_id: int, body: AccountUpdate) -> Account:
    with db.connect() as conn:
        row = _get_account_row(conn, account_id)
        if row["status"] != "aberta":
            raise RuleViolation(
                error="account_already_settled",
                detail=(
                    f"Conta com id {account_id} está com status '{row['status']}' e não pode ser alterada. "
                    f"Apenas contas com status 'aberta' aceitam edição."
                ),
                rule="só contas abertas podem ser editadas",
            )
        fields, params = [], []
        if body.description is not None:
            fields.append("description = ?")
            params.append(body.description)
        if body.amount is not None:
            if row["type"] == "pagar":
                rules.validate_expense_approval(body.amount, row["approver_role"])
            fields.append("amount = ?")
            params.append(body.amount)
        if body.due_date is not None:
            fields.append("due_date = ?")
            params.append(body.due_date.isoformat())
        if not fields:
            return _row_to_account(row)
        params.append(account_id)
        conn.execute(f"UPDATE accounts SET {', '.join(fields)} WHERE id = ?", params)
        conn.commit()
        return _row_to_account(_get_account_row(conn, account_id))


@app.delete(
    "/accounts/{account_id}",
    operation_id="delete_account",
    summary="Exclui uma conta existente",
    description="Exclui uma conta aberta. Contas já liquidadas não podem ser excluídas.",
    status_code=204,
)
def delete_account(account_id: int):
    with db.connect() as conn:
        row = _get_account_row(conn, account_id)
        if row["status"] != "aberta":
            raise RuleViolation(
                error="account_already_settled",
                detail=(
                    f"Conta com id {account_id} está com status '{row['status']}' e não pode ser excluída. "
                    f"Apenas contas com status 'aberta' aceitam exclusão."
                ),
                rule="só contas abertas podem ser excluídas",
            )
        conn.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
        conn.commit()
    return None


def _settle(account_id: int, action: Literal["pay", "receive"], body: SettleRequest | None) -> Account:
    settled = (body.settled_date if body and body.settled_date else date.today()).isoformat()
    new_status = "paga" if action == "pay" else "recebida"
    with db.connect() as conn:
        row = _get_account_row(conn, account_id)
        rules.validate_settlement(row["type"], row["status"], action)
        conn.execute("UPDATE accounts SET status = ?, settled_date = ? WHERE id = ?", (new_status, settled, account_id))
        conn.commit()
        return _row_to_account(_get_account_row(conn, account_id))


@app.post(
    "/accounts/{account_id}/pay",
    operation_id="pay_account",
    summary="Liquida (paga) uma conta a pagar aberta",
)
def pay_account(account_id: int, body: SettleRequest | None = None) -> Account:
    return _settle(account_id, "pay", body)


@app.post(
    "/accounts/{account_id}/receive",
    operation_id="receive_account",
    summary="Liquida (recebe) uma conta a receber aberta",
)
def receive_account(account_id: int, body: SettleRequest | None = None) -> Account:
    return _settle(account_id, "receive", body)


@app.get(
    "/cashflow",
    operation_id="get_cashflow",
    summary="Fluxo de caixa projetado por período",
    description="Soma entradas (contas a receber) e saídas (contas a pagar) com vencimento entre start e end.",
)
def get_cashflow(
    start: date = Query(description="Início do período (YYYY-MM-DD)"),
    end: date = Query(description="Fim do período (YYYY-MM-DD)"),
) -> Cashflow:
    with db.connect() as conn:
        accounts = [
            {"type": r["type"], "amount": r["amount"], "due_date": date.fromisoformat(r["due_date"])}
            for r in conn.execute("SELECT type, amount, due_date FROM accounts")
        ]
    flow = rules.compute_cashflow(accounts, start, end)
    return Cashflow(start=start, end=end, **flow)
