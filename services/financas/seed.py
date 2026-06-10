"""Seed idempotente do serviço Finanças: contas a pagar/receber realistas."""

import sqlite3
from datetime import date, timedelta

# (type, description, counterparty, amount, due_in_days, status, approver_role, settled_in_days)
_ACCOUNTS = [
    ("pagar", "Aluguel do escritório - junho", "Imobiliária Paulista Ltda", 8_500.00, 5, "aberta", "gerente", None),
    ("pagar", "Licenças de software anuais", "TechSoft Brasil S.A.", 3_200.00, 12, "aberta", None, None),
    ("pagar", "Consultoria de implantação ERP", "Vetta Consultoria", 62_000.00, 30, "aberta", "diretor", None),
    ("pagar", "Energia elétrica - maio", "Enel Distribuição SP", 1_840.50, -3, "paga", None, -3),
    ("pagar", "Material de escritório", "Kalunga Comércio", 487.90, 8, "aberta", None, None),
    ("pagar", "Campanha de marketing digital Q3", "Agência Bossa Nova", 24_000.00, 20, "aberta", "gerente", None),
    ("receber", "Fatura #2031 - projeto e-commerce", "Lojas Andrade S.A.", 45_000.00, 10, "aberta", None, None),
    ("receber", "Mensalidade SaaS - plano enterprise", "Cooperativa AgroVale", 7_900.00, 3, "aberta", None, None),
    ("receber", "Fatura #2027 - manutenção mensal", "Clínica Bem Viver", 4_300.00, -7, "recebida", None, -5),
    ("receber", "Consultoria de dados - sprint 4", "Transportadora Rocha", 18_500.00, 25, "aberta", None, None),
    ("receber", "Fatura #2029 - integração de APIs", "Banco Horizonte", 32_750.00, 15, "aberta", None, None),
]


def seed(conn: sqlite3.Connection, today: date | None = None) -> None:
    if conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] > 0:
        return
    base = today or date.today()
    rows = [
        (
            type_,
            description,
            counterparty,
            amount,
            (base + timedelta(days=due_in)).isoformat(),
            status,
            approver_role,
            (base + timedelta(days=settled_in)).isoformat() if settled_in is not None else None,
        )
        for type_, description, counterparty, amount, due_in, status, approver_role, settled_in in _ACCOUNTS
    ]
    conn.executemany(
        "INSERT INTO accounts (type, description, counterparty, amount, due_date, status, approver_role, settled_date)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
