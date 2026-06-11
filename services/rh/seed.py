"""Seed idempotente do serviço RH: funcionários com tempos de casa variados."""

import sqlite3
from datetime import date, timedelta

# (name, department, position, hired_days_ago, salary)
_EMPLOYEES = [
    ("Mariana Albuquerque", "Engenharia", "Engenheira de Software Sênior", 1460, 18_500.00),
    ("Carlos Eduardo Pontes", "Engenharia", "Engenheiro de Dados Pleno", 800, 12_300.00),
    ("Fernanda Yoshida", "Engenharia", "Tech Lead", 2200, 22_000.00),
    ("Rafael Monteiro", "Vendas", "Executivo de Contas", 650, 8_900.00),
    ("Juliana Castro Neves", "Vendas", "Gerente Comercial", 1825, 16_700.00),
    ("André Luiz Sampaio", "Financeiro", "Analista Financeiro Sênior", 1100, 9_800.00),
    ("Beatriz Fonseca", "Financeiro", "Coordenadora de Controladoria", 2920, 14_200.00),
    ("Thiago Nakamura", "RH", "Business Partner", 540, 10_500.00),
    ("Larissa Prado", "Engenharia", "Engenheira de Software Júnior", 90, 6_200.00),  # sem direito a férias
    ("Gustavo Bittencourt", "Operações", "Analista de Logística", 400, 5_400.00),
    ("Paula Souza", "Vendas", "Executiva de Contas", 980, 9_400.00),
]


def seed(conn: sqlite3.Connection, today: date | None = None) -> None:
    if conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] > 0:
        return
    base = today or date.today()
    conn.executemany(
        "INSERT INTO employees (name, department, position, hire_date, salary) VALUES (?, ?, ?, ?, ?)",
        [
            (name, dept, position, (base - timedelta(days=days_ago)).isoformat(), salary)
            for name, dept, position, days_ago, salary in _EMPLOYEES
        ],
    )
    # Mariana (id=1) já tirou 10 dias no ano aquisitivo corrente; saldo de 20.
    conn.execute(
        "INSERT INTO vacations (employee_id, acquisition_year, start_date, days) VALUES (?, ?, ?, ?)",
        (1, base.year, (base - timedelta(days=60)).isoformat(), 10),
    )
    # André (id=6) já reembolsou R$320 de home office no mês corrente; restam R$180.
    conn.execute(
        "INSERT INTO reimbursements (employee_id, category, amount, days, month) VALUES (?, ?, ?, ?, ?)",
        (6, "home_office", 320.00, None, base.strftime("%Y-%m")),
    )
    conn.commit()
