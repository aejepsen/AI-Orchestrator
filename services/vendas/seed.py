"""Seed idempotente do serviço Vendas: pedidos com SKUs do catálogo de Estoque."""

import sqlite3
from datetime import date, timedelta

from vendas import rules

# (customer, salesperson, role, discount_pct, days_ago, items=[(sku, quantity, unit_price)])
_ORDERS = [
    ("Lojas Andrade S.A.", "Rafael Monteiro", "vendedor", 5.0, 2,
     [("CAD-ERG-001", 10, 1_450.00), ("MES-ELE-002", 5, 2_280.00)]),
    ("Cooperativa AgroVale", "Juliana Castro Neves", "gerente", 15.0, 5,
     [("NTB-DEV-004", 3, 9_800.00)]),
    ("Clínica Bem Viver", "Rafael Monteiro", "vendedor", 0.0, 8,
     [("MON-27P-003", 4, 2_150.00), ("TEC-MEC-005", 4, 520.00), ("MOU-ERG-006", 4, 310.00)]),
    ("Banco Horizonte", "Juliana Castro Neves", "gerente", 12.0, 12,
     [("MON-27P-003", 20, 2_150.00), ("HEA-BTH-009", 20, 890.00)]),
    ("Transportadora Rocha", "Rafael Monteiro", "vendedor", 8.0, 15,
     [("CAM-FHD-008", 12, 480.00)]),
    ("Escritório Vasconcellos Advogados", "Rafael Monteiro", "vendedor", 0.0, 20,
     [("SUP-NTB-010", 30, 180.00), ("HUB-USB-007", 10, 420.00)]),
    ("Colégio Santa Cecília", "Juliana Castro Neves", "gerente", 18.0, 25,
     [("NTB-DEV-004", 8, 9_800.00), ("MON-27P-003", 8, 2_150.00)]),
    ("Startup Maré Alta", "Rafael Monteiro", "vendedor", 10.0, 30,
     [("TEC-MEC-005", 6, 520.00), ("MOU-ERG-006", 6, 310.00), ("CAM-FHD-008", 6, 480.00)]),
]


def seed(conn: sqlite3.Connection, today: date | None = None) -> None:
    if conn.execute("SELECT COUNT(*) FROM orders").fetchone()[0] > 0:
        return
    base = today or date.today()
    for customer, salesperson, role, discount_pct, days_ago, items in _ORDERS:
        totals = rules.order_totals(
            [{"quantity": qty, "unit_price": price} for _, qty, price in items], discount_pct
        )
        cur = conn.execute(
            "INSERT INTO orders (customer, salesperson, role, discount_pct, order_date, gross_total, net_total)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (customer, salesperson, role, discount_pct, (base - timedelta(days=days_ago)).isoformat(),
             totals["gross_total"], totals["net_total"]),
        )
        conn.executemany(
            "INSERT INTO order_items (order_id, sku, quantity, unit_price) VALUES (?, ?, ?, ?)",
            [(cur.lastrowid, sku, qty, price) for sku, qty, price in items],
        )
    conn.commit()
