"""Seed idempotente do serviço Estoque: SKUs com níveis variados (incl. abaixo do ponto de reposição)."""

import sqlite3

# (sku, name, category, unit_price, on_hand, reorder_point)
_PRODUCTS = [
    ("CAD-ERG-001", "Cadeira ergonômica Presence", "Mobiliário", 1_450.00, 32, 10),
    ("MES-ELE-002", "Mesa elevatória Standy 140cm", "Mobiliário", 2_280.00, 8, 12),   # abaixo do ponto
    ("MON-27P-003", "Monitor 27\" 4K ProView", "Eletrônicos", 2_150.00, 45, 15),
    ("NTB-DEV-004", "Notebook Dev 32GB RAM", "Eletrônicos", 9_800.00, 6, 8),          # abaixo do ponto
    ("TEC-MEC-005", "Teclado mecânico ABNT2 Silent", "Periféricos", 520.00, 120, 30),
    ("MOU-ERG-006", "Mouse ergonômico vertical", "Periféricos", 310.00, 75, 25),
    ("HUB-USB-007", "Hub USB-C 8 portas", "Periféricos", 420.00, 18, 20),             # abaixo do ponto
    ("CAM-FHD-008", "Webcam Full HD com microfone", "Eletrônicos", 480.00, 54, 15),
    ("HEA-BTH-009", "Headset Bluetooth com cancelamento", "Periféricos", 890.00, 40, 12),
    ("SUP-NTB-010", "Suporte de notebook em alumínio", "Acessórios", 180.00, 200, 50),
]

# (sku, quantity) — reservas ativas iniciais
_RESERVATIONS = [
    ("MON-27P-003", 10),
    ("NTB-DEV-004", 2),
    ("TEC-MEC-005", 20),
]


def seed(conn: sqlite3.Connection) -> None:
    if conn.execute("SELECT COUNT(*) FROM products").fetchone()[0] > 0:
        return
    conn.executemany(
        "INSERT INTO products (sku, name, category, unit_price, on_hand, reorder_point) VALUES (?, ?, ?, ?, ?, ?)",
        _PRODUCTS,
    )
    conn.executemany("INSERT INTO reservations (sku, quantity) VALUES (?, ?)", _RESERVATIONS)
    conn.commit()
