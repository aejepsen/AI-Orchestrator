"""Persistência SQLite (stdlib) do serviço Vendas."""

import os
import sqlite3
from pathlib import Path

_DEFAULT_PATH = "./vendas.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    customer TEXT NOT NULL,
    salesperson TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('vendedor', 'gerente')),
    discount_pct REAL NOT NULL DEFAULT 0 CHECK (discount_pct >= 0),
    order_date TEXT NOT NULL,
    gross_total REAL NOT NULL,
    net_total REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS order_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id INTEGER NOT NULL REFERENCES orders(id),
    sku TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price REAL NOT NULL CHECK (unit_price > 0)
);
"""


def connect() -> sqlite3.Connection:
    path = os.environ.get("DB_PATH", _DEFAULT_PATH)
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()
