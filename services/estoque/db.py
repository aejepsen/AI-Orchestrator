"""Persistência SQLite (stdlib) do serviço Estoque."""

import os
import sqlite3
from pathlib import Path

_DEFAULT_PATH = "./estoque.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT NOT NULL,
    unit_price REAL NOT NULL CHECK (unit_price > 0),
    on_hand INTEGER NOT NULL CHECK (on_hand >= 0),
    reorder_point INTEGER NOT NULL CHECK (reorder_point >= 0)
);

CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL REFERENCES products(sku),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    status TEXT NOT NULL DEFAULT 'ativa' CHECK (status IN ('ativa', 'liberada'))
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


def reserved_quantity(conn: sqlite3.Connection, sku: str) -> int:
    """Total reservado ativo para um SKU."""
    return conn.execute(
        "SELECT COALESCE(SUM(quantity), 0) FROM reservations WHERE sku = ? AND status = 'ativa'", (sku,)
    ).fetchone()[0]
