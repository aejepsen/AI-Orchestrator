"""Persistência SQLite (stdlib) do serviço Finanças."""

import os
import sqlite3
from pathlib import Path

_DEFAULT_PATH = "./financas.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL CHECK (type IN ('pagar', 'receber')),
    description TEXT NOT NULL,
    counterparty TEXT NOT NULL,
    amount REAL NOT NULL CHECK (amount > 0),
    due_date TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'aberta' CHECK (status IN ('aberta', 'paga', 'recebida')),
    approver_role TEXT,
    settled_date TEXT
);
"""


def connect() -> sqlite3.Connection:
    path = os.environ.get("DB_PATH", _DEFAULT_PATH)
    Path(path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    conn.commit()
