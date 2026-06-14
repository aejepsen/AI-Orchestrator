"""Persistência SQLite (stdlib) do serviço RH."""

import os
import sqlite3
from pathlib import Path

_DEFAULT_PATH = "./rh.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    department TEXT NOT NULL,
    position TEXT NOT NULL,
    hire_date TEXT NOT NULL,
    salary REAL NOT NULL CHECK (salary > 0)
);

CREATE TABLE IF NOT EXISTS vacations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL REFERENCES employees(id),
    acquisition_year INTEGER NOT NULL,
    start_date TEXT NOT NULL,
    days INTEGER NOT NULL CHECK (days > 0)
);

CREATE TABLE IF NOT EXISTS reimbursements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER NOT NULL REFERENCES employees(id),
    category TEXT NOT NULL CHECK (category IN ('viagem', 'refeicao', 'home_office')),
    amount REAL NOT NULL CHECK (amount > 0),
    days INTEGER,
    month TEXT NOT NULL
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
