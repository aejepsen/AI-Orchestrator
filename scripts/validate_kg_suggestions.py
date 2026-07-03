"""Curadoria das sugestões do link prediction contra a FONTE DE VERDADE (serviços).

Princípio do projeto: a regra de negócio vive na API. Uma relação candidata só
é APROVADA se corroborada pelos dados dos microsserviços (seeds em sqlite
:memory:); relações entre entidades que só existem no enriquecimento do KG
não têm evidência externa → REJEITADAS por default (não inventar arestas).

Vereditos:
- aprovada           — fato confirmado nos dados do serviço
- contradiz          — serviço tem o fato DIFERENTE (ex.: funcionário em outro depto)
- sem_evidencia      — relação/entidade sem lastro em serviço → rejeitar

Uso:
    .venv/bin/python scripts/validate_kg_suggestions.py [--csv <kg_suggestions_*.csv>]
Saída: evals/results/kg_suggestions_validated_<ts>.csv + resumo no stdout.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "services"))

RESULTS_DIR = ROOT / "evals" / "results"


def _service_db(service: str) -> sqlite3.Connection:
    db = importlib.import_module(f"{service}.db")
    seed = importlib.import_module(f"{service}.seed")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    db.init_schema(conn)
    seed.seed(conn)
    return conn


def _norm(text: str) -> str:
    return text.strip().casefold()


class FactChecker:
    """Fatos extraídos dos seeds dos 4 serviços, indexados para lookup."""

    def __init__(self) -> None:
        rh = _service_db("rh")
        estoque = _service_db("estoque")
        vendas = _service_db("vendas")
        financas = _service_db("financas")

        self.employee_department = {
            _norm(r["name"]): _norm(r["department"]) for r in rh.execute("SELECT name, department FROM employees")
        }
        self.employee_position = {
            _norm(r["name"]): _norm(r["position"]) for r in rh.execute("SELECT name, position FROM employees")
        }
        self.product_category = {
            _norm(r["name"]): _norm(r["category"]) for r in estoque.execute("SELECT name, category FROM products")
        }
        sku_to_product = {r["sku"]: _norm(r["name"]) for r in estoque.execute("SELECT sku, name FROM products")}
        self.customer_bought: set[tuple[str, str]] = set()
        for r in vendas.execute(
            "SELECT o.customer AS customer, i.sku AS sku FROM orders o JOIN order_items i ON i.order_id = o.id"
        ):
            product = sku_to_product.get(r["sku"])
            if product:
                self.customer_bought.add((_norm(r["customer"]), product))
        self.seller_sold_to = {
            (_norm(r["salesperson"]), _norm(r["customer"]))
            for r in vendas.execute("SELECT salesperson, customer FROM orders")
        }
        # Alçada (regra de finanças): >50k diretor, >5k gerente, senão auto.
        self.expense_approver: dict[str, str] = {}
        for r in financas.execute("SELECT description, amount FROM accounts WHERE type = 'pagar'"):
            role = "diretor" if r["amount"] > 50_000 else ("gerente" if r["amount"] > 5_000 else "auto")
            self.expense_approver[_norm(r["description"])] = role

    def verdict(self, relation: str, head: str, tail: str) -> tuple[str, str]:
        head_type, _, head_name = head.partition(":")
        tail_type, _, tail_name = tail.partition(":")
        h, t = _norm(head_name), _norm(tail_name)

        if relation == "TRABALHA_EM" and head_type == "funcionario":
            actual = self.employee_department.get(h)
            if actual is None:
                return "sem_evidencia", "funcionário não existe no serviço de RH (entidade só do enriquecimento)"
            return ("aprovada", "confirmado em rh.employees") if actual == t else (
                "contradiz", f"RH registra departamento '{actual}'"
            )
        if relation == "TEM_CARGO" and head_type == "funcionario":
            actual = self.employee_position.get(h)
            if actual is None:
                return "sem_evidencia", "funcionário não existe no serviço de RH"
            return ("aprovada", "confirmado em rh.employees") if actual == t else (
                "contradiz", f"RH registra cargo '{actual}'"
            )
        if relation == "PERTENCE_A" and head_type == "produto":
            actual = self.product_category.get(h)
            if actual is None:
                return "sem_evidencia", "produto não existe no serviço de estoque"
            return ("aprovada", "confirmado em estoque.products") if actual == t else (
                "contradiz", f"estoque registra categoria '{actual}'"
            )
        if relation == "COMPROU" and head_type in ("cliente", "cliente_recebivel"):
            if (h, t) in self.customer_bought:
                return "aprovada", "confirmado em vendas.orders/order_items"
            return "sem_evidencia", "nenhum pedido do cliente contém este produto"
        if relation == "VENDEU_PARA":
            if (h, t) in self.seller_sold_to:
                return "aprovada", "confirmado em vendas.orders"
            return "sem_evidencia", "nenhum pedido liga este vendedor a este cliente"
        if relation == "REQUER_APROVACAO" and head_type == "despesa":
            actual = self.expense_approver.get(h)
            if actual is None:
                return "sem_evidencia", "despesa não existe no serviço de finanças"
            if actual == "auto":
                return "contradiz", "despesa abaixo da alçada (não exige aprovador)"
            return ("aprovada", f"alçada de finanças exige '{actual}'") if actual == t else (
                "contradiz", f"alçada exige '{actual}'"
            )
        return "sem_evidencia", f"relação {relation} sem lastro verificável nos serviços"


def latest_suggestions() -> Path | None:
    candidates = sorted(RESULTS_DIR.glob("kg_suggestions_2*.csv"))
    return candidates[-1] if candidates else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida sugestões do KG contra os seeds dos serviços")
    parser.add_argument("--csv", default=None, help="CSV de sugestões (default: mais recente)")
    args = parser.parse_args()

    path = Path(args.csv) if args.csv else latest_suggestions()
    if not path or not path.exists():
        print("Nenhum kg_suggestions_*.csv encontrado — rode scripts/kg_link_prediction.py antes.", file=sys.stderr)
        return 2

    checker = FactChecker()
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    validated = []
    for row in rows:
        verdict, reason = checker.verdict(row["relation"], row["head"], row["tail"])
        validated.append({**row, "veredito": verdict, "motivo": reason})

    out = RESULTS_DIR / f"kg_suggestions_validated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(validated[0].keys()))
        writer.writeheader()
        writer.writerows(validated)

    counts: dict[str, int] = {}
    for row in validated:
        counts[row["veredito"]] = counts.get(row["veredito"], 0) + 1
    print(f"Sugestões: {len(validated)} de {path.name}")
    print(f"Vereditos: {counts}")
    for row in validated:
        if row["veredito"] == "aprovada":
            print(f"  ✓ [{row['relation']}] {row['head']} → {row['tail']} ({row['motivo']})")
        elif row["veredito"] == "contradiz":
            print(f"  ✗ [{row['relation']}] {row['head']} → {row['tail']} — {row['motivo']}")
    print(f"CSV validado: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
