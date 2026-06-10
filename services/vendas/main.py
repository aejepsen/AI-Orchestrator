"""Serviço Vendas: pedidos, política de desconto por papel e comissão.

Os itens referenciam SKUs como strings; a validação de estoque é responsabilidade
do orquestrador (Fase 3), não deste serviço.
"""

from contextlib import asynccontextmanager
from datetime import date
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from common import NotFound, register_error_handlers, register_internal_auth
from vendas import db, rules, seed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    with db.connect() as conn:
        db.init_schema(conn)
        seed.seed(conn)
    yield


app = FastAPI(
    title="Serviço de Vendas",
    description=(
        "Pedidos de venda com política de desconto por papel (até 10% 'vendedor', até 20% 'gerente', "
        "nunca acima de 20%) e comissão sobre o total líquido (2% até R$10.000; 3,5% acima)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
register_error_handlers(app)
register_internal_auth(app)

SalesRole = Literal["vendedor", "gerente"]


class OrderItem(BaseModel):
    sku: str = Field(description="SKU do produto (catálogo do serviço de Estoque)")
    quantity: int = Field(gt=0, description="Quantidade vendida")
    unit_price: float = Field(gt=0, description="Preço unitário negociado em reais")


class OrderCreate(BaseModel):
    customer: str = Field(min_length=2, description="Nome do cliente")
    salesperson: str = Field(min_length=2, description="Nome do vendedor responsável")
    role: SalesRole = Field(description="Papel de quem concede o desconto: 'vendedor' ou 'gerente'")
    discount_pct: float = Field(default=0, ge=0, description="Desconto percentual sobre o total bruto")
    items: list[OrderItem] = Field(min_length=1, description="Itens do pedido")
    order_date: date | None = Field(default=None, description="Data do pedido; default = hoje")


class Order(BaseModel):
    id: int
    customer: str
    salesperson: str
    role: SalesRole
    discount_pct: float
    order_date: date
    items: list[OrderItem]
    gross_total: float
    net_total: float


class Commission(BaseModel):
    order_id: int
    net_total: float
    rate: float = Field(description="Taxa aplicada: 0.02 até R$10.000 líquidos; 0.035 acima")
    commission: float


class Health(BaseModel):
    status: str
    service: str


def _load_order(conn, order_id: int) -> Order:
    row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
    if row is None:
        raise NotFound(
            error="order_not_found",
            detail=f"Pedido com id {order_id} não existe. Liste os pedidos em GET /orders para ver os ids válidos.",
        )
    items = [
        OrderItem(sku=r["sku"], quantity=r["quantity"], unit_price=r["unit_price"])
        for r in conn.execute("SELECT * FROM order_items WHERE order_id = ? ORDER BY id", (order_id,))
    ]
    return Order(
        id=row["id"],
        customer=row["customer"],
        salesperson=row["salesperson"],
        role=row["role"],
        discount_pct=row["discount_pct"],
        order_date=date.fromisoformat(row["order_date"]),
        items=items,
        gross_total=row["gross_total"],
        net_total=row["net_total"],
    )


@app.get("/health", operation_id="health_check", summary="Verifica se o serviço está no ar")
def health() -> Health:
    return Health(status="ok", service="vendas")


@app.get(
    "/orders",
    operation_id="list_orders",
    summary="Lista pedidos de venda com itens e totais",
)
def list_orders() -> list[Order]:
    with db.connect() as conn:
        ids = [r["id"] for r in conn.execute("SELECT id FROM orders ORDER BY order_date DESC, id DESC")]
        return [_load_order(conn, order_id) for order_id in ids]


@app.get(
    "/orders/{order_id}",
    operation_id="get_order",
    summary="Detalha um pedido pelo id",
)
def get_order(order_id: int) -> Order:
    with db.connect() as conn:
        return _load_order(conn, order_id)


@app.post(
    "/orders",
    operation_id="create_order",
    summary="Cria um pedido de venda",
    description=(
        "Valida a política de desconto antes de criar: até 10% para role='vendedor', "
        "até 20% para role='gerente'; acima de 20% é recusado para qualquer papel (422)."
    ),
    status_code=201,
)
def create_order(body: OrderCreate) -> Order:
    rules.validate_discount(body.discount_pct, body.role)
    totals = rules.order_totals([item.model_dump() for item in body.items], body.discount_pct)
    order_date = body.order_date or date.today()
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO orders (customer, salesperson, role, discount_pct, order_date, gross_total, net_total)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (body.customer, body.salesperson, body.role, body.discount_pct, order_date.isoformat(),
             totals["gross_total"], totals["net_total"]),
        )
        conn.executemany(
            "INSERT INTO order_items (order_id, sku, quantity, unit_price) VALUES (?, ?, ?, ?)",
            [(cur.lastrowid, item.sku, item.quantity, item.unit_price) for item in body.items],
        )
        conn.commit()
        return _load_order(conn, cur.lastrowid)


@app.get(
    "/orders/{order_id}/commission",
    operation_id="get_order_commission",
    summary="Calcula a comissão de um pedido",
    description="Comissão sobre o total líquido do pedido: 2% até R$10.000; 3,5% acima.",
)
def get_order_commission(order_id: int) -> Commission:
    with db.connect() as conn:
        order = _load_order(conn, order_id)
    result = rules.compute_commission(order.net_total)
    return Commission(order_id=order_id, net_total=order.net_total, **result)
