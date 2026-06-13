"""Serviço Vendas: pedidos, política de desconto por papel e comissão.

Os itens referenciam SKUs como strings; a validação de estoque é responsabilidade
do orquestrador (Fase 3), não deste serviço.
"""

from contextlib import asynccontextmanager
from datetime import date
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from common import NotFound, RuleViolation, register_error_handlers, register_internal_auth
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
OrderStatus = Literal["ativo", "concluido", "cancelado"]


class SellerCreate(BaseModel):
    name: str = Field(min_length=1, description="Nome do vendedor")
    region: str = Field(default="", description="Região de atuação")


class SellerUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, description="Nome do vendedor")
    region: str | None = Field(default=None, description="Região de atuação")


class Seller(BaseModel):
    id: int
    name: str
    region: str


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


class OrderUpdate(BaseModel):
    quantity: int | None = Field(default=None, gt=0, description="Nova quantidade do item")
    unit_price: float | None = Field(default=None, gt=0, description="Novo preço unitário")
    discount_pct: float | None = Field(default=None, ge=0, description="Novo desconto percentual")


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
    status: OrderStatus = "ativo"


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
    status = "ativo"
    try:
        status = row["status"]
    except (IndexError, KeyError):
        pass
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
        status=status,
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


# ── Sellers CRUD ──────────────────────────────────────────────────────────


@app.get(
    "/sellers",
    operation_id="list_sellers",
    summary="Lista vendedores cadastrados",
)
def list_sellers() -> list[Seller]:
    with db.connect() as conn:
        return [
            Seller(id=r["id"], name=r["name"], region=r["region"])
            for r in conn.execute("SELECT * FROM sellers ORDER BY id")
        ]


@app.get(
    "/sellers/{seller_id}",
    operation_id="get_seller",
    summary="Detalha um vendedor pelo id",
)
def get_seller(seller_id: int) -> Seller:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM sellers WHERE id = ?", (seller_id,)).fetchone()
        if row is None:
            raise NotFound(
                error="seller_not_found",
                detail=f"Vendedor com id {seller_id} não existe.",
            )
        return Seller(id=row["id"], name=row["name"], region=row["region"])


@app.post(
    "/sellers",
    operation_id="create_seller",
    summary="Cria um novo vendedor",
    status_code=201,
)
def create_seller(body: SellerCreate) -> Seller:
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO sellers (name, region) VALUES (?, ?)",
            (body.name, body.region),
        )
        conn.commit()
        return Seller(id=cur.lastrowid, name=body.name, region=body.region)


@app.put(
    "/sellers/{seller_id}",
    operation_id="update_seller",
    summary="Atualiza nome ou região de um vendedor",
)
def update_seller(seller_id: int, body: SellerUpdate) -> Seller:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM sellers WHERE id = ?", (seller_id,)).fetchone()
        if row is None:
            raise NotFound(
                error="seller_not_found",
                detail=f"Vendedor com id {seller_id} não existe.",
            )
        name = body.name if body.name is not None else row["name"]
        region = body.region if body.region is not None else row["region"]
        conn.execute("UPDATE sellers SET name = ?, region = ? WHERE id = ?", (name, region, seller_id))
        conn.commit()
        return Seller(id=seller_id, name=name, region=region)


@app.delete(
    "/sellers/{seller_id}",
    operation_id="delete_seller",
    summary="Exclui um vendedor",
    description="Não permite excluir vendedor com pedidos ativos.",
)
def delete_seller(seller_id: int) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM sellers WHERE id = ?", (seller_id,)).fetchone()
        if row is None:
            raise NotFound(
                error="seller_not_found",
                detail=f"Vendedor com id {seller_id} não existe.",
            )
        active_orders = conn.execute(
            "SELECT COUNT(*) FROM orders WHERE salesperson = ? AND status = 'ativo'",
            (row["name"],),
        ).fetchone()[0]
        if active_orders > 0:
            raise RuleViolation(
                error="seller_has_active_orders",
                detail=f"Vendedor '{row['name']}' possui {active_orders} pedido(s) ativo(s). Conclua ou cancele antes de excluir.",
                rule="não é permitido excluir vendedor com pedidos ativos",
            )
        conn.execute("DELETE FROM sellers WHERE id = ?", (seller_id,))
        conn.commit()
        return {"deleted": seller_id}


# ── Orders Update / Delete ────────────────────────────────────────────────


@app.put(
    "/orders/{order_id}",
    operation_id="update_order",
    summary="Atualiza quantidade, preço ou desconto de um pedido",
    description="Não permite alterar pedido já concluído ou cancelado. Regras de desconto são reaplicadas.",
)
def update_order(order_id: int, body: OrderUpdate) -> Order:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if row is None:
            raise NotFound(
                error="order_not_found",
                detail=f"Pedido com id {order_id} não existe.",
            )
        if row["status"] != "ativo":
            raise RuleViolation(
                error="order_not_editable",
                detail=f"Pedido {order_id} está '{row['status']}' e não pode ser alterado.",
                rule="só pedidos ativos podem ser alterados",
            )
        discount_pct = body.discount_pct if body.discount_pct is not None else row["discount_pct"]
        rules.validate_discount(discount_pct, row["role"])

        items_rows = conn.execute("SELECT * FROM order_items WHERE order_id = ? ORDER BY id", (order_id,)).fetchall()
        if body.quantity is not None or body.unit_price is not None:
            for item_row in items_rows:
                new_qty = body.quantity if body.quantity is not None else item_row["quantity"]
                new_price = body.unit_price if body.unit_price is not None else item_row["unit_price"]
                conn.execute(
                    "UPDATE order_items SET quantity = ?, unit_price = ? WHERE id = ?",
                    (new_qty, new_price, item_row["id"]),
                )

        updated_items = conn.execute(
            "SELECT * FROM order_items WHERE order_id = ? ORDER BY id", (order_id,)
        ).fetchall()
        totals = rules.order_totals(
            [{"quantity": r["quantity"], "unit_price": r["unit_price"]} for r in updated_items],
            discount_pct,
        )
        conn.execute(
            "UPDATE orders SET discount_pct = ?, gross_total = ?, net_total = ? WHERE id = ?",
            (discount_pct, totals["gross_total"], totals["net_total"], order_id),
        )
        conn.commit()
        return _load_order(conn, order_id)


@app.delete(
    "/orders/{order_id}",
    operation_id="delete_order",
    summary="Cancela/exclui um pedido",
    description="Não permite excluir pedido já concluído.",
)
def delete_order(order_id: int) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM orders WHERE id = ?", (order_id,)).fetchone()
        if row is None:
            raise NotFound(
                error="order_not_found",
                detail=f"Pedido com id {order_id} não existe.",
            )
        if row["status"] == "concluido":
            raise RuleViolation(
                error="order_already_completed",
                detail=f"Pedido {order_id} já foi concluído e não pode ser excluído.",
                rule="pedidos concluídos não podem ser excluídos",
            )
        conn.execute("UPDATE orders SET status = 'cancelado' WHERE id = ?", (order_id,))
        conn.commit()
        return {"cancelled": order_id}
