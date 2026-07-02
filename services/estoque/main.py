"""Serviço Estoque: produtos/SKUs, saldo, reservas e ponto de reposição."""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from common import (
    NotFound,
    RuleViolation,
    register_admin_reset,
    register_error_handlers,
    register_internal_auth,
)
from sqlite3 import IntegrityError
from estoque import db, rules, seed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    with db.connect() as conn:
        db.init_schema(conn)
        seed.seed(conn)
    yield


app = FastAPI(
    title="Serviço de Estoque",
    description=(
        "Produtos por SKU com saldo físico, reservas e ponto de reposição. "
        "Reservas são limitadas ao disponível (estoque físico menos reservas ativas)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
register_error_handlers(app)
register_internal_auth(app)
register_admin_reset(app, service="estoque", connect=db.connect, init_schema=db.init_schema, seed=seed.seed)


class ProductCreate(BaseModel):
    sku: str = Field(min_length=1, description="SKU único do produto")
    name: str = Field(min_length=1, description="Nome do produto")
    category: str = Field(default="Geral", description="Categoria do produto")
    unit_price: float = Field(default=0.01, gt=0, description="Preço unitário")
    quantity: int = Field(ge=0, description="Quantidade física em estoque")
    reorder_point: int = Field(ge=0, description="Ponto de reposição")


class ProductUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, description="Nome do produto")
    quantity: int | None = Field(default=None, ge=0, description="Quantidade física em estoque")
    reorder_point: int | None = Field(default=None, ge=0, description="Ponto de reposição")


class Product(BaseModel):
    sku: str
    name: str
    category: str
    unit_price: float
    on_hand: int = Field(description="Quantidade física em estoque")
    reserved: int = Field(description="Quantidade em reservas ativas")
    available: int = Field(description="Disponível para reserva (on_hand - reserved)")
    reorder_point: int = Field(description="Ponto de reposição")


class ReservationRequest(BaseModel):
    sku: str = Field(description="SKU do produto a reservar")
    quantity: int = Field(gt=0, description="Quantidade a reservar")


class Reservation(ReservationRequest):
    id: int
    status: Literal["ativa", "liberada"]


class ReplenishmentItem(BaseModel):
    sku: str
    name: str
    available: int
    reorder_point: int
    suggested_quantity: int = Field(description="Quantidade sugerida de compra (reorder_point*2 - disponível)")


class Health(BaseModel):
    status: str
    service: str


def _get_product_row(conn, sku: str):
    row = conn.execute("SELECT * FROM products WHERE sku = ?", (sku,)).fetchone()
    if row is None:
        raise NotFound(
            error="product_not_found",
            detail=f"SKU '{sku}' não existe. Liste os produtos em GET /products para ver os SKUs válidos.",
        )
    return row


def _row_to_product(conn, row) -> Product:
    reserved = db.reserved_quantity(conn, row["sku"])
    return Product(
        sku=row["sku"],
        name=row["name"],
        category=row["category"],
        unit_price=row["unit_price"],
        on_hand=row["on_hand"],
        reserved=reserved,
        available=rules.available(row["on_hand"], reserved),
        reorder_point=row["reorder_point"],
    )


@app.get("/health", operation_id="health_check", summary="Verifica se o serviço está no ar")
def health() -> Health:
    return Health(status="ok", service="estoque")


@app.get(
    "/products",
    operation_id="list_products",
    summary="Lista produtos com saldo e disponibilidade",
)
def list_products() -> list[Product]:
    with db.connect() as conn:
        return [_row_to_product(conn, r) for r in conn.execute("SELECT * FROM products ORDER BY sku")]


@app.get(
    "/products/{sku}",
    operation_id="get_product",
    summary="Detalha um produto pelo SKU, incluindo saldo disponível",
)
def get_product(sku: str) -> Product:
    with db.connect() as conn:
        return _row_to_product(conn, _get_product_row(conn, sku))


@app.get(
    "/reservations",
    operation_id="list_reservations",
    summary="Lista reservas de estoque",
)
def list_reservations() -> list[Reservation]:
    with db.connect() as conn:
        return [
            Reservation(id=r["id"], sku=r["sku"], quantity=r["quantity"], status=r["status"])
            for r in conn.execute("SELECT * FROM reservations ORDER BY id")
        ]


@app.post(
    "/reservations",
    operation_id="create_reservation",
    summary="Reserva quantidade de um SKU",
    description="Cria uma reserva ativa. Reservar acima do disponível (on_hand - reserved) retorna 422.",
    status_code=201,
)
def create_reservation(body: ReservationRequest) -> Reservation:
    with db.connect() as conn:
        product = _get_product_row(conn, body.sku)
        reserved = db.reserved_quantity(conn, body.sku)
        rules.validate_reservation(body.sku, product["on_hand"], reserved, body.quantity)
        cur = conn.execute("INSERT INTO reservations (sku, quantity) VALUES (?, ?)", (body.sku, body.quantity))
        conn.commit()
        return Reservation(id=cur.lastrowid, sku=body.sku, quantity=body.quantity, status="ativa")


@app.post(
    "/reservations/{reservation_id}/release",
    operation_id="release_reservation",
    summary="Libera uma reserva ativa, devolvendo a quantidade ao disponível",
)
def release_reservation(reservation_id: int) -> Reservation:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM reservations WHERE id = ?", (reservation_id,)).fetchone()
        if row is None:
            raise NotFound(
                error="reservation_not_found",
                detail=f"Reserva com id {reservation_id} não existe. Liste em GET /reservations.",
            )
        if row["status"] != "ativa":
            raise RuleViolation(
                error="reservation_already_released",
                detail=f"Reserva {reservation_id} já foi liberada e não pode ser liberada novamente.",
                rule="só reservas ativas podem ser liberadas",
            )
        conn.execute("UPDATE reservations SET status = 'liberada' WHERE id = ?", (reservation_id,))
        conn.commit()
        return Reservation(id=row["id"], sku=row["sku"], quantity=row["quantity"], status="liberada")


@app.get(
    "/replenishment",
    operation_id="get_replenishment",
    summary="SKUs abaixo do ponto de reposição com quantidade sugerida de compra",
    description="Retorna os SKUs cujo disponível está abaixo do reorder_point e a quantidade sugerida para repor.",
)
def get_replenishment() -> list[ReplenishmentItem]:
    items = []
    with db.connect() as conn:
        for row in conn.execute("SELECT * FROM products ORDER BY sku"):
            reserved = db.reserved_quantity(conn, row["sku"])
            suggested = rules.replenishment_suggestion(row["on_hand"], reserved, row["reorder_point"])
            if suggested is not None:
                items.append(
                    ReplenishmentItem(
                        sku=row["sku"],
                        name=row["name"],
                        available=rules.available(row["on_hand"], reserved),
                        reorder_point=row["reorder_point"],
                        suggested_quantity=suggested,
                    )
                )
    return items


@app.post(
    "/products",
    operation_id="create_product",
    summary="Cria um novo produto no catálogo",
    description="SKU deve ser único. Retorna 422 se o SKU já existir.",
    status_code=201,
)
def create_product(body: ProductCreate) -> Product:
    with db.connect() as conn:
        try:
            conn.execute(
                "INSERT INTO products (sku, name, category, unit_price, on_hand, reorder_point)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (body.sku, body.name, body.category, body.unit_price, body.quantity, body.reorder_point),
            )
            conn.commit()
        except IntegrityError:
            raise RuleViolation(
                error="sku_already_exists",
                detail=f"SKU '{body.sku}' já existe no catálogo. Escolha um SKU diferente.",
                rule="SKU deve ser único",
            )
        return _row_to_product(conn, _get_product_row(conn, body.sku))


@app.put(
    "/products/{sku}",
    operation_id="update_product",
    summary="Atualiza nome, quantidade ou ponto de reposição de um produto",
)
def update_product(sku: str, body: ProductUpdate) -> Product:
    with db.connect() as conn:
        row = _get_product_row(conn, sku)
        name = body.name if body.name is not None else row["name"]
        on_hand = body.quantity if body.quantity is not None else row["on_hand"]
        reorder_point = body.reorder_point if body.reorder_point is not None else row["reorder_point"]
        conn.execute(
            "UPDATE products SET name = ?, on_hand = ?, reorder_point = ? WHERE sku = ?",
            (name, on_hand, reorder_point, sku),
        )
        conn.commit()
        return _row_to_product(conn, _get_product_row(conn, sku))


@app.delete(
    "/products/{sku}",
    operation_id="delete_product",
    summary="Exclui um produto do catálogo",
    description="Não permite excluir produto com reservas ativas.",
)
def delete_product(sku: str) -> dict:
    with db.connect() as conn:
        _get_product_row(conn, sku)
        active = db.reserved_quantity(conn, sku)
        if active > 0:
            raise RuleViolation(
                error="product_has_active_reservations",
                detail=f"Produto '{sku}' possui {active} unidade(s) em reservas ativas. Libere as reservas antes de excluir.",
                rule="não é permitido excluir produto com reservas ativas",
            )
        conn.execute("DELETE FROM reservations WHERE sku = ?", (sku,))
        conn.execute("DELETE FROM products WHERE sku = ?", (sku,))
        conn.commit()
        return {"deleted": sku}
