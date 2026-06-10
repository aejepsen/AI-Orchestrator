"""Serviço Estoque: produtos/SKUs, saldo, reservas e ponto de reposição."""

from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from common import NotFound, RuleViolation, register_error_handlers, register_internal_auth
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
