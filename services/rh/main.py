"""Serviço RH: funcionários, férias (CLT simplificada), reembolsos e headcount."""

from contextlib import asynccontextmanager
from datetime import date
from typing import AsyncIterator, Literal

from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

from common import NotFound, register_error_handlers, register_internal_auth
from rh import db, rules, seed


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    with db.connect() as conn:
        db.init_schema(conn)
        seed.seed(conn)
    yield


app = FastAPI(
    title="Serviço de RH",
    description=(
        "Funcionários, férias (30 dias após 12 meses de admissão, fracionáveis em até 3 períodos, "
        "um deles ≥14 dias), reembolsos com limite por categoria e headcount por departamento."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
register_error_handlers(app)
register_internal_auth(app)

ReimbursementCategory = Literal["viagem", "refeicao", "home_office"]


class Employee(BaseModel):
    id: int
    name: str
    department: str
    position: str
    hire_date: date
    salary: float


class VacationRequest(BaseModel):
    employee_id: int = Field(description="Id do funcionário")
    start_date: date = Field(description="Início do período de férias (YYYY-MM-DD)")
    days: int = Field(gt=0, description="Duração do período em dias corridos")
    acquisition_year: int = Field(description="Ano aquisitivo ao qual o período pertence (ex.: 2026)")


class Vacation(VacationRequest):
    id: int


class VacationBalance(BaseModel):
    employee_id: int
    acquisition_year: int
    entitled_days: int
    taken_days: int
    balance: int
    periods: list[Vacation]


class ReimbursementRequest(BaseModel):
    employee_id: int = Field(description="Id do funcionário")
    category: ReimbursementCategory = Field(description="Categoria: viagem, refeicao ou home_office")
    amount: float = Field(gt=0, description="Valor solicitado em reais")
    days: int | None = Field(default=None, gt=0, description="Dias cobertos (obrigatório para refeicao)")
    month: str = Field(pattern=r"^\d{4}-\d{2}$", description="Mês de competência (YYYY-MM)")


class Reimbursement(ReimbursementRequest):
    id: int


class Headcount(BaseModel):
    department: str
    headcount: int


class Health(BaseModel):
    status: str
    service: str


def _get_employee_row(conn, employee_id: int):
    row = conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
    if row is None:
        raise NotFound(
            error="employee_not_found",
            detail=f"Funcionário com id {employee_id} não existe. Liste em GET /employees para ver os ids válidos.",
        )
    return row


def _row_to_employee(row) -> Employee:
    return Employee(
        id=row["id"],
        name=row["name"],
        department=row["department"],
        position=row["position"],
        hire_date=date.fromisoformat(row["hire_date"]),
        salary=row["salary"],
    )


@app.get("/health", operation_id="health_check", summary="Verifica se o serviço está no ar")
def health() -> Health:
    return Health(status="ok", service="rh")


@app.get(
    "/employees",
    operation_id="list_employees",
    summary="Lista funcionários",
    description="Retorna todos os funcionários, com filtro opcional por departamento.",
)
def list_employees(department: str | None = Query(default=None, description="Filtra por departamento")) -> list[Employee]:
    query, params = "SELECT * FROM employees", []
    if department is not None:
        query += " WHERE department = ?"
        params.append(department)
    with db.connect() as conn:
        return [_row_to_employee(r) for r in conn.execute(query + " ORDER BY name", params)]


@app.get("/employees/{employee_id}", operation_id="get_employee", summary="Detalha um funcionário pelo id")
def get_employee(employee_id: int) -> Employee:
    with db.connect() as conn:
        return _row_to_employee(_get_employee_row(conn, employee_id))


@app.get(
    "/employees/{employee_id}/vacation-balance",
    operation_id="get_vacation_balance",
    summary="Saldo de férias do funcionário em um ano aquisitivo",
    description="Retorna dias de direito (30), dias usados e saldo do ano aquisitivo informado (default: ano atual).",
)
def get_vacation_balance(
    employee_id: int,
    acquisition_year: int | None = Query(default=None, description="Ano aquisitivo; default = ano corrente"),
) -> VacationBalance:
    year = acquisition_year or date.today().year
    with db.connect() as conn:
        _get_employee_row(conn, employee_id)
        rows = conn.execute(
            "SELECT * FROM vacations WHERE employee_id = ? AND acquisition_year = ? ORDER BY start_date",
            (employee_id, year),
        ).fetchall()
    periods = [
        Vacation(
            id=r["id"],
            employee_id=r["employee_id"],
            acquisition_year=r["acquisition_year"],
            start_date=date.fromisoformat(r["start_date"]),
            days=r["days"],
        )
        for r in rows
    ]
    taken = sum(p.days for p in periods)
    return VacationBalance(
        employee_id=employee_id,
        acquisition_year=year,
        entitled_days=rules.VACATION_DAYS_PER_YEAR,
        taken_days=taken,
        balance=rules.VACATION_DAYS_PER_YEAR - taken,
        periods=periods,
    )


@app.post(
    "/vacations",
    operation_id="request_vacation",
    summary="Solicita um período de férias",
    description=(
        "Valida elegibilidade (12 meses de admissão), fracionamento (máx. 3 períodos, um ≥14 dias, "
        "demais ≥5 dias) e saldo de 30 dias do ano aquisitivo. Violação de regra retorna 422."
    ),
    status_code=201,
)
def request_vacation(body: VacationRequest) -> Vacation:
    with db.connect() as conn:
        employee = _get_employee_row(conn, body.employee_id)
        taken = [
            r["days"]
            for r in conn.execute(
                "SELECT days FROM vacations WHERE employee_id = ? AND acquisition_year = ?",
                (body.employee_id, body.acquisition_year),
            )
        ]
        rules.validate_vacation_request(
            hire_date=date.fromisoformat(employee["hire_date"]),
            start_date=body.start_date,
            days=body.days,
            taken_days=taken,
        )
        cur = conn.execute(
            "INSERT INTO vacations (employee_id, acquisition_year, start_date, days) VALUES (?, ?, ?, ?)",
            (body.employee_id, body.acquisition_year, body.start_date.isoformat(), body.days),
        )
        conn.commit()
        return Vacation(id=cur.lastrowid, **body.model_dump())


@app.post(
    "/reimbursements",
    operation_id="request_reimbursement",
    summary="Solicita um reembolso",
    description=(
        "Limites por categoria: viagem R$3.000 por pedido; refeicao R$100 por dia (campo days obrigatório); "
        "home_office R$500 acumulados por mês. Acima do limite retorna 422."
    ),
    status_code=201,
)
def request_reimbursement(body: ReimbursementRequest) -> Reimbursement:
    with db.connect() as conn:
        _get_employee_row(conn, body.employee_id)
        month_total = conn.execute(
            "SELECT COALESCE(SUM(amount), 0) FROM reimbursements"
            " WHERE employee_id = ? AND category = 'home_office' AND month = ?",
            (body.employee_id, body.month),
        ).fetchone()[0]
        rules.validate_reimbursement(body.category, body.amount, body.days, month_total)
        cur = conn.execute(
            "INSERT INTO reimbursements (employee_id, category, amount, days, month) VALUES (?, ?, ?, ?, ?)",
            (body.employee_id, body.category, body.amount, body.days, body.month),
        )
        conn.commit()
        return Reimbursement(id=cur.lastrowid, **body.model_dump())


@app.get(
    "/headcount",
    operation_id="get_headcount",
    summary="Headcount por departamento",
    description="Conta funcionários por departamento; filtro opcional por um departamento específico.",
)
def get_headcount(department: str | None = Query(default=None, description="Filtra por departamento")) -> list[Headcount]:
    query = "SELECT department, COUNT(*) AS headcount FROM employees"
    params = []
    if department is not None:
        query += " WHERE department = ?"
        params.append(department)
    query += " GROUP BY department ORDER BY department"
    with db.connect() as conn:
        return [Headcount(department=r["department"], headcount=r["headcount"]) for r in conn.execute(query, params)]
