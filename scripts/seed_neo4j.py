"""Seed Neo4j com entidades dos microsserviços e relações cross-domain.

Uso:
    docker compose --profile graph up -d neo4j
    python -m scripts.seed_neo4j              # usa env vars do .env
    python -m scripts.seed_neo4j --uri bolt://localhost:7687 --user neo4j --password changeme

Idempotente: usa MERGE — pode rodar múltiplas vezes sem duplicar.
Alinhado com _EXPAND_CYPHER em gateway/knowledge_graph.py: todos os nós
usam label :Entity com propriedades {name, type, domain}.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Entidades extraídas dos seeds dos microsserviços ──────────────────────

# Estoque: produtos e categorias
_PRODUCTS = [
    ("CAD-ERG-001", "Cadeira ergonômica Presence", "Mobiliário"),
    ("MES-ELE-002", "Mesa elevatória Standy 140cm", "Mobiliário"),
    ("MON-27P-003", "Monitor 27\" 4K ProView", "Eletrônicos"),
    ("NTB-DEV-004", "Notebook Dev 32GB RAM", "Eletrônicos"),
    ("TEC-MEC-005", "Teclado mecânico ABNT2 Silent", "Periféricos"),
    ("MOU-ERG-006", "Mouse ergonômico vertical", "Periféricos"),
    ("HUB-USB-007", "Hub USB-C 8 portas", "Periféricos"),
    ("CAM-FHD-008", "Webcam Full HD com microfone", "Eletrônicos"),
    ("HEA-BTH-009", "Headset Bluetooth com cancelamento", "Periféricos"),
    ("SUP-NTB-010", "Suporte de notebook em alumínio", "Acessórios"),
]

_CATEGORIES = sorted({cat for _, _, cat in _PRODUCTS})

# RH: funcionários e departamentos
_EMPLOYEES = [
    ("Mariana Albuquerque", "Engenharia"),
    ("Carlos Eduardo Pontes", "Engenharia"),
    ("Fernanda Yoshida", "Engenharia"),
    ("Rafael Monteiro", "Vendas"),
    ("Juliana Castro Neves", "Vendas"),
    ("André Luiz Sampaio", "Financeiro"),
    ("Beatriz Fonseca", "Financeiro"),
    ("Thiago Nakamura", "RH"),
    ("Larissa Prado", "Engenharia"),
    ("Gustavo Bittencourt", "Operações"),
    ("Paula Souza", "Vendas"),
]

_DEPARTMENTS = sorted({dept for _, dept in _EMPLOYEES})

# Vendas: vendedores, clientes, e quais SKUs cada cliente comprou
_SELLERS = [
    ("Rafael Monteiro", "Sudeste"),
    ("Juliana Castro Neves", "Sul"),
]

# (cliente, vendedor, [skus comprados])
_ORDERS = [
    ("Lojas Andrade S.A.", "Rafael Monteiro", ["CAD-ERG-001", "MES-ELE-002"]),
    ("Cooperativa AgroVale", "Juliana Castro Neves", ["NTB-DEV-004"]),
    ("Clínica Bem Viver", "Rafael Monteiro", ["MON-27P-003", "TEC-MEC-005", "MOU-ERG-006"]),
    ("Banco Horizonte", "Juliana Castro Neves", ["MON-27P-003", "HEA-BTH-009"]),
    ("Transportadora Rocha", "Rafael Monteiro", ["CAM-FHD-008"]),
    ("Escritório Vasconcellos Advogados", "Rafael Monteiro", ["SUP-NTB-010", "HUB-USB-007"]),
    ("Colégio Santa Cecília", "Juliana Castro Neves", ["NTB-DEV-004", "MON-27P-003"]),
    ("Startup Maré Alta", "Rafael Monteiro", ["TEC-MEC-005", "MOU-ERG-006", "CAM-FHD-008"]),
]

_CUSTOMERS = sorted({c for c, _, _ in _ORDERS})

# Finanças: contas a pagar reais (espelham services/financas/seed.py).
# (descricao, fornecedor, valor, status) — a despesa vira nó; o fornecedor a EMITE.
_EXPENSES = [
    ("Aluguel do escritório", "Imobiliária Paulista Ltda", 8_500.00, "aberta"),
    ("Licenças de software anuais", "TechSoft Brasil S.A.", 3_200.00, "aberta"),
    ("Consultoria de implantação ERP", "Vetta Consultoria", 62_000.00, "aberta"),
    ("Energia elétrica", "Enel Distribuição SP", 1_840.50, "paga"),
    ("Material de escritório", "Kalunga Comércio", 487.90, "aberta"),
    ("Campanha de marketing digital", "Agência Bossa Nova", 24_000.00, "aberta"),
]

# Alçada de aprovação — espelha services/financas/rules.required_approver_role:
# ≤ R$5.000 auto-aprovada; ≤ R$50.000 exige 'gerente'; acima exige 'diretor'.
_AUTO_APPROVAL_LIMIT = 5_000.00
_MANAGER_APPROVAL_LIMIT = 50_000.00

# Cargos de aprovação (rh) — alvo das relações cross-domain de finanças.
_APPROVAL_ROLES = ("gerente", "diretor")

# Fornecedores que abastecem o estoque (cross-domain financas→estoque).
# (fornecedor, [categorias de estoque])
_SUPPLIES = [
    ("Kalunga Comércio", ["Periféricos", "Acessórios"]),
]

# Economia por SKU (espelha services/estoque/seed.py): preço, saldo, ponto de reposição.
# sku -> (unit_price, on_hand, reorder_point)
_PRODUCT_STOCK = {
    "CAD-ERG-001": (1_450.00, 32, 10),
    "MES-ELE-002": (2_280.00, 8, 12),
    "MON-27P-003": (2_150.00, 45, 15),
    "NTB-DEV-004": (9_800.00, 6, 8),
    "TEC-MEC-005": (520.00, 120, 30),
    "MOU-ERG-006": (310.00, 75, 25),
    "HUB-USB-007": (420.00, 18, 20),
    "CAM-FHD-008": (480.00, 54, 15),
    "HEA-BTH-009": (890.00, 40, 12),
    "SUP-NTB-010": (180.00, 200, 50),
}

# Reservas ativas (espelha services/estoque/seed.py) p/ disponível e abaixo do ponto.
_RESERVATIONS = {"MON-27P-003": 10, "NTB-DEV-004": 2, "TEC-MEC-005": 20}


def _required_approver(amount: float) -> str | None:
    """Alçada mínima exigida para a despesa (espelha o serviço Finanças)."""
    if amount <= _AUTO_APPROVAL_LIMIT:
        return None
    if amount <= _MANAGER_APPROVAL_LIMIT:
        return "gerente"
    return "diretor"

_COUNTERPARTIES_RECEBER = [
    "Lojas Andrade S.A.",
    "Cooperativa AgroVale",
    "Clínica Bem Viver",
    "Transportadora Rocha",
    "Banco Horizonte",
]

# ── Cypher statements ────────────────────────────────────────────────────

# Constraint garante unicidade e cria índice implícito.
_CONSTRAINT = (
    "CREATE CONSTRAINT entity_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.name, e.type, e.domain) IS UNIQUE"
)

_MERGE_ENTITY = (
    "MERGE (e:Entity {name: $name, type: $type, domain: $domain}) "
    "SET e.sku = $sku"
)

_MERGE_REL = (
    "MATCH (a:Entity {name: $a_name, type: $a_type, domain: $a_domain}) "
    "MATCH (b:Entity {name: $b_name, type: $b_type, domain: $b_domain}) "
    "MERGE (a)-[r:%s]->(b)"  # rel type injetado via format (nomes fixos, sem input externo)
)


def _seed(session) -> dict[str, int]:  # noqa: ANN001
    """Popula Neo4j. Retorna contagem de entidades e relações criadas."""
    counts: dict[str, int] = {"entities": 0, "relations": 0}

    # ── Constraint + índices ──────────────────────────────────────────
    session.run(_CONSTRAINT)
    session.run("CREATE INDEX entity_sku IF NOT EXISTS FOR (e:Entity) ON (e.sku)")

    # ── Categorias (estoque) ──────────────────────────────────────────
    for cat in _CATEGORIES:
        session.run(_MERGE_ENTITY, name=cat, type="categoria", domain="estoque", sku=None)
        counts["entities"] += 1

    # ── Produtos (estoque) ────────────────────────────────────────────
    for sku, name, category in _PRODUCTS:
        session.run(_MERGE_ENTITY, name=name, type="produto", domain="estoque", sku=sku)
        counts["entities"] += 1
        session.run(
            _MERGE_REL % "PERTENCE_A",
            a_name=name, a_type="produto", a_domain="estoque",
            b_name=category, b_type="categoria", b_domain="estoque",
        )
        counts["relations"] += 1

    # ── Economia por produto (estoque): preço, saldo, ponto de reposição ──
    # Espelha services/estoque: disponível = on_hand − reservas; abaixo do ponto
    # quando disponível < reorder_point. Enriquece o nó para o expand_context.
    for sku, (price, on_hand, reorder) in _PRODUCT_STOCK.items():
        reserved = _RESERVATIONS.get(sku, 0)
        available = on_hand - reserved
        session.run(
            "MATCH (e:Entity {sku: $sku, type: 'produto', domain: 'estoque'}) "
            "SET e.unit_price = $price, e.on_hand = $on_hand, e.reserved = $reserved, "
            "e.available = $available, e.reorder_point = $reorder, e.below_reorder = $below",
            sku=sku, price=price, on_hand=on_hand, reserved=reserved,
            available=available, reorder=reorder, below=available < reorder,
        )

    # ── Departamentos (rh) ────────────────────────────────────────────
    for dept in _DEPARTMENTS:
        session.run(_MERGE_ENTITY, name=dept, type="departamento", domain="rh", sku=None)
        counts["entities"] += 1

    # ── Funcionários (rh) ─────────────────────────────────────────────
    for name, dept in _EMPLOYEES:
        session.run(_MERGE_ENTITY, name=name, type="funcionario", domain="rh", sku=None)
        counts["entities"] += 1
        session.run(
            _MERGE_REL % "TRABALHA_EM",
            a_name=name, a_type="funcionario", a_domain="rh",
            b_name=dept, b_type="departamento", b_domain="rh",
        )
        counts["relations"] += 1

    # ── Vendedores (vendas) ───────────────────────────────────────────
    for name, region in _SELLERS:
        session.run(
            "MERGE (e:Entity {name: $name, type: $type, domain: $domain}) SET e.region = $region",
            name=name, type="vendedor", domain="vendas", region=region,
        )
        counts["entities"] += 1
        # Cross-domain: vendedor (vendas) → funcionário (rh) — mesma pessoa.
        session.run(
            _MERGE_REL % "MESMO_QUE",
            a_name=name, a_type="vendedor", a_domain="vendas",
            b_name=name, b_type="funcionario", b_domain="rh",
        )
        counts["relations"] += 1

    # ── Clientes (vendas) ─────────────────────────────────────────────
    for customer in _CUSTOMERS:
        session.run(_MERGE_ENTITY, name=customer, type="cliente", domain="vendas", sku=None)
        counts["entities"] += 1

    # ── Pedidos: cliente→produto, vendedor→cliente ────────────────────
    seen_comprou: set[tuple[str, str]] = set()
    seen_vendeu: set[tuple[str, str]] = set()

    sku_to_name = {sku: name for sku, name, _ in _PRODUCTS}

    for customer, seller, skus in _ORDERS:
        # Vendedor → cliente
        if (seller, customer) not in seen_vendeu:
            session.run(
                _MERGE_REL % "VENDEU_PARA",
                a_name=seller, a_type="vendedor", a_domain="vendas",
                b_name=customer, b_type="cliente", b_domain="vendas",
            )
            counts["relations"] += 1
            seen_vendeu.add((seller, customer))

        # Cliente → produto (cross-domain vendas→estoque)
        for sku in skus:
            product_name = sku_to_name[sku]
            if (customer, product_name) not in seen_comprou:
                session.run(
                    _MERGE_REL % "COMPROU",
                    a_name=customer, a_type="cliente", a_domain="vendas",
                    b_name=product_name, b_type="produto", b_domain="estoque",
                )
                counts["relations"] += 1
                seen_comprou.add((customer, product_name))

    # ── Cargos de aprovação (rh) — alvo de relações cross-domain de finanças ──
    for role in _APPROVAL_ROLES:
        session.run(_MERGE_ENTITY, name=role, type="cargo", domain="rh", sku=None)
        counts["entities"] += 1

    # ── Contrapartes (finanças) — fornecedores + despesas (contas a pagar) ──
    # Cada fornecedor EMITE uma despesa; despesa acima da alçada REQUER_APROVACAO
    # de um cargo (rh) — relação cross-domain financas→rh com a regra de negócio real.
    for descricao, fornecedor, valor, status in _EXPENSES:
        session.run(_MERGE_ENTITY, name=fornecedor, type="fornecedor", domain="financas", sku=None)
        counts["entities"] += 1
        session.run(
            "MERGE (d:Entity {name: $name, type: 'despesa', domain: 'financas'}) "
            "SET d.amount = $valor, d.status = $status",
            name=descricao, valor=valor, status=status,
        )
        counts["entities"] += 1
        session.run(
            _MERGE_REL % "EMITE",
            a_name=fornecedor, a_type="fornecedor", a_domain="financas",
            b_name=descricao, b_type="despesa", b_domain="financas",
        )
        counts["relations"] += 1
        required = _required_approver(valor)
        if required is not None:
            session.run(
                _MERGE_REL % "REQUER_APROVACAO",
                a_name=descricao, a_type="despesa", a_domain="financas",
                b_name=required, b_type="cargo", b_domain="rh",
            )
            counts["relations"] += 1

    # ── Fornecedores que abastecem o estoque (cross-domain financas→estoque) ──
    for fornecedor, categorias in _SUPPLIES:
        for categoria in categorias:
            session.run(
                _MERGE_REL % "ABASTECE",
                a_name=fornecedor, a_type="fornecedor", a_domain="financas",
                b_name=categoria, b_type="categoria", b_domain="estoque",
            )
            counts["relations"] += 1

    # ── Contrapartes (finanças) — clientes recebíveis ─────────────────
    # Cross-domain: contraparte recebível (financas) → cliente (vendas)
    for name in _COUNTERPARTIES_RECEBER:
        session.run(_MERGE_ENTITY, name=name, type="cliente_recebivel", domain="financas", sku=None)
        counts["entities"] += 1
        # Se existe como cliente em vendas, cria relação cross-domain.
        if name in _CUSTOMERS:
            session.run(
                _MERGE_REL % "MESMO_QUE",
                a_name=name, a_type="cliente_recebivel", a_domain="financas",
                b_name=name, b_type="cliente", b_domain="vendas",
            )
            counts["relations"] += 1

    return counts


def main(uri: str, user: str, password: str) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("neo4j driver não instalado. pip install neo4j")
        sys.exit(1)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        logger.info("Conectado a %s", uri)
        with driver.session() as session:
            counts = _seed(session)
        logger.info(
            "Seed concluído: %d entidades, %d relações.",
            counts["entities"], counts["relations"],
        )
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Popula Neo4j com entidades dos microsserviços.")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "changeme"))
    args = parser.parse_args()
    main(args.uri, args.user, args.password)
