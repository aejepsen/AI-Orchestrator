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

# Finanças: contrapartes (pagar/receber)
_COUNTERPARTIES_PAGAR = [
    "Imobiliária Paulista Ltda",
    "TechSoft Brasil S.A.",
    "Vetta Consultoria",
    "Enel Distribuição SP",
    "Kalunga Comércio",
    "Agência Bossa Nova",
]

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

    # ── Contrapartes (finanças) — fornecedores ────────────────────────
    for name in _COUNTERPARTIES_PAGAR:
        session.run(_MERGE_ENTITY, name=name, type="fornecedor", domain="financas", sku=None)
        counts["entities"] += 1

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
