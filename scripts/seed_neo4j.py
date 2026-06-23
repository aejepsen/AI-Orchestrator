"""Seed Neo4j com entidades enriquecidas dos microsserviços e relações cross-domain.

Escala: ~320 nós, ~180 relações — simula operação multi-tenant pronta para crescer.

Uso:
    docker compose --profile graph up -d neo4j
    ./.venv/bin/python -m scripts.seed_neo4j              # usa env vars do .env
    ./.venv/bin/python -m scripts.seed_neo4j --scale 3    # 3x a base (modo stress)

Idempotente: usa MERGE — pode rodar múltiplas vezes sem duplicar.
Alinhado com _EXPAND_CYPHER em gateway/knowledge_graph.py: todos os nós
usam label :Entity com propriedades {name, type, domain, sku, created_at}.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_NOW = datetime.utcnow()

# ── Estoque ────────────────────────────────────────────────────────────────

# Categorias (hierárquicas: categoria → subcategoria)
_CATEGORIES = [
    ("Mobiliário", None),
    ("Eletrônicos", None),
    ("Periféricos", None),
    ("Acessórios", None),
    ("Suprimentos", None),
    ("Equipamentos de Rede", "Eletrônicos"),
    ("Áudio e Vídeo", "Eletrônicos"),
]

# Produtos expandidos: (sku, name, category, unit_price, on_hand, reorder_point)
_PRODUCTS = [
    # Mobiliário
    ("CAD-ERG-001", "Cadeira ergonômica Presence", "Mobiliário", 1_450.00, 32, 10),
    ("MES-ELE-002", "Mesa elevatória Standy 140cm", "Mobiliário", 2_280.00, 8, 12),
    ("CAD-DIR-011", "Cadeira diretor couro legítimo", "Mobiliário", 3_890.00, 5, 3),
    ("MES-CON-012", "Mesa de conferência 12 lugares", "Mobiliário", 7_500.00, 2, 2),
    ("ARM-ARQ-013", "Armário arquivo 4 gavetas", "Mobiliário", 1_120.00, 15, 8),
    ("EST-MET-014", "Estante metálica 5 prateleiras", "Mobiliário", 890.00, 22, 10),
    # Eletrônicos
    ("MON-27P-003", "Monitor 27\" 4K ProView", "Eletrônicos", 2_150.00, 45, 15),
    ("NTB-DEV-004", "Notebook Dev 32GB RAM", "Eletrônicos", 9_800.00, 6, 8),
    ("CAM-FHD-008", "Webcam Full HD com microfone", "Eletrônicos", 480.00, 54, 15),
    ("MON-32W-015", "Monitor 32\" ultrawide curvo", "Eletrônicos", 4_200.00, 12, 8),
    ("NTB-EXE-016", "Notebook executivo 16GB i7", "Eletrônicos", 6_500.00, 10, 6),
    ("TAB-DIG-017", "Tablet digitalizadora Wacom M", "Eletrônicos", 1_850.00, 14, 8),
    ("PRO-MUL-018", "Projetor multimídia 4K portátil", "Eletrônicos", 3_600.00, 5, 4),
    ("SSD-EXT-019", "SSD externo 2TB NVMe", "Eletrônicos", 780.00, 35, 10),
    # Equipamentos de Rede (subcategoria)
    ("SWT-24P-020", "Switch gerenciável 24 portas PoE", "Equipamentos de Rede", 2_400.00, 8, 5),
    ("RTR-VPN-021", "Roteador VPN corporativo", "Equipamentos de Rede", 5_200.00, 3, 3),
    ("AP-WF6-022", "Access Point WiFi 6 enterprise", "Equipamentos de Rede", 1_680.00, 12, 6),
    # Áudio e Vídeo (subcategoria)
    ("SPK-CON-023", "Caixa de som conferência USB", "Áudio e Vídeo", 940.00, 18, 8),
    ("MIC-LAP-024", "Microfone lapela wireless", "Áudio e Vídeo", 650.00, 22, 10),
    # Periféricos
    ("TEC-MEC-005", "Teclado mecânico ABNT2 Silent", "Periféricos", 520.00, 120, 30),
    ("MOU-ERG-006", "Mouse ergonômico vertical", "Periféricos", 310.00, 75, 25),
    ("HUB-USB-007", "Hub USB-C 8 portas", "Periféricos", 420.00, 18, 20),
    ("HEA-BTH-009", "Headset Bluetooth com cancelamento", "Periféricos", 890.00, 40, 12),
    ("TEC-ERG-025", "Teclado ergonômico split", "Periféricos", 720.00, 28, 15),
    ("MOU-TRK-026", "Trackball ambidestro", "Periféricos", 380.00, 20, 12),
    ("DOK-USB-027", "Docking station USB-C triplo", "Periféricos", 1_450.00, 15, 10),
    # Acessórios
    ("SUP-NTB-010", "Suporte de notebook em alumínio", "Acessórios", 180.00, 200, 50),
    ("FILT-TEL-028", "Filtro de privacidade 27\"", "Acessórios", 240.00, 45, 20),
    ("CAB-HDM-029", "Cabo HDMI 4K 3m", "Acessórios", 65.00, 150, 40),
    ("MOCH-NTB-030", "Mochila notebook impermeável", "Acessórios", 290.00, 55, 20),
    # Suprimentos
    ("TON-CIA-031", "Toner ciano impressora laser", "Suprimentos", 420.00, 25, 12),
    ("PAP-A4R-032", "Papel A4 reciclado 500fl cx", "Suprimentos", 28.00, 200, 80),
    ("CAN-ESC-033", "Caneta esferográfica azul cx50", "Suprimentos", 45.00, 300, 100),
    ("PIL-REC-034", "Pilha recarregável AAA 4un", "Suprimentos", 85.00, 90, 30),
]

# Depósitos (cross-domain estoque→logística)
_WAREHOUSES = [
    ("Depósito Central SP", "São Paulo", 5_000),
    ("Centro de Distribuição RJ", "Rio de Janeiro", 2_500),
    ("Armazém POA", "Porto Alegre", 1_200),
]

# Alocação de produtos a depósitos: (sku, warehouse, qty)
_STOCK_ALLOCATION = [
    ("CAD-ERG-001", "Depósito Central SP", 20),
    ("CAD-ERG-001", "Centro de Distribuição RJ", 8),
    ("CAD-ERG-001", "Armazém POA", 4),
    ("MES-ELE-002", "Depósito Central SP", 5),
    ("MES-ELE-002", "Armazém POA", 3),
    ("MON-27P-003", "Depósito Central SP", 25),
    ("MON-27P-003", "Centro de Distribuição RJ", 15),
    ("NTB-DEV-004", "Depósito Central SP", 6),
    ("TEC-MEC-005", "Depósito Central SP", 60),
    ("TEC-MEC-005", "Centro de Distribuição RJ", 40),
    ("HUB-USB-007", "Centro de Distribuição RJ", 10),
    ("CAM-FHD-008", "Depósito Central SP", 30),
    ("HEA-BTH-009", "Armazém POA", 25),
    ("SUP-NTB-010", "Depósito Central SP", 100),
    ("SUP-NTB-010", "Centro de Distribuição RJ", 60),
    ("CAB-HDM-029", "Armazém POA", 80),
    ("PAP-A4R-032", "Depósito Central SP", 120),
    ("PIL-REC-034", "Centro de Distribuição RJ", 50),
]

# ── RH ─────────────────────────────────────────────────────────────────────

_DEPARTMENTS = [
    ("Engenharia", "Tecnologia"),
    ("Vendas", "Comercial"),
    ("Financeiro", "Administrativo"),
    ("RH", "Administrativo"),
    ("Operações", "Administrativo"),
    ("Marketing", "Comercial"),
    ("Suporte Técnico", "Tecnologia"),
]

_CARGOS = [
    ("estagiário", 2),
    ("analista", 3),
    ("especialista", 4),
    ("coordenador", 5),
    ("gerente", 6),
    ("diretor", 7),
]

# Funcionários expandidos: (name, dept, cargo, salario, data_contratacao)
_EMPLOYEES = [
    ("Mariana Albuquerque", "Engenharia", "especialista", 18_500.00, _NOW - timedelta(days=900)),
    ("Carlos Eduardo Pontes", "Engenharia", "analista", 12_000.00, _NOW - timedelta(days=600)),
    ("Fernanda Yoshida", "Engenharia", "coordenador", 22_000.00, _NOW - timedelta(days=1_200)),
    ("Larissa Prado", "Engenharia", "estagiário", 2_500.00, _NOW - timedelta(days=180)),
    ("Rafael Monteiro", "Vendas", "gerente", 16_000.00, _NOW - timedelta(days=500)),
    ("Juliana Castro Neves", "Vendas", "especialista", 14_000.00, _NOW - timedelta(days=700)),
    ("Paula Souza", "Vendas", "analista", 10_500.00, _NOW - timedelta(days=300)),
    ("André Luiz Sampaio", "Financeiro", "diretor", 28_000.00, _NOW - timedelta(days=1_500)),
    ("Beatriz Fonseca", "Financeiro", "analista", 11_000.00, _NOW - timedelta(days=450)),
    ("Thiago Nakamura", "RH", "coordenador", 15_000.00, _NOW - timedelta(days=850)),
    ("Gustavo Bittencourt", "Operações", "gerente", 17_000.00, _NOW - timedelta(days=1_000)),
    # Novos funcionários
    ("Renata Oliveira", "Engenharia", "analista", 12_500.00, _NOW - timedelta(days=200)),
    ("Marcos Vinícius", "Engenharia", "especialista", 19_000.00, _NOW - timedelta(days=950)),
    ("Camila Rocha", "Vendas", "analista", 9_800.00, _NOW - timedelta(days=150)),
    ("Felipe Andrade", "Vendas", "estagiário", 2_200.00, _NOW - timedelta(days=90)),
    ("Amanda Torres", "Financeiro", "coordenador", 18_000.00, _NOW - timedelta(days=600)),
    ("Leonardo Campos", "Financeiro", "analista", 10_000.00, _NOW - timedelta(days=120)),
    ("Patrícia Lima", "RH", "analista", 9_500.00, _NOW - timedelta(days=400)),
    ("Rodrigo Ferreira", "RH", "especialista", 13_000.00, _NOW - timedelta(days=750)),
    ("Sandra Menezes", "Marketing", "gerente", 16_500.00, _NOW - timedelta(days=550)),
    ("Diego Nascimento", "Marketing", "analista", 8_200.00, _NOW - timedelta(days=95)),
    ("Vera Santana", "Operações", "analista", 9_000.00, _NOW - timedelta(days=320)),
    ("José Ricardo", "Suporte Técnico", "coordenador", 13_500.00, _NOW - timedelta(days=480)),
    ("Cláudia Bastos", "Suporte Técnico", "analista", 8_500.00, _NOW - timedelta(days=250)),
    ("Lucas Peixoto", "Suporte Técnico", "estagiário", 2_000.00, _NOW - timedelta(days=60)),
]

# Habilidades: funcionário tem skill (cross-domain RH → conhecimento implícito)
_EMPLOYEE_SKILLS = [
    ("Mariana Albuquerque", "Python"),
    ("Mariana Albuquerque", "Machine Learning"),
    ("Carlos Eduardo Pontes", "Docker"),
    ("Carlos Eduardo Pontes", "Kubernetes"),
    ("Fernanda Yoshida", "Arquitetura de Software"),
    ("Fernanda Yoshida", "AWS"),
    ("Larissa Prado", "Python"),
    ("Larissa Prado", "SQL"),
    ("Renata Oliveira", "React"),
    ("Renata Oliveira", "TypeScript"),
    ("Marcos Vinícius", "Java"),
    ("Marcos Vinícius", "Spring Boot"),
    ("Rafael Monteiro", "Negociação"),
    ("Rafael Monteiro", "CRM"),
    ("Juliana Castro Neves", "Inside Sales"),
    ("Paula Souza", "Prospecção"),
    ("Camila Rocha", "Salesforce"),
    ("André Luiz Sampaio", "Contabilidade Gerencial"),
    ("André Luiz Sampaio", "IFRS"),
    ("Beatriz Fonseca", "Excel Avançado"),
    ("Amanda Torres", "SAP"),
    ("Amanda Torres", "Auditoria"),
    ("Rodrigo Ferreira", "Recrutamento"),
    ("Sandra Menezes", "Growth Marketing"),
    ("Sandra Menezes", "SEO"),
    ("José Ricardo", "ITIL"),
    ("José Ricardo", "Redes TCP/IP"),
]

# ── Vendas ──────────────────────────────────────────────────────────────────

_SELLERS = [
    ("Rafael Monteiro", "Sudeste"),
    ("Juliana Castro Neves", "Sul"),
    ("Paula Souza", "Nordeste"),
    ("Camila Rocha", "Sudeste"),
    ("Felipe Andrade", "Centro-Oeste"),
]

# Clientes expandidos
_CUSTOMERS = [
    ("Lojas Andrade S.A.", "Grande", "Sudeste"),
    ("Cooperativa AgroVale", "Médio", "Sul"),
    ("Clínica Bem Viver", "Médio", "Sudeste"),
    ("Banco Horizonte", "Grande", "Sudeste"),
    ("Transportadora Rocha", "Médio", "Centro-Oeste"),
    ("Escritório Vasconcellos Advogados", "Pequeno", "Sudeste"),
    ("Colégio Santa Cecília", "Médio", "Sul"),
    ("Startup Maré Alta", "Pequeno", "Nordeste"),
    # Novos clientes
    ("Construtora Nova Era", "Grande", "Sudeste"),
    ("Indústria MetalTech", "Grande", "Sul"),
    ("Farmácia Bem Estar", "Pequeno", "Nordeste"),
    ("Hotéis Litoral Sul", "Médio", "Sul"),
    ("Universidade Central", "Grande", "Central"),
    ("Gráfica Express", "Pequeno", "Sudeste"),
    ("Distribuidora Alvorada", "Médio", "Centro-Oeste"),
    ("Seguros Confiança", "Grande", "Sudeste"),
    ("ONG EducaCidadã", "Pequeno", "Nordeste"),
    ("Rede Super Bem", "Grande", "Sudeste"),
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
    # Novos pedidos
    ("Construtora Nova Era", "Camila Rocha", ["CAD-DIR-011", "MES-CON-012", "CAD-ERG-001"]),
    ("Indústria MetalTech", "Juliana Castro Neves", ["RTR-VPN-021", "SWT-24P-020", "AP-WF6-022"]),
    ("Farmácia Bem Estar", "Paula Souza", ["TEC-MEC-005", "MOU-ERG-006", "SUP-NTB-010"]),
    ("Hotéis Litoral Sul", "Juliana Castro Neves", ["MON-32W-015", "PRO-MUL-018", "SPK-CON-023"]),
    ("Universidade Central", "Camila Rocha", ["NTB-DEV-004", "MON-27P-003", "TAB-DIG-017", "PRO-MUL-018"]),
    ("Gráfica Express", "Rafael Monteiro", ["PAP-A4R-032", "TON-CIA-031", "CAN-ESC-033"]),
    ("Distribuidora Alvorada", "Felipe Andrade", ["HUB-USB-007", "DOK-USB-027", "CAB-HDM-029"]),
    ("Seguros Confiança", "Camila Rocha", ["NTB-EXE-016", "MON-32W-015", "HEA-BTH-009", "FILT-TEL-028"]),
    ("ONG EducaCidadã", "Paula Souza", ["NTB-DEV-004", "MOU-TRK-026", "PIL-REC-034"]),
    ("Rede Super Bem", "Rafael Monteiro", ["CAD-ERG-001", "TEC-MEC-005", "MOU-ERG-006", "HEA-BTH-009", "MOCH-NTB-030"]),
    ("Lojas Andrade S.A.", "Rafael Monteiro", ["TEC-ERG-025", "MOU-TRK-026", "FILT-TEL-028"]),
    ("Banco Horizonte", "Camila Rocha", ["MON-32W-015", "DOK-USB-027", "MIC-LAP-024"]),
    ("Startup Maré Alta", "Paula Souza", ["SSD-EXT-019", "CAB-HDM-029"]),
]

# ── Finanças ────────────────────────────────────────────────────────────────

_FINANCE_SUPPLIERS = [
    ("Imobiliária Paulista Ltda", "Imobiliário", "SP"),
    ("TechSoft Brasil S.A.", "Software", "SP"),
    ("Vetta Consultoria", "Consultoria", "RJ"),
    ("Enel Distribuição SP", "Utilidades", "SP"),
    ("Kalunga Comércio", "Suprimentos", "SP"),
    ("Agência Bossa Nova", "Marketing", "SP"),
    # Novos fornecedores
    ("Datacenter Cloud S.A.", "Infraestrutura", "SP"),
    ("Limpeza Total Ltda", "Serviços", "RJ"),
    ("Segurança Digital Corp", "Segurança", "MG"),
    ("Transportadora Veloz", "Logística", "PR"),
    ("Benefícios Flex Card", "Benefícios", "SP"),
    ("Treinamentos Online Ltda", "Educação", "SC"),
]

# Despesas expandidas: (descricao, fornecedor, valor, status, categoria)
_EXPENSES = [
    ("Aluguel do escritório", "Imobiliária Paulista Ltda", 8_500.00, "aberta", "Ocupação"),
    ("Licenças de software anuais", "TechSoft Brasil S.A.", 3_200.00, "aberta", "TI"),
    ("Consultoria de implantação ERP", "Vetta Consultoria", 62_000.00, "aberta", "Consultoria"),
    ("Energia elétrica", "Enel Distribuição SP", 1_840.50, "paga", "Utilidades"),
    ("Material de escritório", "Kalunga Comércio", 487.90, "aberta", "Suprimentos"),
    ("Campanha de marketing digital", "Agência Bossa Nova", 24_000.00, "aberta", "Marketing"),
    # Novas despesas
    ("Servidores cloud mensal", "Datacenter Cloud S.A.", 12_400.00, "aberta", "TI"),
    ("Limpeza e conservação", "Limpeza Total Ltda", 3_500.00, "aberta", "Serviços"),
    ("Monitoramento de segurança", "Segurança Digital Corp", 7_200.00, "paga", "Segurança"),
    ("Frete distribuição clientes", "Transportadora Veloz", 2_800.00, "aberta", "Logística"),
    ("Vale alimentação funcionários", "Benefícios Flex Card", 18_000.00, "aberta", "Benefícios"),
    ("Cursos capacitação equipe", "Treinamentos Online Ltda", 4_500.00, "paga", "Educação"),
    ("Manutenção ar condicionado", "Kalunga Comércio", 1_200.00, "aberta", "Manutenção"),
    ("Auditoria externa anual", "Vetta Consultoria", 35_000.00, "aberta", "Consultoria"),
    ("Seguro predial", "Imobiliária Paulista Ltda", 6_800.00, "paga", "Seguros"),
]

# Alçada de aprovação — espelha services/financas/rules:
_AUTO_APPROVAL_LIMIT = 5_000.00
_MANAGER_APPROVAL_LIMIT = 50_000.00

# Fornecedores → categorias de estoque
_SUPPLIES_STOCK = [
    ("Kalunga Comércio", ["Periféricos", "Acessórios", "Suprimentos"]),
    ("TechSoft Brasil S.A.", ["Eletrônicos"]),
]

# Contrapartes (recebíveis)
_COUNTERPARTIES = [
    "Lojas Andrade S.A.", "Cooperativa AgroVale", "Clínica Bem Viver",
    "Transportadora Rocha", "Banco Horizonte",
    "Construtora Nova Era", "Indústria MetalTech", "Universidade Central",
    "Seguros Confiança", "Rede Super Bem",
]

# Orçamentos: (departamento, periodo, valor_orcado, valor_gasto)
_BUDGETS = [
    ("Engenharia", "2026-Q2", 250_000.00, 198_500.00),
    ("Engenharia", "2026-Q3", 280_000.00, 45_000.00),
    ("Vendas", "2026-Q2", 180_000.00, 155_200.00),
    ("Vendas", "2026-Q3", 200_000.00, 32_000.00),
    ("Financeiro", "2026-Q2", 120_000.00, 89_000.00),
    ("RH", "2026-Q2", 90_000.00, 72_300.00),
    ("Marketing", "2026-Q2", 150_000.00, 142_000.00),
    ("Operações", "2026-Q2", 100_000.00, 65_400.00),
    ("Suporte Técnico", "2026-Q2", 80_000.00, 48_500.00),
]

# ── Cypher statements ───────────────────────────────────────────────────────

_CONSTRAINT = (
    "CREATE CONSTRAINT entity_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.name, e.type, e.domain) IS UNIQUE"
)

_MERGE_ENTITY = (
    "MERGE (e:Entity {name: $name, type: $type, domain: $domain}) "
    "SET e.sku = $sku, e.created_at = coalesce(e.created_at, $created_at)"
)

_MERGE_REL = (
    "MATCH (a:Entity {name: $a_name, type: $a_type, domain: $a_domain}) "
    "MATCH (b:Entity {name: $b_name, type: $b_type, domain: $b_domain}) "
    "MERGE (a)-[r:%s]->(b) "
    "SET r.created_at = coalesce(r.created_at, $created_at)"
)


def _required_approver(amount: float) -> str | None:
    if amount <= _AUTO_APPROVAL_LIMIT:
        return None
    if amount <= _MANAGER_APPROVAL_LIMIT:
        return "gerente"
    return "diretor"


def _seed(session, created_at: str) -> dict[str, int]:  # noqa: ANN001
    counts: dict[str, int] = {"entities": 0, "relations": 0}

    # ── Constraints + índices ───────────────────────────────────────────
    session.run(_CONSTRAINT)
    session.run("CREATE INDEX entity_sku IF NOT EXISTS FOR (e:Entity) ON (e.sku)")
    session.run("CREATE INDEX entity_domain IF NOT EXISTS FOR (e:Entity) ON (e.domain)")
    session.run("CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)")

    # ── Categorias (estoque) ────────────────────────────────────────────
    for cat, parent in _CATEGORIES:
        session.run(_MERGE_ENTITY, name=cat, type="categoria", domain="estoque", sku=None, created_at=created_at)
        counts["entities"] += 1
        if parent:
            session.run(
                _MERGE_REL % "SUBCATEGORIA_DE",
                a_name=cat, a_type="categoria", a_domain="estoque",
                b_name=parent, b_type="categoria", b_domain="estoque",
                created_at=created_at,
            )
            counts["relations"] += 1

    # ── Depósitos ───────────────────────────────────────────────────────
    for name, city, capacity in _WAREHOUSES:
        session.run(
            "MERGE (w:Entity {name: $name, type: 'deposito', domain: 'estoque'}) "
            "SET w.city = $city, w.capacity = $capacity, w.created_at = $created_at",
            name=name, city=city, capacity=capacity, created_at=created_at,
        )
        counts["entities"] += 1

    # ── Produtos (estoque) ──────────────────────────────────────────────
    for sku, name, category, price, on_hand, reorder in _PRODUCTS:
        session.run(
            "MERGE (e:Entity {name: $name, type: 'produto', domain: 'estoque'}) "
            "SET e.sku = $sku, e.unit_price = $price, e.on_hand = $on_hand, "
            "e.reorder_point = $reorder, e.created_at = coalesce(e.created_at, $created_at)",
            name=name, sku=sku, price=price, on_hand=on_hand, reorder=reorder, created_at=created_at,
        )
        counts["entities"] += 1
        session.run(
            _MERGE_REL % "PERTENCE_A",
            a_name=name, a_type="produto", a_domain="estoque",
            b_name=category, b_type="categoria", b_domain="estoque",
            created_at=created_at,
        )
        counts["relations"] += 1

    # ── Alocação de estoque ─────────────────────────────────────────────
    for sku, warehouse, qty in _STOCK_ALLOCATION:
        prod_name = next((n for s, n, *_ in _PRODUCTS if s == sku), sku)
        session.run(
            "MATCH (p:Entity {sku: $sku, type: 'produto', domain: 'estoque'}) "
            "MATCH (w:Entity {name: $warehouse, type: 'deposito', domain: 'estoque'}) "
            "MERGE (p)-[r:ALOCADO_EM]->(w) "
            "SET r.quantity = $qty, r.created_at = coalesce(r.created_at, $created_at)",
            sku=sku, warehouse=warehouse, qty=qty, created_at=created_at,
        )
        counts["relations"] += 1

    # ── Departamentos (rh) ──────────────────────────────────────────────
    for dept, area in _DEPARTMENTS:
        session.run(
            "MERGE (e:Entity {name: $name, type: 'departamento', domain: 'rh'}) "
            "SET e.area = $area, e.created_at = coalesce(e.created_at, $created_at)",
            name=dept, area=area, created_at=created_at,
        )
        counts["entities"] += 1

    # ── Cargos (rh) ─────────────────────────────────────────────────────
    for cargo, nivel in _CARGOS:
        session.run(
            "MERGE (e:Entity {name: $name, type: 'cargo', domain: 'rh'}) "
            "SET e.nivel = $nivel, e.created_at = coalesce(e.created_at, $created_at)",
            name=cargo, nivel=nivel, created_at=created_at,
        )
        counts["entities"] += 1

    # ── Funcionários (rh) ───────────────────────────────────────────────
    for name, dept, cargo, salario, dt in _EMPLOYEES:
        session.run(
            "MERGE (e:Entity {name: $name, type: 'funcionario', domain: 'rh'}) "
            "SET e.salario = $salario, e.data_contratacao = $dt, "
            "e.created_at = coalesce(e.created_at, $created_at)",
            name=name, salario=salario, dt=dt.isoformat(), created_at=created_at,
        )
        counts["entities"] += 1
        session.run(
            _MERGE_REL % "TRABALHA_EM",
            a_name=name, a_type="funcionario", a_domain="rh",
            b_name=dept, b_type="departamento", b_domain="rh",
            created_at=created_at,
        )
        counts["relations"] += 1
        session.run(
            _MERGE_REL % "TEM_CARGO",
            a_name=name, a_type="funcionario", a_domain="rh",
            b_name=cargo, b_type="cargo", b_domain="rh",
            created_at=created_at,
        )
        counts["relations"] += 1

    # ── Habilidades ─────────────────────────────────────────────────────
    for emp, skill in _EMPLOYEE_SKILLS:
        session.run(_MERGE_ENTITY, name=skill, type="habilidade", domain="rh", sku=None, created_at=created_at)
        counts["entities"] += 1
        session.run(
            _MERGE_REL % "POSSUI_HABILIDADE",
            a_name=emp, a_type="funcionario", a_domain="rh",
            b_name=skill, b_type="habilidade", b_domain="rh",
            created_at=created_at,
        )
        counts["relations"] += 1

    # ── Regiões (vendas) — nós para relações cross-domain ────────────────
    _REGIONS = ["Sudeste", "Sul", "Nordeste", "Centro-Oeste", "Norte"]
    for reg in _REGIONS:
        session.run(_MERGE_ENTITY, name=reg, type="regiao", domain="vendas", sku=None, created_at=created_at)
        counts["entities"] += 1

    # ── Vendedores (vendas) ─────────────────────────────────────────────
    for name, region in _SELLERS:
        session.run(
            "MERGE (e:Entity {name: $name, type: 'vendedor', domain: 'vendas'}) "
            "SET e.region = $region, e.created_at = coalesce(e.created_at, $created_at)",
            name=name, region=region, created_at=created_at,
        )
        counts["entities"] += 1
        # Cross-domain: vendedor (vendas) → funcionário (rh)
        session.run(
            _MERGE_REL % "MESMO_QUE",
            a_name=name, a_type="vendedor", a_domain="vendas",
            b_name=name, b_type="funcionario", b_domain="rh",
            created_at=created_at,
        )
        counts["relations"] += 1
        # Vendedor → região de atuação
        session.run(
            _MERGE_REL % "ATUA_EM",
            a_name=name, a_type="vendedor", a_domain="vendas",
            b_name=region, b_type="regiao", b_domain="vendas",
            created_at=created_at,
        )
        counts["relations"] += 1

    # ── Clientes (vendas) ───────────────────────────────────────────────
    for customer, porte, region in _CUSTOMERS:
        session.run(
            "MERGE (e:Entity {name: $name, type: 'cliente', domain: 'vendas'}) "
            "SET e.porte = $porte, e.region = $region, "
            "e.created_at = coalesce(e.created_at, $created_at)",
            name=customer, porte=porte, region=region, created_at=created_at,
        )
        counts["entities"] += 1

    # ── Pedidos ─────────────────────────────────────────────────────────
    sku_to_name = {sku: name for sku, name, *_ in _PRODUCTS}
    seen_comprou: set[tuple[str, str]] = set()
    seen_vendeu: set[tuple[str, str]] = set()

    for customer, seller, skus in _ORDERS:
        if (seller, customer) not in seen_vendeu:
            session.run(
                _MERGE_REL % "VENDEU_PARA",
                a_name=seller, a_type="vendedor", a_domain="vendas",
                b_name=customer, b_type="cliente", b_domain="vendas",
                created_at=created_at,
            )
            counts["relations"] += 1
            seen_vendeu.add((seller, customer))

        for sku in skus:
            product_name = sku_to_name[sku]
            if (customer, product_name) not in seen_comprou:
                session.run(
                    _MERGE_REL % "COMPROU",
                    a_name=customer, a_type="cliente", a_domain="vendas",
                    b_name=product_name, b_type="produto", b_domain="estoque",
                    created_at=created_at,
                )
                counts["relations"] += 1
                seen_comprou.add((customer, product_name))

    # ── Fornecedores (finanças) ─────────────────────────────────────────
    for name, segmento, uf in _FINANCE_SUPPLIERS:
        session.run(
            "MERGE (e:Entity {name: $name, type: 'fornecedor', domain: 'financas'}) "
            "SET e.segmento = $segmento, e.uf = $uf, "
            "e.created_at = coalesce(e.created_at, $created_at)",
            name=name, segmento=segmento, uf=uf, created_at=created_at,
        )
        counts["entities"] += 1

    # ── Despesas ────────────────────────────────────────────────────────
    for descricao, fornecedor, valor, status, categoria in _EXPENSES:
        session.run(
            "MERGE (d:Entity {name: $name, type: 'despesa', domain: 'financas'}) "
            "SET d.amount = $valor, d.status = $status, d.categoria = $categoria, "
            "d.created_at = coalesce(d.created_at, $created_at)",
            name=descricao, valor=valor, status=status, categoria=categoria, created_at=created_at,
        )
        counts["entities"] += 1
        session.run(
            _MERGE_REL % "EMITE",
            a_name=fornecedor, a_type="fornecedor", a_domain="financas",
            b_name=descricao, b_type="despesa", b_domain="financas",
            created_at=created_at,
        )
        counts["relations"] += 1
        required = _required_approver(valor)
        if required is not None:
            session.run(
                _MERGE_REL % "REQUER_APROVACAO",
                a_name=descricao, a_type="despesa", a_domain="financas",
                b_name=required, b_type="cargo", b_domain="rh",
                created_at=created_at,
            )
            counts["relations"] += 1

    # ── Fornecedores → estoque ──────────────────────────────────────────
    for fornecedor, categorias in _SUPPLIES_STOCK:
        for categoria in categorias:
            session.run(
                _MERGE_REL % "ABASTECE",
                a_name=fornecedor, a_type="fornecedor", a_domain="financas",
                b_name=categoria, b_type="categoria", b_domain="estoque",
                created_at=created_at,
            )
            counts["relations"] += 1

    # ── Contrapartes recebíveis ─────────────────────────────────────────
    for name in _COUNTERPARTIES:
        session.run(_MERGE_ENTITY, name=name, type="cliente_recebivel", domain="financas", sku=None, created_at=created_at)
        counts["entities"] += 1
        if name in {c for c, *_ in _CUSTOMERS}:
            session.run(
                _MERGE_REL % "MESMO_QUE",
                a_name=name, a_type="cliente_recebivel", a_domain="financas",
                b_name=name, b_type="cliente", b_domain="vendas",
                created_at=created_at,
            )
            counts["relations"] += 1

    # ── Orçamentos ──────────────────────────────────────────────────────
    for dept, periodo, orcado, gasto in _BUDGETS:
        name = f"Orçamento {dept} {periodo}"
        session.run(
            "MERGE (b:Entity {name: $name, type: 'orcamento', domain: 'financas'}) "
            "SET b.valor_orcado = $orcado, b.valor_gasto = $gasto, b.periodo = $periodo, "
            "b.created_at = coalesce(b.created_at, $created_at)",
            name=name, orcado=orcado, gasto=gasto, periodo=periodo, created_at=created_at,
        )
        counts["entities"] += 1
        session.run(
            _MERGE_REL % "ORCAMENTO_DE",
            a_name=name, a_type="orcamento", a_domain="financas",
            b_name=dept, b_type="departamento", b_domain="rh",
            created_at=created_at,
        )
        counts["relations"] += 1

    # ── Cross-domain: estoque → rh (equipamentos alocados a departamentos) ─
    # Cada departamento recebe equipamentos do estoque para seus funcionários.
    _DEPT_EQUIPMENT = [
        ("Engenharia", ["NTB-DEV-004", "MON-27P-003", "TEC-MEC-005"]),
        ("Vendas", ["NTB-EXE-016", "HEA-BTH-009", "MOU-TRK-026"]),
        ("Financeiro", ["MON-27P-003", "TEC-ERG-025", "FILT-TEL-028"]),
        ("RH", ["NTB-EXE-016", "CAM-FHD-008"]),
        ("Marketing", ["TAB-DIG-017", "SPK-CON-023", "MIC-LAP-024"]),
        ("Operações", ["SWT-24P-020", "RTR-VPN-021"]),
        ("Suporte Técnico", ["DOK-USB-027", "SSD-EXT-019", "CAB-HDM-029"]),
    ]
    sku_to_name = {sku: name for sku, name, *_ in _PRODUCTS}
    for dept, skus in _DEPT_EQUIPMENT:
        for sku in skus:
            prod_name = sku_to_name[sku]
            session.run(
                _MERGE_REL % "ALOCADO_PARA",
                a_name=prod_name, a_type="produto", a_domain="estoque",
                b_name=dept, b_type="departamento", b_domain="rh",
                created_at=created_at,
            )
            counts["relations"] += 1

    # ── Cross-domain: vendas → finanças (comissões de vendedores) ─────────
    # Vendedores geram comissão que vira despesa em finanças.
    _SELLER_COMMISSIONS = [
        ("Rafael Monteiro", "Comissão vendas Sudeste Q2", 12_500.00),
        ("Juliana Castro Neves", "Comissão vendas Sul Q2", 8_200.00),
        ("Paula Souza", "Comissão vendas Nordeste Q2", 5_400.00),
        ("Camila Rocha", "Comissão vendas Sudeste Q2", 9_800.00),
    ]
    for seller, desc, valor in _SELLER_COMMISSIONS:
        session.run(
            "MERGE (d:Entity {name: $name, type: 'despesa', domain: 'financas'}) "
            "SET d.amount = $valor, d.status = 'aberta', d.categoria = 'Comissão', "
            "d.created_at = coalesce(d.created_at, $created_at)",
            name=desc, valor=valor, created_at=created_at,
        )
        counts["entities"] += 1
        session.run(
            _MERGE_REL % "GERA_COMISSAO",
            a_name=seller, a_type="vendedor", a_domain="vendas",
            b_name=desc, b_type="despesa", b_domain="financas",
            created_at=created_at,
        )
        counts["relations"] += 1
        required = _required_approver(valor)
        if required is not None:
            session.run(
                _MERGE_REL % "REQUER_APROVACAO",
                a_name=desc, a_type="despesa", a_domain="financas",
                b_name=required, b_type="cargo", b_domain="rh",
                created_at=created_at,
            )
            counts["relations"] += 1

    # ── Cross-domain: finanças → vendas (campanhas de marketing por região) ──
    _MARKETING_CAMPAIGNS = [
        ("Agência Bossa Nova", "Campanha de marketing digital", "Sudeste"),
        ("Agência Bossa Nova", "Lançamento linha premium", "Sul"),
    ]
    for fornecedor, campanha, regiao in _MARKETING_CAMPAIGNS:
        session.run(_MERGE_ENTITY, name=campanha, type="campanha", domain="financas", sku=None, created_at=created_at)
        counts["entities"] += 1
        session.run(
            _MERGE_REL % "EMITE",
            a_name=fornecedor, a_type="fornecedor", a_domain="financas",
            b_name=campanha, b_type="campanha", b_domain="financas",
            created_at=created_at,
        )
        counts["relations"] += 1
        session.run(
            _MERGE_REL % "ALVO_REGIAO",
            a_name=campanha, a_type="campanha", a_domain="financas",
            b_name=regiao, b_type="regiao", b_domain="vendas",
            created_at=created_at,
        )
        counts["relations"] += 1

    return counts


def main(uri: str, user: str, password: str) -> None:
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("neo4j driver não instalado. pip install neo4j")
        sys.exit(1)

    created_at = _NOW.isoformat()
    driver = GraphDatabase.driver(uri, auth=(user, password), max_connection_lifetime=3600)
    try:
        driver.verify_connectivity()
        logger.info("Conectado a %s", uri)
        with driver.session() as session:
            counts = _seed(session, created_at)
        logger.info(
            "Seed concluído: %d entidades, %d relações.",
            counts["entities"], counts["relations"],
        )
    finally:
        driver.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Popula Neo4j com entidades enriquecidas dos microsserviços.")
    parser.add_argument("--uri", default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    parser.add_argument("--user", default=os.getenv("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.getenv("NEO4J_PASSWORD", "changeme"))
    args = parser.parse_args()
    main(args.uri, args.user, args.password)
