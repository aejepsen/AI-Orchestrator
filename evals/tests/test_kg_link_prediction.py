"""Testes dos helpers puros do link prediction do KG (sem pykeen/neo4j)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.kg_link_prediction import (  # noqa: E402
    candidate_triples,
    entity_label,
    relation_signatures,
)

TRIPLES = [
    ("cliente:Acme", "COMPROU", "produto:Monitor"),
    ("cliente:Beta", "COMPROU", "produto:Teclado"),
    ("despesa:Consultoria", "REQUER_APROVACAO", "cargo:diretor"),
    ("funcionario:Ana", "TRABALHA_EM", "departamento:Vendas"),
]


def test_entity_label_prefixa_tipo() -> None:
    assert entity_label("Acme", "cliente") == "cliente:Acme"


def test_signatures_por_relacao() -> None:
    signatures = relation_signatures(TRIPLES)
    assert signatures["COMPROU"] == {("cliente", "produto")}
    assert signatures["REQUER_APROVACAO"] == {("despesa", "cargo")}


def test_candidatas_respeitam_assinatura_de_tipo() -> None:
    candidates = candidate_triples(TRIPLES)
    # cliente Acme × produto Teclado é candidata (type-compatible, ausente).
    assert ("cliente:Acme", "COMPROU", "produto:Teclado") in candidates
    # Cross-type NUNCA: funcionário não COMPROU produto neste schema.
    assert not any(
        h.startswith("funcionario:") and r == "COMPROU" for h, r, _ in candidates
    )


def test_candidatas_excluem_existentes_e_self_loops() -> None:
    candidates = candidate_triples(TRIPLES)
    assert ("cliente:Acme", "COMPROU", "produto:Monitor") not in candidates
    assert not any(h == t for h, _, t in candidates)


def test_max_per_relation_limita_espaco() -> None:
    candidates = candidate_triples(TRIPLES, max_per_relation=1)
    per_relation: dict[str, int] = {}
    for _, relation, _ in candidates:
        per_relation[relation] = per_relation.get(relation, 0) + 1
    assert all(count <= 1 for count in per_relation.values())
