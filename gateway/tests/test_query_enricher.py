"""Testes unitários do enricher contextual (Semiose — Camada A)."""

from __future__ import annotations

from gateway.query_enricher import (
    ContextSignal,
    _has_strong_conflict,
    _regex_extract,
    enrich_query,
    gather_signals,
)


# ── Extração de entidades por regex ──────────────────────────────────────


def test_regex_extract_sku_3_partes():
    assert "CAD-ERG-001" in _regex_extract("qual o saldo do CAD-ERG-001?")


def test_regex_extract_money_e_cpf():
    ents = _regex_extract("pague R$ 1.200,00 ao 123.456.789-00")
    assert any("R$" in e for e in ents)
    assert "123.456.789-00" in ents


def test_regex_extract_sku_2_partes_nao_casa():
    # Limitação conhecida e documentada: SKU de 2 partes não casa.
    assert _regex_extract("saldo do MON-027?") == []


# ── gather_signals ───────────────────────────────────────────────────────


def test_gather_signals_prioriza_last_route():
    state = {"sanitized": "e o ponto de reposição?", "_last_route": {"domains": ["estoque"]}}
    sig = gather_signals(state, spacy_enabled=False)
    assert sig.last_domain == "estoque"


def test_gather_signals_fallback_keywords_do_history():
    state = {
        "sanitized": "e agora?",
        "history": [{"role": "user", "content": "quais contas a pagar vencem?"}],
    }
    sig = gather_signals(state, spacy_enabled=False)
    assert sig.last_domain == "financas"


def test_gather_signals_extrai_entidades_da_query():
    state = {"sanitized": "saldo do CAD-ERG-001?", "_last_route": {"domains": ["estoque"]}}
    sig = gather_signals(state, spacy_enabled=False)
    assert "CAD-ERG-001" in sig.recent_entities


# ── enrich_query ─────────────────────────────────────────────────────────


def test_enrich_query_sem_sinais_retorna_original():
    q, enriched = enrich_query("qual o preço?", ContextSignal())
    assert enriched is False
    assert q == "qual o preço?"


def test_enrich_query_mesmo_dominio_adiciona_prefixo():
    sig = ContextSignal(last_domain="estoque")
    q, enriched = enrich_query("e o ponto de reposição?", sig)
    assert enriched is True
    assert q.startswith("[domínio: estoque]")


def test_enrich_query_inclui_entidades_no_prefixo():
    sig = ContextSignal(last_domain="estoque", recent_entities=["CAD-ERG-001"])
    q, enriched = enrich_query("e o ponto de reposição?", sig)
    assert enriched is True
    assert "CAD-ERG-001" in q


def test_enrich_query_troca_de_topico_dropa_enriquecimento():
    sig = ContextSignal(last_domain="rh")
    q, enriched = enrich_query("qual o saldo do estoque do monitor?", sig)
    assert enriched is False
    assert q == "qual o saldo do estoque do monitor?"


# ── _has_strong_conflict ─────────────────────────────────────────────────


def test_conflict_detecta_keyword_de_outro_dominio():
    assert _has_strong_conflict("quais férias pendentes?", "estoque") is True


def test_conflict_vocab_cross_domain_nao_dispara():
    # "SKU" em vendas é detalhe de order_items, não troca para estoque.
    assert _has_strong_conflict("quais SKUs foram incluídos?", "vendas") is False


def test_conflict_entidade_estruturada_implica_dominio():
    # CPF implica RH; com contexto vendas → conflito (troca de tópico).
    assert _has_strong_conflict("qual o cargo do 123.456.789-00?", "vendas") is True


# ── KG feedback (Camada A+B): kg_neighbors + enrich_query(kg=...) ─────────


class _FakeKG:
    """KG mock: retorna vizinhos fixos (ou levanta) para testar o enricher."""

    def __init__(self, related=None, raise_on=False):
        self._related = related or []
        self._raise = raise_on

    def expand(self, entity_name, entity_type, target_domain=""):
        if self._raise:
            raise RuntimeError("kg down")
        return {"status": 200, "body": {"related": self._related}}


def test_kg_neighbors_resolve_sku():
    from gateway.query_enricher import kg_neighbors

    kg = _FakeKG(related=[
        {"name": "Mobiliário", "type": "categoria", "domain": "estoque"},
        {"name": "Lojas Andrade S.A.", "type": "cliente", "domain": "vendas"},
    ])
    out = kg_neighbors(["CAD-ERG-001"], kg)
    assert "Mobiliário (categoria/estoque)" in out
    assert "Lojas Andrade S.A. (cliente/vendas)" in out


def test_kg_neighbors_sem_kg_ou_sem_entidade():
    from gateway.query_enricher import kg_neighbors

    rel = [{"name": "x", "type": "t", "domain": "estoque"}]
    assert kg_neighbors(["CAD-ERG-001"], None) == []
    assert kg_neighbors([], _FakeKG(related=rel)) == []


def test_kg_neighbors_graceful_em_excecao():
    from gateway.query_enricher import kg_neighbors

    assert kg_neighbors(["CAD-ERG-001"], _FakeKG(raise_on=True)) == []


def test_kg_neighbors_dedup_e_limite():
    from gateway.query_enricher import kg_neighbors

    rel = [{"name": f"n{i}", "type": "t", "domain": "estoque"} for i in range(10)]
    rel.append({"name": "n0", "type": "t", "domain": "estoque"})  # duplicata
    out = kg_neighbors(["CAD-ERG-001"], _FakeKG(related=rel), limit=6)
    assert len(out) == 6
    assert len(set(out)) == 6


def test_enrich_query_com_kg_injeta_relacionado():
    kg = _FakeKG(related=[{"name": "Mobiliário", "type": "categoria", "domain": "estoque"}])
    sig = ContextSignal(last_domain="estoque", recent_entities=["CAD-ERG-001"])
    q, enriched = enrich_query("e o ponto de reposição?", sig, kg=kg)
    assert enriched is True
    assert "relacionado: Mobiliário (categoria/estoque)" in q


def test_enrich_query_kg_none_mantem_comportamento():
    sig = ContextSignal(last_domain="estoque", recent_entities=["CAD-ERG-001"])
    q, enriched = enrich_query("e o ponto?", sig, kg=None)
    assert "relacionado:" not in q
