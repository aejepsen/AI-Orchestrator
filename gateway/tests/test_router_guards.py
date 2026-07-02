"""Testes dos guards determinísticos pós-classificação (_apply_routing_guards)."""

from gateway.router import RoutePlan, _apply_routing_guards


def _plan(domains: list[str]) -> RoutePlan:
    return RoutePlan(domains=domains, plan="p", clarification=None)


class TestFornecedorFinanceiro:
    def test_fase2_nao_readiciona_estoque_removido_pela_fase1(self) -> None:
        # Bug corrigido: fase 1 removia estoque ("despesas acima" = finanças),
        # fase 2 re-adicionava por "fornecedores" — medido: 3 falhas SUPER.
        plan = _apply_routing_guards(
            "Liste os fornecedores com despesas acima de R$ 10.000.",
            _plan(["financas", "estoque"]),
        )
        assert plan.domains == ["financas"]

    def test_fornecedor_sem_contexto_financeiro_ganha_estoque(self) -> None:
        plan = _apply_routing_guards(
            "Qual o fornecedor do SKU CAD-ERG-001?", _plan(["estoque"])
        )
        assert "estoque" in plan.domains


class TestConceitoFinanceiro:
    def test_folha_de_departamento_ganha_financas_e_rh(self) -> None:
        plan = _apply_routing_guards(
            "Qual o custo total de folha de pagamento do departamento financeiro?",
            _plan(["rh"]),
        )
        assert set(plan.domains) == {"rh", "financas"}

    def test_margem_de_lucro_em_pedidos_ganha_financas(self) -> None:
        plan = _apply_routing_guards(
            "Qual a margem de lucro nos pedidos da região Centro-Oeste?", _plan(["vendas"])
        )
        assert set(plan.domains) == {"vendas", "financas"}

    def test_nao_cria_rota_do_nada(self) -> None:
        # Clarification (domains vazio) fica intacta — guard só complementa.
        plan = RoutePlan(domains=[], plan="", clarification="Pode detalhar?")
        result = _apply_routing_guards("qual o orçamento disso aí?", plan)
        assert result.domains == []

    def test_pergunta_sem_conceito_financeiro_nao_muda(self) -> None:
        plan = _apply_routing_guards("Liste os produtos do armazém SP.", _plan(["estoque"]))
        assert plan.domains == ["estoque"]


class TestQuemAprova:
    def test_quem_aprova_despesas_vira_financas_e_rh(self) -> None:
        plan = _apply_routing_guards(
            "Quem aprova despesas acima de R$ 50.000?", _plan(["financas"])
        )
        assert set(plan.domains) >= {"financas", "rh"}
