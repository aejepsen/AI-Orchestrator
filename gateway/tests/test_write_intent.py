"""Testes da detecção determinística de write intent (gate do HITL)."""

import threading

import pytest

from gateway.graph import GatewayGraph
from gateway.write_intent import detect_write_intent


class TestDetectWriteIntent:
    @pytest.mark.parametrize(
        "question",
        [
            "Crie uma conta a pagar de R$ 300 para a Papelaria Central",
            "quero pagar a conta 3 hoje",
            "Reserve 50 unidades do SKU CAD-ERG-001",
            "cancele o pedido 12 do cliente Beta",
            "Conceda 10 dias de férias ao Carlos",
            "admita a funcionária Ana no departamento de Vendas",
            "exclua a conta 7",
            "atualize o valor da conta 5 para R$ 900",
            "aprove a despesa de consultoria",
            "libere a reserva 4 do estoque",
        ],
    )
    def test_escrita_detectada(self, question: str) -> None:
        is_write, terms = detect_write_intent(question)
        assert is_write
        assert terms

    @pytest.mark.parametrize(
        "question",
        [
            "Quais contas a pagar vencem esta semana?",
            "quanto temos em contas a receber?",
            "Qual o saldo do SKU CAD-ERG-001 no estoque?",
            "liste os funcionários do departamento de Vendas",
            "qual a comissão da Juliana neste mês?",
            "quantos dias de férias o Carlos ainda tem?",
            "qual o fluxo de caixa projetado para julho?",
            "existem pedidos a concluir hoje?",
        ],
    )
    def test_leitura_nao_dispara(self, question: str) -> None:
        is_write, terms = detect_write_intent(question)
        assert not is_write, f"falso positivo: {terms}"

    def test_acentos_e_caixa_nao_importam(self) -> None:
        assert detect_write_intent("LANÇAR nova despesa de viagem")[0]

    def test_termos_reportados_para_preview(self) -> None:
        _, terms = detect_write_intent("crie o pedido e reserve o estoque")
        assert terms == ["crie", "reserve"]


class TestConfirmDispatchGate:
    """O nó confirm_dispatch só pausa quando há callback E write intent."""

    def _graph_stub(self, on_confirm) -> GatewayGraph:
        graph = GatewayGraph.__new__(GatewayGraph)
        graph._local = threading.local()
        graph._local.on_confirm = on_confirm
        return graph

    def _state(self, question: str) -> dict:
        return {
            "sanitized": question,
            "route": {"domains": ["financas"], "plan": "plano"},
        }

    def test_sem_callback_auto_aprova(self) -> None:
        graph = self._graph_stub(on_confirm=None)
        assert graph._confirm_dispatch(self._state("pague a conta 3")) == {}

    def test_leitura_nao_chama_callback_nem_interrompe(self) -> None:
        calls: list = []
        graph = self._graph_stub(on_confirm=lambda d, p: calls.append((d, p)))
        result = graph._confirm_dispatch(self._state("quais contas a pagar vencem hoje?"))
        assert result == {}
        assert calls == []

    def test_escrita_chama_callback_e_interrupt(self, monkeypatch) -> None:
        calls: list = []
        payloads: list = []

        def fake_interrupt(payload):
            payloads.append(payload)
            return {"approved": True}

        monkeypatch.setattr("langgraph.types.interrupt", fake_interrupt)
        graph = self._graph_stub(on_confirm=lambda d, p: calls.append((d, p)))
        result = graph._confirm_dispatch(self._state("pague a conta 3"))
        assert result == {}
        assert calls == [(["financas"], "plano")]
        assert payloads[0]["type"] == "confirm_dispatch"
        assert payloads[0]["write_terms"] == ["pague"]

    def test_escrita_rejeitada_cancela(self, monkeypatch) -> None:
        monkeypatch.setattr("langgraph.types.interrupt", lambda payload: {"approved": False})
        graph = self._graph_stub(on_confirm=lambda d, p: None)
        result = graph._confirm_dispatch(self._state("exclua a conta 7"))
        assert result["final_answer"] == "Operação cancelada pelo usuário."
