"""Testes do índice BM25 in-process (S2 — metade lexical do retrieval híbrido)."""

from gateway.bm25 import BM25Index, tokenize


class TestTokenize:
    def test_lowercase_sem_acentos_sem_stopwords(self) -> None:
        assert tokenize("Férias do João") == ["ferias", "joao"]

    def test_sku_alfanumerico_vira_tokens(self) -> None:
        assert tokenize("SKU CAD-ERG-001") == ["sku", "cad", "erg", "001"]


class TestBM25Index:
    CORPUS = [
        "Qual o saldo do SKU ABC-123 no estoque?",
        "Quais contas a pagar vencem hoje?",
        "Quantos dias de férias o funcionário tem?",
    ]

    def test_termo_exato_rankeia_documento_certo(self) -> None:
        index = BM25Index(self.CORPUS)
        ranking = index.rank("saldo do sku ABC-123", limit=3)
        assert ranking[0] == 0

    def test_acentos_nao_importam(self) -> None:
        index = BM25Index(self.CORPUS)
        assert index.rank("ferias funcionario", limit=1) == [2]

    def test_sem_match_lexical_fica_fora_do_rank(self) -> None:
        index = BM25Index(self.CORPUS)
        assert index.rank("previsão do tempo amanhã", limit=3) == []

    def test_termo_raro_pesa_mais_que_comum(self) -> None:
        corpus = ["estoque de produto", "estoque de sku raro", "estoque geral"]
        index = BM25Index(corpus)
        scores = index.scores("sku no estoque")
        assert scores[1] == max(scores)

    def test_corpus_vazio_nao_quebra(self) -> None:
        index = BM25Index([])
        assert len(index) == 0
        assert index.scores("qualquer coisa") == []
        assert index.rank("qualquer coisa", limit=5) == []
