"""BM25 (Okapi) in-process sobre o golden de roteamento — metade lexical do S2.

Semiose — S2 (retrieval híbrido): espelha "Contextual Embeddings + Contextual
BM25" (Anthropic, 2024). O corpus é pequeno (centenas de exemplos) e estático
por processo: implementação stdlib evita dependência nova na imagem do gateway
e mantém a tokenização idêntica à normalização do router léxico (lowercase,
sem acentos) — vocabulário único entre as camadas.

IDF na variante Lucene (log(1 + (N - df + 0.5)/(df + 0.5))): sempre positiva,
sem o floor artificial da formulação clássica.
"""

from __future__ import annotations

import math
import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Palavras funcionais PT: sem elas, um match acidental de "do"/"que" não dá
# rank BM25 (e, na fusão RRF, não promove documento sem sinal lexical real).
_STOPWORDS = frozenset(
    "a o as os um uma uns umas de do da dos das no na nos nas em ao aos "
    "e ou que se com por para pra pelo pela sem sob sobre entre como".split()
)

K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase + remoção de acentos + tokens alfanuméricos, sem stopwords."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [token for token in _TOKEN_RE.findall(text) if token not in _STOPWORDS]


class BM25Index:
    """Okapi BM25 sobre um corpus imutável de documentos."""

    def __init__(self, documents: list[str], *, k1: float = K1, b: float = B) -> None:
        self._k1 = k1
        self._b = b
        self._doc_tokens = [tokenize(doc) for doc in documents]
        self._doc_lens = [len(tokens) for tokens in self._doc_tokens]
        self._avgdl = sum(self._doc_lens) / len(self._doc_lens) if documents else 0.0
        self._tf: list[dict[str, int]] = []
        df: dict[str, int] = {}
        for tokens in self._doc_tokens:
            counts: dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self._tf.append(counts)
            for token in counts:
                df[token] = df.get(token, 0) + 1
        n = len(documents)
        self._idf = {
            token: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5)) for token, freq in df.items()
        }

    def __len__(self) -> int:
        return len(self._doc_tokens)

    def scores(self, query: str) -> list[float]:
        """Score BM25 da query contra cada documento (mesma ordem do corpus)."""
        tokens = tokenize(query)
        result: list[float] = []
        for tf, doc_len in zip(self._tf, self._doc_lens):
            score = 0.0
            for token in tokens:
                freq = tf.get(token)
                if not freq:
                    continue
                norm = 1 - self._b + self._b * doc_len / self._avgdl if self._avgdl else 1.0
                score += self._idf[token] * (freq * (self._k1 + 1)) / (freq + self._k1 * norm)
            result.append(score)
        return result

    def rank(self, query: str, *, limit: int) -> list[int]:
        """Índices dos documentos por score desc; só documentos com score > 0."""
        scores = self.scores(query)
        order = sorted(
            (i for i, score in enumerate(scores) if score > 0.0),
            key=lambda i: scores[i],
            reverse=True,
        )
        return order[:limit]
