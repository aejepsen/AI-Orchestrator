"""Detector de prompt injection com BERTimbau + fallback regex."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class InjectionDetector:
    """Classificador binário de prompt injection baseado em BERTimbau fine-tunado.

    Carrega o modelo lazily na primeira chamada a ``score()`` ou ``is_injection()``.
    Se o modelo não estiver disponível (path inexistente, dependência faltando),
    retorna ``-1.0`` em ``score()`` e ``False`` em ``is_injection()`` — nunca
    bloqueia o pipeline; o fallback regex em ``sanitize.flag_injection`` assume.
    """

    def __init__(self, model_path: str | None = None, threshold: float = 0.7) -> None:
        self._model_path = model_path
        self._threshold = threshold
        self._model = None  # lazy
        self._tokenizer = None  # lazy
        self._available = False

    # -- propriedades públicas -------------------------------------------------

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = value

    @property
    def available(self) -> bool:
        """True após modelo carregado com sucesso."""
        return self._available

    # -- lazy load -------------------------------------------------------------

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return self._available
        if self._model_path is None or not Path(self._model_path).exists():
            logger.warning("InjectionDetector: modelo não encontrado em %s", self._model_path)
            self._available = False
            return False
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch  # noqa: F401 — garante disponibilidade

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_path)
            self._model = AutoModelForSequenceClassification.from_pretrained(self._model_path)
            self._model.eval()
            self._available = True
            logger.info("InjectionDetector loaded: %s", self._model_path)
        except Exception as e:
            logger.warning("InjectionDetector: falha ao carregar modelo: %s", e)
            self._available = False
        return self._available

    # -- API pública -----------------------------------------------------------

    def score(self, text: str) -> float:
        """Retorna probabilidade de ser injection (0.0 a 1.0).

        Retorna ``-1.0`` se o modelo não estiver disponível.
        """
        if not self._ensure_model():
            return -1.0  # sinaliza indisponível
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            return probs[0][1].item()  # prob da classe 1 (injection)

    def is_injection(self, text: str) -> bool:
        """True se a probabilidade de injection >= threshold."""
        s = self.score(text)
        if s < 0:
            return False  # modelo indisponível, não bloqueia
        return s >= self._threshold
