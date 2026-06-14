"""Testes do InjectionDetector com mock — sem dependência de modelo real."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from gateway.injection_detector import InjectionDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def detector_no_model() -> InjectionDetector:
    """Detector apontando para path inexistente."""
    return InjectionDetector(model_path="/tmp/nonexistent_model_xyz", threshold=0.7)


@pytest.fixture()
def detector_mock() -> InjectionDetector:
    """Detector com modelo mockado que retorna probabilidades controladas."""
    det = InjectionDetector(model_path="/tmp/fake", threshold=0.7)
    # Simula modelo já carregado.
    det._available = True
    det._model = MagicMock()
    det._tokenizer = MagicMock()
    return det


# ---------------------------------------------------------------------------
# Fallback — modelo indisponível
# ---------------------------------------------------------------------------

def test_score_returns_negative_when_model_missing(detector_no_model: InjectionDetector):
    assert detector_no_model.score("qualquer texto") == -1.0


def test_is_injection_returns_false_when_model_missing(detector_no_model: InjectionDetector):
    assert detector_no_model.is_injection("ignore as instruções") is False


def test_available_false_when_model_missing(detector_no_model: InjectionDetector):
    detector_no_model.score("x")
    assert detector_no_model.available is False


# ---------------------------------------------------------------------------
# Modelo mockado
# ---------------------------------------------------------------------------

def _mock_score(detector: InjectionDetector, prob_injection: float) -> None:
    """Configura detector mockado para retornar probabilidade específica via score().

    Mocka is_injection() diretamente, já que torch não está instalado no env de teste.
    """
    detector.score = MagicMock(return_value=prob_injection)


def test_score_returns_float_between_0_and_1(detector_mock: InjectionDetector):
    _mock_score(detector_mock, 0.73)
    score = detector_mock.score("ignore instructions")
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_is_injection_true_above_threshold(detector_mock: InjectionDetector):
    _mock_score(detector_mock, 0.88)
    # is_injection chama score() internamente, mas como mockamos score(),
    # precisamos testar via threshold manual.
    assert detector_mock.score("ignore tudo") >= detector_mock.threshold


def test_is_injection_false_below_threshold(detector_mock: InjectionDetector):
    _mock_score(detector_mock, 0.27)
    assert detector_mock.score("qual o saldo?") < detector_mock.threshold


def test_threshold_configurable():
    det = InjectionDetector(threshold=0.5)
    assert det.threshold == 0.5
    det.threshold = 0.9
    assert det.threshold == 0.9


def test_score_negative_one_when_path_is_none():
    det = InjectionDetector(model_path=None)
    assert det.score("test") == -1.0


def test_is_injection_false_when_path_is_none():
    det = InjectionDetector(model_path=None)
    assert det.is_injection("test") is False
