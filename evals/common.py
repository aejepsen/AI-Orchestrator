"""Utilidades compartilhadas dos evals.

`with_deadline` é o watchdog por caso: nenhum caso pode segurar o eval para
sempre (socket meio-morto no Ollama segura o read até o timeout do httpx e,
com retries, o tempo total fica não-determinístico — já custou 3h de espera).
Caso estoure o prazo, vira falha de caso e o eval segue para o próximo.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Any, Callable

# Prazo generoso por caso: cold load do modelo + tool-loop completo cabem aqui;
# só um processo travado de verdade estoura.
DEFAULT_CASE_DEADLINE_S = 420.0


class CaseTimeout(Exception):
    """Caso individual estourou o deadline — falha do caso, não do eval."""


def with_deadline(
    fn: Callable[..., Any],
    *args: Any,
    deadline_s: float = DEFAULT_CASE_DEADLINE_S,
    **kwargs: Any,
) -> Any:
    """Executa `fn` com prazo máximo. Estourou → CaseTimeout.

    A thread abandonada termina sozinha quando o httpx desistir; o eval não
    espera por ela (daemon worker não bloqueia o exit do processo).
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=deadline_s)
        except FutureTimeoutError as exc:
            raise CaseTimeout(f"caso excedeu {deadline_s:.0f}s — abortado pelo watchdog") from exc
    finally:
        executor.shutdown(wait=False)
