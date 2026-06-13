"""Cliente Ollama /api/chat tipado.

Decisões de latência (medições da Fase 0 e eval da Fase 2):
- `keep_alive=-1`: MoE residente permanente — zero cold load entre chamadas.
- `think=true`: com think=false o qwen3 vaza a cadeia de raciocínio no
  `content` ("Okay, let's see...") e decide regras de negócio sem chamar a
  tool (medido no 1º eval: 7/40 falhas). Com think=true o raciocínio vai
  para o campo `thinking` (descartado) e o `content` fica limpo.
- `stream=false`: o loop de agente consome a resposta inteira por iteração.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import httpx


class LLMError(RuntimeError):
    """Falha de transporte ou resposta malformada do Ollama."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ChatResponse:
    content: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for raw in raw_calls:
        function = raw.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise LLMError(f"Argumentos de tool_call não são JSON válido: {arguments!r}") from exc
        calls.append(ToolCall(name=function.get("name", ""), arguments=arguments or {}))
    return tuple(calls)


class OllamaClient:
    """Cliente síncrono do endpoint /api/chat com suporte a tools."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 300.0,
        keep_alive: str = "30m",
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        # NUNCA -1: modelo residente permanente impede o Ollama de despejar
        # VRAM quando outro modelo precisa carregar → deadlock do scheduler
        # (request espera espaço que nunca libera). 30m mantém quente sob
        # tráfego real e ainda permite eviction.
        # Ollama 0.30+ rejeita keep_alive string sem unidade (ex.: "-1").
        # Converte pra int se for numérico (segundos); senão mantém string ("30m").
        try:
            self._keep_alive = int(keep_alive)
        except (ValueError, TypeError):
            self._keep_alive = keep_alive
        self._client = client or httpx.Client(timeout=timeout_s)

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        """Embeddings em lote via /api/embed (modelo dedicado, ex.: nomic-embed-text)."""
        payload = {"model": model, "input": texts, "keep_alive": self._keep_alive}
        try:
            response = self._client.post(f"{self._base_url}/api/embed", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Falha no embedding via Ollama ({self._base_url}): {exc}") from exc
        data = response.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise LLMError(f"Resposta de /api/embed malformada: {data!r}")
        return embeddings

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        format: str | dict[str, Any] | None = None,
    ) -> ChatResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            # think só em modelos com modo de raciocínio (qwen3, não qwen3.5);
            # Qwen3.5 Small series não emite <think> — Ollama rejeita think=true.
            "think": self._model.startswith("qwen3") and "qwen3.5" not in self._model,
            "keep_alive": self._keep_alive,
            "options": {"temperature": temperature},
        }
        if tools:
            payload["tools"] = tools
        if format is not None:
            payload["format"] = format
        try:
            response = self._client.post(f"{self._base_url}/api/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError(f"Falha na chamada ao Ollama ({self._base_url}): {exc}") from exc

        data = response.json()
        message = data.get("message")
        if not isinstance(message, dict):
            raise LLMError(f"Resposta do Ollama sem campo 'message': {data!r}")
        return ChatResponse(
            content=(message.get("content") or "").strip(),
            tool_calls=_parse_tool_calls(message.get("tool_calls") or []),
        )
