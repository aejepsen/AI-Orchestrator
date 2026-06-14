"""Subagentes especialistas: loop de tool-calling escopado por domínio.

Subagente = mesmo MoE residente com system prompt e tools restritos ao seu
microsserviço (least-privilege por escopo de tool, decisão pós-Fase 0).
O loop reinjetará erros 422/404 como observação — a regra de negócio vive
na API, nunca no modelo.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from gateway.config import Settings, load_settings
from gateway.llm import ChatResponse, OllamaClient, ToolCall
from gateway.tools.registry import ToolNotFound, ToolRegistry

_DOMAIN_LABELS = {
    "financas": "Finanças (contas a pagar/receber, fluxo de caixa, alçada de aprovação)",
    "rh": "RH (funcionários, férias CLT, reembolsos, headcount)",
    "estoque": "Estoque (produtos por SKU, saldo, reservas, ponto de reposição)",
    "vendas": "Vendas (pedidos, política de desconto, comissão)",
}

_SYSTEM_PROMPT_TEMPLATE = """Você é o agente especialista de {label}.
Hoje é {today}.

Regras obrigatórias:
- Use exclusivamente as ferramentas fornecidas para consultar ou alterar dados; nunca invente valores, ids, SKUs ou totais.
- REGRA CRÍTICA PARA OPERAÇÕES DE ESCRITA (criar, atualizar, excluir): se o usuário NÃO forneceu TODOS os dados obrigatórios para a operação, NÃO execute a ferramenta. Em vez disso, liste os campos necessários e peça ao usuário que forneça os valores. NUNCA fabrique nomes, salários, datas, quantidades ou qualquer outro dado — isso é inadmissível. Exemplo: se o usuário pede "incluir um funcionário" sem informar nome, departamento, cargo, data de admissão e salário, responda listando esses campos e pedindo os valores.
- REGRA CRÍTICA PARA RESPOSTAS: NUNCA inclua dados fictícios, inventados ou "ilustrativos" na resposta. Se uma informação não veio de uma ferramenta (tool call), NÃO a inclua. Responda SOMENTE com dados retornados pelas ferramentas. Se a ferramenta não retornou a informação solicitada, diga explicitamente que o dado não está disponível.
- Toda afirmação numérica da sua resposta deve vir do retorno de uma ferramenta.
- NUNCA decida você mesmo o resultado de uma regra de negócio: mesmo que a operação pareça inválida, execute a chamada e deixe a API validar — a regra vive na API, não em você. \
Isso vale MESMO quando uma leitura prévia indica que vai falhar (ex.: saldo insuficiente para a reserva pedida): execute a operação assim mesmo e reporte a resposta oficial da API (422 com `detail`), nunca a sua dedução.
- Quando o usuário citar pessoa, produto ou recurso por nome, resolva primeiro o id/SKU com a ferramenta de listagem; nunca chute ids.
- Para perguntas sobre dados (salário, preço, saldo, totais), chame a ferramenta de leitura correspondente antes de concluir que a informação não está disponível — os retornos contêm todos os campos do recurso.
- Se uma ferramenta retornar status 422, leia os campos `detail` e `rule`, e então corrija a chamada (somente se a correção não exigir inventar dados) ou explique ao usuário a regra violada. Nunca finja sucesso.
- Se retornar status 404, verifique se o id está certo (resolva pelo nome via listagem) antes de concluir que o recurso não existe; ao concluir, cite o `detail`.
- Sua resposta final deve conter somente a resposta ao usuário, em português, curta, direta e fundamentada nos dados retornados — sem raciocínio intermediário.
"""


@dataclass(frozen=True)
class ToolTraceEntry:
    name: str
    args: dict[str, Any]
    status: int


@dataclass(frozen=True)
class AgentResult:
    final_answer: str
    tool_trace: tuple[ToolTraceEntry, ...]
    iters: int
    domain: str = ""
    elapsed_s: float = 0.0
    stop_reason: str = "answer"  # answer | max_iters | deadline

    def trace_as_dicts(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "args": t.args, "status": t.status} for t in self.tool_trace]


def build_system_prompt(domain: str, today: date | None = None) -> str:
    label = _DOMAIN_LABELS.get(domain, domain)
    return _SYSTEM_PROMPT_TEMPLATE.format(label=label, today=(today or date.today()).isoformat())


def _assistant_message(response: ChatResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {"function": {"name": call.name, "arguments": call.arguments}} for call in response.tool_calls
        ],
    }


def _execute_call(registry: ToolRegistry, domain: str, call: ToolCall) -> dict[str, Any]:
    try:
        return registry.execute(domain, call.name, call.arguments)
    except ToolNotFound as exc:
        return {
            "status": 0,
            "body": {
                "error": "unknown_tool",
                "detail": f"A ferramenta '{exc.name}' não existe neste domínio. Disponíveis: {', '.join(exc.known)}.",
                "rule": "use somente as ferramentas fornecidas",
            },
        }


def run_domain_agent(
    domain: str,
    task: str,
    *,
    max_iters: int = 6,
    registry: ToolRegistry | None = None,
    llm: OllamaClient | None = None,
    settings: Settings | None = None,
    deadline_s: float | None = None,
) -> AgentResult:
    """Executa uma task no subagente do domínio até resposta final ou limite."""
    settings = settings or load_settings()
    registry = registry or ToolRegistry(
        settings.service_urls,
        timeout_s=settings.http_timeout_s,
        internal_api_key=settings.internal_api_key,
    )
    llm = llm or OllamaClient(settings.ollama_url, settings.model, timeout_s=settings.llm_timeout_s, keep_alive=settings.keep_alive)
    deadline_s = deadline_s if deadline_s is not None else settings.agent_deadline_s

    tools = registry.tools_for(domain)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(domain)},
        {"role": "user", "content": task},
    ]
    trace: list[ToolTraceEntry] = []
    started = time.monotonic()
    last_content = ""
    stop_reason = "max_iters"
    iters = 0

    for _ in range(max_iters):
        iters += 1
        response = llm.chat(messages, tools=tools)
        last_content = response.content or last_content

        if not response.has_tool_calls:
            stop_reason = "answer"
            break

        messages.append(_assistant_message(response))
        for call in response.tool_calls:
            result = _execute_call(registry, domain, call)
            trace.append(ToolTraceEntry(name=call.name, args=call.arguments, status=result["status"]))
            messages.append(
                {
                    "role": "tool",
                    "tool_name": call.name,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                }
            )

        if time.monotonic() - started > deadline_s:
            stop_reason = "deadline"
            break

    final_answer = last_content
    if stop_reason != "answer" and not final_answer:
        final_answer = (
            "Não foi possível concluir a tarefa dentro dos limites de execução "
            f"({'iterações' if stop_reason == 'max_iters' else 'tempo'})."
        )
    return AgentResult(
        final_answer=final_answer,
        tool_trace=tuple(trace),
        iters=iters,
        domain=domain,
        elapsed_s=round(time.monotonic() - started, 2),
        stop_reason=stop_reason,
    )


@dataclass
class DomainAgentRunner:
    """Reusa registry + cliente LLM entre tasks (evita refetch de OpenAPI)."""

    settings: Settings = field(default_factory=load_settings)

    def __post_init__(self) -> None:
        self._registry = ToolRegistry(
            self.settings.service_urls,
            timeout_s=self.settings.http_timeout_s,
            internal_api_key=self.settings.internal_api_key,
        )
        self._llm = OllamaClient(self.settings.ollama_url, self.settings.model, timeout_s=self.settings.llm_timeout_s, keep_alive=self.settings.keep_alive)

    def run(self, domain: str, task: str, *, max_iters: int | None = None) -> AgentResult:
        return run_domain_agent(
            domain,
            task,
            max_iters=max_iters or self.settings.agent_max_iters,
            registry=self._registry,
            llm=self._llm,
            settings=self.settings,
        )
