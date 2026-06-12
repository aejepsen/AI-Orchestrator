"""Grafo do orquestrador (LangGraph StateGraph).

Topologia: sanitize → classify → (clarification? → respond_clarification)
                                  senão → dispatch (fan-out paralelo) → synthesize.

Decisões:
- PoC stateless por request: compile sem checkpointer persistente.
- `dispatch` paraleliza subagentes com ThreadPoolExecutor: o Ollama serializa
  de fato as gerações, mas o paralelismo vale para os microsserviços e mantém
  o desenho fan-out/fan-in correto para quando houver mais capacidade.
- `synthesize` com 1 domínio devolve a resposta do agente direto (zero
  chamada extra ao LLM); >1 domínio o MoE sintetiza fundado só nas respostas.
- Observabilidade (base da Fase 5): logger "gateway", um log JSON por nó com
  trace_id, nó, latência ms e domínios.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from gateway.agents import DomainAgentRunner
from gateway.llm import OllamaClient
from gateway.router import classify_intent
from gateway.sanitize import sanitize_question

logger = logging.getLogger("gateway")

# Callback opcional invocado a cada subagente concluído: (domain, answer).
AgentCallback = Callable[[str, str], None]


class GraphState(TypedDict, total=False):
    question: str
    sanitized: str
    route: dict[str, Any]
    agent_results: dict[str, dict[str, Any]]
    final_answer: str
    trace_id: str
    error: str | None
    # Canal interno (não serializado): callback por subagente concluído.
    _on_agent: Any


_SYNTH_SYSTEM = """Você é o orquestrador de um sistema corporativo multi-agente.
Agentes especialistas já responderam à pergunta do usuário, cada um no seu domínio.
Sintetize UMA resposta final em português, curta e direta, fundamentada EXCLUSIVAMENTE
nas respostas dos agentes — nunca invente números ou fatos que não estejam nelas.
Se algum agente reportou impedimento ou erro, reflita isso na resposta final."""


def _log_node(state: GraphState, node: str, started: float, domains: list[str] | None = None) -> None:
    logger.info(
        json.dumps(
            {
                "trace_id": state.get("trace_id", ""),
                "node": node,
                "latency_ms": round((time.monotonic() - started) * 1000, 1),
                "domains": domains if domains is not None else state.get("route", {}).get("domains", []),
            },
            ensure_ascii=False,
        )
    )


class GatewayGraph:
    """Grafo compilado, reusável entre requests (runner e LLM cacheados)."""

    def __init__(
        self,
        runner: DomainAgentRunner,
        llm: OllamaClient | None = None,
        semantic: "SemanticRouter | None" = None,
    ) -> None:
        self._runner = runner
        self._llm = llm or OllamaClient(
            runner.settings.ollama_url,
            runner.settings.model,
            timeout_s=runner.settings.llm_timeout_s,
            keep_alive=runner.settings.keep_alive,
        )
        settings = getattr(runner, "settings", None)
        if semantic is None and settings is not None and settings.semantic_enabled:
            from gateway.semantic_router import SemanticRouter

            semantic = SemanticRouter(
                settings.qdrant_url,
                self._llm,
                embed_model=settings.embed_model,
                examples_path=settings.routing_examples_path,
                threshold=settings.semantic_threshold,
                top_k=settings.semantic_top_k,
            )
        self._semantic = semantic
        self._compiled = self._build()

    # -- nós -----------------------------------------------------------------

    def _sanitize(self, state: GraphState) -> GraphState:
        started = time.monotonic()
        update: GraphState = {"sanitized": sanitize_question(state["question"])}
        if not state.get("trace_id"):
            update["trace_id"] = str(uuid.uuid4())
        _log_node({**state, **update}, "sanitize", started, domains=[])
        return update

    def _classify(self, state: GraphState) -> GraphState:
        started = time.monotonic()
        route = classify_intent(state["sanitized"], self._llm, semantic=self._semantic)
        update: GraphState = {"route": route.model_dump()}
        _log_node({**state, **update}, "classify", started)
        return update

    def _respond_clarification(self, state: GraphState) -> GraphState:
        started = time.monotonic()
        update: GraphState = {
            "final_answer": state["route"].get("clarification") or "Pode esclarecer sua pergunta?",
            "agent_results": {},
        }
        _log_node(state, "respond_clarification", started, domains=[])
        return update

    def _dispatch(self, state: GraphState) -> GraphState:
        started = time.monotonic()
        route = state["route"]
        domains: list[str] = route.get("domains", [])
        plan = route.get("plan", "")
        task = state["sanitized"]
        if plan:
            task = f"{task}\n\n(Nota do orquestrador: {plan})"
        on_agent: AgentCallback | None = state.get("_on_agent")  # type: ignore[typeddict-item]

        def run_one(domain: str) -> tuple[str, dict[str, Any]]:
            try:
                result = self._runner.run(domain, task)
                payload = {"answer": result.final_answer, "trace": result.trace_as_dicts()}
            except Exception as exc:  # noqa: BLE001 — falha de um agente não derruba o fan-out
                payload = {"answer": f"O agente de {domain} falhou: {exc}", "trace": [], "error": str(exc)}
            if on_agent:
                on_agent(domain, payload["answer"])
            return domain, payload

        if len(domains) == 1:
            results = dict([run_one(domains[0])])
        else:
            with ThreadPoolExecutor(max_workers=len(domains)) as pool:
                results = dict(pool.map(run_one, domains))
        _log_node(state, "dispatch", started, domains=domains)
        return {"agent_results": results}

    def _synthesize(self, state: GraphState) -> GraphState:
        started = time.monotonic()
        results = state["agent_results"]
        domains = state["route"].get("domains", [])
        if len(domains) == 1:
            answer = results[domains[0]]["answer"]
        else:
            blocks = "\n".join(f"[{domain}]\n{results[domain]['answer']}" for domain in domains)
            response = self._llm.chat(
                [
                    {"role": "system", "content": _SYNTH_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"<user_question>\n{state['sanitized']}\n</user_question>\n\n"
                            f"<agent_answers>\n{blocks}\n</agent_answers>"
                        ),
                    },
                ]
            )
            answer = response.content
        _log_node(state, "synthesize", started, domains=domains)
        return {"final_answer": answer}

    # -- montagem ------------------------------------------------------------

    def _build(self):
        graph = StateGraph(GraphState)
        graph.add_node("sanitize", self._sanitize)
        graph.add_node("classify", self._classify)
        graph.add_node("respond_clarification", self._respond_clarification)
        graph.add_node("dispatch", self._dispatch)
        graph.add_node("synthesize", self._synthesize)

        graph.set_entry_point("sanitize")
        graph.add_edge("sanitize", "classify")
        graph.add_conditional_edges(
            "classify",
            lambda state: "respond_clarification" if state["route"].get("clarification") else "dispatch",
            {"respond_clarification": "respond_clarification", "dispatch": "dispatch"},
        )
        graph.add_edge("dispatch", "synthesize")
        graph.add_edge("respond_clarification", END)
        graph.add_edge("synthesize", END)
        return graph.compile()

    # -- API -----------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        on_agent: AgentCallback | None = None,
        trace_id: str | None = None,
    ) -> GraphState:
        """Executa o grafo para uma pergunta; retorna o estado final."""
        state: dict[str, Any] = {"question": question, "trace_id": trace_id or str(uuid.uuid4())}
        if on_agent:
            state["_on_agent"] = on_agent
        return self._compiled.invoke(state)

    def stream(
        self,
        question: str,
        *,
        on_agent: AgentCallback | None = None,
        trace_id: str | None = None,
    ):
        """Itera updates por nó (stream_mode='updates') — base do SSE."""
        state: dict[str, Any] = {"question": question, "trace_id": trace_id or str(uuid.uuid4())}
        if on_agent:
            state["_on_agent"] = on_agent
        yield from self._compiled.stream(state, stream_mode="updates")
