"""Grafo do orquestrador (LangGraph StateGraph).

Topologia: sanitize → classify → (clarification? → respond_clarification)
                                  senão → confirm_dispatch → dispatch → synthesize.

Decisões:
- MemorySaver como checkpointer in-memory (produção: SqliteSaver ou PostgresSaver).
- `dispatch` paraleliza subagentes com ThreadPoolExecutor: o Ollama serializa
  de fato as gerações, mas o paralelismo vale para os microsserviços e mantém
  o desenho fan-out/fan-in correto para quando houver mais capacidade.
- `synthesize` com 1 domínio devolve a resposta do agente direto (zero
  chamada extra ao LLM); >1 domínio o MoE sintetiza fundado só nas respostas.
- Observabilidade Langfuse: trace por request, span por nó, generation por LLM call.
- HITL: confirm_dispatch pausa para aprovação humana via interrupt().
- Estado conversacional: history acumulado via thread_id + checkpointer.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from gateway.agents import DomainAgentRunner
from gateway.config import Settings
from gateway.llm import OllamaClient
from gateway.router import classify_intent
from gateway.sanitize import flag_injection, sanitize_question
from gateway.tracing import TraceHandle, Tracer

logger = logging.getLogger("gateway")

# Callback opcional invocado a cada subagente concluído: (domain, answer).
AgentCallback = Callable[[str, str], None]
# Callback opcional invocado quando confirmação é necessária: (domains, plan).
ConfirmCallback = Callable[[list[str], str], None]


class GraphState(TypedDict, total=False):
    question: str
    sanitized: str
    route: dict[str, Any]
    agent_results: dict[str, dict[str, Any]]
    final_answer: str
    trace_id: str
    error: str | None
    # Estado conversacional (multi-turn).
    history: list[dict[str, str]]  # [{"role": "user", "content": ...}, ...]
    # HITL
    pending_confirmation: dict[str, Any] | None
    # Segurança: flag de injection semântica detectada.
    _injection_suspect: bool


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
        tracer: Tracer | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._runner = runner
        self._llm = llm or OllamaClient(
            runner.settings.ollama_url,
            runner.settings.model,
            timeout_s=runner.settings.llm_timeout_s,
            keep_alive=runner.settings.keep_alive,
        )
        self._settings = settings or getattr(runner, "settings", None)
        if semantic is None and self._settings is not None and self._settings.semantic_enabled:
            from gateway.semantic_router import SemanticRouter

            # Tenta SBERT (CPU, sem dependência de Ollama para embeddings).
            # Fallback para OllamaEmbedder se sentence-transformers não instalado.
            try:
                from gateway.embedder import SBERTEmbedder

                embedder = SBERTEmbedder(
                    model_name=self._settings.sbert_model,
                    cache_dir=self._settings.sbert_cache_dir,
                )
                logger.info("Embedder: SBERTEmbedder (%s)", self._settings.sbert_model)
            except Exception:  # noqa: BLE001
                from gateway.embedder import OllamaEmbedder

                embedder = OllamaEmbedder(self._llm, model=self._settings.embed_model)
                logger.info("Embedder: OllamaEmbedder fallback (%s)", self._settings.embed_model)

            semantic = SemanticRouter(
                self._settings.qdrant_url,
                embedder,
                examples_path=self._settings.routing_examples_path,
                threshold=self._settings.semantic_threshold,
                top_k=self._settings.semantic_top_k,
            )
        self._semantic = semantic
        self._tracer = tracer

        # Injection Detector (BERTimbau). Fallback regex se indisponível.
        self._injection_detector = None
        if self._settings and self._settings.injection_detector_enabled:
            from gateway.injection_detector import InjectionDetector

            self._injection_detector = InjectionDetector(
                model_path=self._settings.injection_model,
                threshold=self._settings.injection_threshold,
            )

        # Checkpointer: SqliteSaver se THREAD_DB_PATH configurado, senão MemorySaver.
        try:
            if self._settings and self._settings.thread_db_path:
                try:
                    from langgraph.checkpoint.sqlite import SqliteSaver

                    self._checkpointer = SqliteSaver.from_conn_string(self._settings.thread_db_path)
                    logger.info("Checkpointer: SqliteSaver (%s)", self._settings.thread_db_path)
                except ImportError:
                    logger.warning("langgraph-checkpoint-sqlite não instalado; usando MemorySaver (in-memory)")
                    from langgraph.checkpoint.memory import MemorySaver

                    self._checkpointer = MemorySaver()
            else:
                from langgraph.checkpoint.memory import MemorySaver

                self._checkpointer = MemorySaver()
        except ImportError:
            self._checkpointer = None

        self._compiled = self._build()

        # Thread-local: callbacks e trace não vão pro state (checkpointer não serializa lambdas).
        self._local = threading.local()

    # -- nós -----------------------------------------------------------------

    def _sanitize(self, state: GraphState) -> GraphState:
        started = time.monotonic()
        trace: TraceHandle | None = getattr(self._local, "trace", None)
        span = trace.span(name="sanitize") if trace else None
        # Limpa estado residual de turns anteriores (checkpointer mantém tudo).
        update: GraphState = {
            "sanitized": sanitize_question(state["question"]),
            "_injection_suspect": flag_injection(state["question"], detector=self._injection_detector),
            "final_answer": "",
            "agent_results": {},
            "route": {},
            "error": None,
            "pending_confirmation": None,
        }
        if not state.get("trace_id"):
            update["trace_id"] = str(uuid.uuid4())
        if update.get("_injection_suspect"):
            logger.warning("injection_suspect detected in question: %s", state["question"][:120])
        _log_node({**state, **update}, "sanitize", started, domains=[])
        if span:
            span.end()
        return update

    def _classify(self, state: GraphState) -> GraphState:
        started = time.monotonic()
        trace: TraceHandle | None = getattr(self._local, "trace", None)
        span = trace.span(name="classify") if trace else None
        route = classify_intent(state["sanitized"], self._llm, semantic=self._semantic)
        update: GraphState = {"route": route.model_dump()}
        _log_node({**state, **update}, "classify", started)
        if span:
            span.end()
        return update

    def _respond_clarification(self, state: GraphState) -> GraphState:
        started = time.monotonic()
        trace: TraceHandle | None = getattr(self._local, "trace", None)
        span = trace.span(name="respond_clarification") if trace else None
        answer = state["route"].get("clarification") or "Pode esclarecer sua pergunta?"
        update: GraphState = {
            "final_answer": answer,
            "agent_results": {},
        }
        _log_node(state, "respond_clarification", started, domains=[])
        if span:
            span.end()
        return update

    def _confirm_dispatch(self, state: GraphState) -> GraphState:
        """Pausa para confirmação humana antes de executar agentes.

        Só ativa interrupt() quando há callback de confirmação (on_confirm).
        Sem callback (testes, evals, modo automático): auto-aprova.
        """
        on_confirm: ConfirmCallback | None = getattr(self._local, "on_confirm", None)

        # Sem callback = modo automático: prossegue direto pro dispatch.
        if not on_confirm:
            return {}

        route = state["route"]
        domains = route.get("domains", [])
        plan = route.get("plan", "")

        # Notifica frontend sobre confirmação pendente.
        on_confirm(domains, plan)

        # Tenta usar interrupt() do LangGraph para HITL.
        try:
            from langgraph.types import interrupt

            decision = interrupt({
                "type": "confirm_dispatch",
                "domains": domains,
                "plan": plan,
                "question": state["sanitized"],
            })

            if not decision.get("approved", False):
                return {
                    "final_answer": "Operação cancelada pelo usuário.",
                    "agent_results": {},
                }
        except ImportError:
            pass

        return {}

    def _dispatch(self, state: GraphState) -> GraphState:
        started = time.monotonic()
        trace: TraceHandle | None = getattr(self._local, "trace", None)
        span = trace.span(name="dispatch") if trace else None
        route = state["route"]
        domains: list[str] = route.get("domains", [])
        plan = route.get("plan", "")
        task = state["sanitized"]
        if plan:
            task = f"{task}\n\n(Nota do orquestrador: {plan})"
        on_agent: AgentCallback | None = getattr(self._local, "on_agent", None)

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
        if span:
            span.end()
        return {"agent_results": results}

    def _synthesize(self, state: GraphState) -> GraphState:
        started = time.monotonic()
        trace: TraceHandle | None = getattr(self._local, "trace", None)
        span = trace.span(name="synthesize") if trace else None
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
                ],
                trace=trace,
            )
            answer = response.content

        # Acumula histórico conversacional.
        history = list(state.get("history") or [])
        history.append({"role": "user", "content": state.get("question", state.get("sanitized", ""))})
        history.append({"role": "assistant", "content": answer})

        _log_node(state, "synthesize", started, domains=domains)
        if span:
            span.end()
        return {"final_answer": answer, "history": history}

    # -- montagem ------------------------------------------------------------

    @staticmethod
    def _route_after_confirm(state: GraphState) -> str:
        """Após confirm_dispatch: se rejeitado (final_answer existe), vai pra END."""
        if state.get("final_answer"):
            return "end"
        return "dispatch"

    def _build(self):
        graph = StateGraph(GraphState)
        graph.add_node("sanitize", self._sanitize)
        graph.add_node("classify", self._classify)
        graph.add_node("respond_clarification", self._respond_clarification)
        graph.add_node("confirm_dispatch", self._confirm_dispatch)
        graph.add_node("dispatch", self._dispatch)
        graph.add_node("synthesize", self._synthesize)

        graph.set_entry_point("sanitize")
        graph.add_edge("sanitize", "classify")
        graph.add_conditional_edges(
            "classify",
            lambda state: "respond_clarification" if state["route"].get("clarification") else "confirm_dispatch",
            {"respond_clarification": "respond_clarification", "confirm_dispatch": "confirm_dispatch"},
        )
        graph.add_conditional_edges(
            "confirm_dispatch",
            self._route_after_confirm,
            {"dispatch": "dispatch", "end": END},
        )
        graph.add_edge("dispatch", "synthesize")
        graph.add_edge("respond_clarification", END)
        graph.add_edge("synthesize", END)

        return graph.compile(checkpointer=self._checkpointer)

    # -- API -----------------------------------------------------------------

    def run(
        self,
        question: str,
        *,
        on_agent: AgentCallback | None = None,
        trace_id: str | None = None,
        thread_id: str | None = None,
        on_confirm: ConfirmCallback | None = None,
    ) -> GraphState:
        """Executa o grafo para uma pergunta; retorna o estado final."""
        tid = trace_id or str(uuid.uuid4())
        state: dict[str, Any] = {"question": question, "trace_id": tid}

        # Callbacks e trace em thread-local (não serializáveis pelo checkpointer).
        self._local.on_agent = on_agent
        self._local.on_confirm = on_confirm
        self._local.trace = self._tracer.trace(trace_id=tid, name="chat") if self._tracer else None

        config: dict[str, Any] | None = None
        if self._checkpointer:
            config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

        result = self._compiled.invoke(state, config=config)

        if self._tracer:
            self._tracer.flush()

        return result

    def stream(
        self,
        question: str,
        *,
        on_agent: AgentCallback | None = None,
        trace_id: str | None = None,
        thread_id: str | None = None,
        on_confirm: ConfirmCallback | None = None,
    ):
        """Itera updates por nó (stream_mode='updates') — base do SSE."""
        tid = trace_id or str(uuid.uuid4())
        state: dict[str, Any] = {"question": question, "trace_id": tid}

        self._local.on_agent = on_agent
        self._local.on_confirm = on_confirm
        self._local.trace = self._tracer.trace(trace_id=tid, name="chat") if self._tracer else None

        config: dict[str, Any] | None = None
        if self._checkpointer:
            config = {"configurable": {"thread_id": thread_id or str(uuid.uuid4())}}

        yield from self._compiled.stream(state, config=config, stream_mode="updates")

        if self._tracer:
            self._tracer.flush()
