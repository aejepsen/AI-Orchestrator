import { useCallback, useEffect, useRef, useState } from "react";

import { AgentCard } from "./components/AgentCard";
import { AgentSkeleton } from "./components/AgentSkeleton";
import { Composer } from "./components/Composer";
import { Pipeline, type Stage } from "./components/Pipeline";
import { RouteChips } from "./components/RouteChips";
import { Unlock } from "./components/Unlock";
import { ChatHttpError, streamChat, type RoutePlan } from "./lib/sse";

const TOKEN_KEY = "aio:access-token";
const UNLOCKED_KEY = "aio:unlocked";

/** Sanitize é local e rápido; depois disso o gateway está roteando. */
const ROUTE_STAGE_DELAY_MS = 600;

type ItemBody =
  | { kind: "user"; text: string }
  | { kind: "route"; route: RoutePlan }
  | { kind: "agent"; domain: string; answer: string }
  | { kind: "clarification"; text: string }
  | { kind: "final"; text: string }
  | { kind: "aborted" }
  | { kind: "error"; text: string };

type Item = ItemBody & { id: string };

let seq = 0;
const nextId = () => `item-${++seq}`;

export default function App() {
  const [unlocked, setUnlocked] = useState(() => localStorage.getItem(UNLOCKED_KEY) === "1");
  const [authError, setAuthError] = useState<string | null>(null);
  const [items, setItems] = useState<Item[]>([]);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState<Stage>("sanitize");
  const [startedAt, setStartedAt] = useState(0);
  const [pendingAgents, setPendingAgents] = useState<string[]>([]);
  const endRef = useRef<HTMLDivElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [items, busy, pendingAgents]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const append = useCallback((item: ItemBody) => {
    setItems((prev) => [...prev, { ...item, id: nextId() }]);
  }, []);

  function unlock(token: string) {
    localStorage.setItem(TOKEN_KEY, token);
    localStorage.setItem(UNLOCKED_KEY, "1");
    setAuthError(null);
    setUnlocked(true);
  }

  function stop() {
    abortRef.current?.abort();
  }

  async function send(question: string) {
    append({ kind: "user", text: question });
    setBusy(true);
    setStage("sanitize");
    setStartedAt(Date.now());
    setPendingAgents([]);

    const controller = new AbortController();
    abortRef.current = controller;

    let expectedAgents = 0;
    let finishedAgents = 0;
    let routed = false;

    // Sanitize é quase instantâneo; sem evento dedicado, avançamos para
    // "roteamento" após um curto delay, até o evento `route` chegar.
    const routeTimer = window.setTimeout(() => {
      if (!routed && !controller.signal.aborted) setStage("route");
    }, ROUTE_STAGE_DELAY_MS);

    try {
      await streamChat(
        question,
        localStorage.getItem(TOKEN_KEY),
        (event) => {
          switch (event.type) {
            case "route":
              routed = true;
              expectedAgents = event.data.domains.length;
              if (event.data.clarification) {
                setStage("synthesize");
                append({ kind: "clarification", text: event.data.clarification });
              } else {
                setStage("agents");
                setPendingAgents(event.data.domains);
                append({ kind: "route", route: event.data });
              }
              break;
            case "agent":
              finishedAgents += 1;
              setPendingAgents((prev) => prev.filter((d) => d !== event.data.domain));
              if (expectedAgents > 0 && finishedAgents >= expectedAgents) setStage("synthesize");
              append({ kind: "agent", domain: event.data.domain, answer: event.data.answer });
              break;
            case "final":
              append({ kind: "final", text: event.data.answer });
              break;
            case "error":
              append({ kind: "error", text: event.data.detail ?? "Falha não tratada no grafo." });
              break;
          }
        },
        controller.signal,
      );
    } catch (err) {
      if (controller.signal.aborted) {
        append({ kind: "aborted" });
      } else if (err instanceof ChatHttpError && err.status === 401) {
        localStorage.removeItem(UNLOCKED_KEY);
        setAuthError("Token recusado pelo gateway. Verifique e tente de novo.");
        setUnlocked(false);
      } else if (err instanceof ChatHttpError && err.status === 429) {
        append({ kind: "error", text: "Gateway ocupado — limite de consultas atingido. Aguarde alguns minutos." });
      } else if (err instanceof ChatHttpError) {
        append({ kind: "error", text: err.message });
      } else {
        append({ kind: "error", text: "Conexão com o gateway interrompida. Tente novamente." });
      }
    } finally {
      window.clearTimeout(routeTimer);
      abortRef.current = null;
      setPendingAgents([]);
      setBusy(false);
    }
  }

  if (!unlocked) return <Unlock error={authError} onUnlock={unlock} />;

  return (
    <div className="flex min-h-dvh flex-col">
      <header className="sticky top-0 z-10 border-b border-line bg-bg/85 backdrop-blur">
        <div className="mx-auto flex w-full max-w-3xl items-baseline gap-3 px-4 py-4 sm:px-6">
          <h1 className="text-sm font-semibold tracking-tight">AI-Orchestrator</h1>
          <p className="text-xs text-faint">Gateway multi-agente on-premise</p>
          <a
            href="/apresentacao/"
            target="_blank"
            rel="noopener"
            className="ml-auto self-center rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-muted transition-colors hover:border-emerald-400/40 hover:text-ink"
          >
            Apresentação ↗
          </a>
        </div>
      </header>

      <main className="mx-auto w-full max-w-3xl flex-1 px-4 py-6 sm:px-6">
        {items.length === 0 && !busy && (
          <div className="animate-fade-up mt-[18vh] space-y-3 text-center">
            <p className="text-xl font-medium tracking-[-0.02em] text-ink">O que você quer saber do negócio?</p>
            <p className="mx-auto max-w-md text-sm leading-relaxed text-muted">
              Uma pergunta entra, é roteada para os agentes de finanças, RH, estoque e vendas, e volta sintetizada —
              tudo rodando local, sem nuvem.
            </p>
          </div>
        )}

        <div className="space-y-4">
          {items.map((item) => {
            switch (item.kind) {
              case "user":
                return (
                  <div key={item.id} className="animate-fade-up flex justify-end">
                    <p className="max-w-[85%] rounded-2xl rounded-br-md bg-raised px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap text-ink">
                      {item.text}
                    </p>
                  </div>
                );
              case "route":
                return <RouteChips key={item.id} route={item.route} />;
              case "agent":
                return <AgentCard key={item.id} domain={item.domain} answer={item.answer} />;
              case "clarification":
                return (
                  <div
                    key={item.id}
                    className="animate-fade-up rounded-lg border border-amber-400/20 bg-amber-400/5 px-4 py-3"
                  >
                    <p className="font-mono text-[11px] tracking-wide text-amber-300 uppercase">precisa de contexto</p>
                    <p className="mt-1 text-sm leading-relaxed whitespace-pre-wrap text-ink/90">{item.text}</p>
                  </div>
                );
              case "final":
                return (
                  <div key={item.id} className="animate-fade-up flex justify-start">
                    <p className="max-w-[92%] rounded-2xl rounded-bl-md border border-line bg-surface px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap text-ink">
                      {item.text}
                    </p>
                  </div>
                );
              case "aborted":
                return (
                  <p key={item.id} className="animate-fade-up font-mono text-[11px] tracking-wide text-faint uppercase">
                    interrompido
                  </p>
                );
              case "error":
                return (
                  <div
                    key={item.id}
                    className="animate-fade-up rounded-lg border border-red-400/20 bg-red-400/5 px-4 py-3"
                  >
                    <p className="font-mono text-[11px] tracking-wide text-red-400 uppercase">erro</p>
                    <p className="mt-1 text-sm leading-relaxed text-red-200/80">{item.text}</p>
                  </div>
                );
            }
          })}
          {busy && pendingAgents.map((domain) => <AgentSkeleton key={`skeleton-${domain}`} domain={domain} />)}
          {busy && <Pipeline stage={stage} startedAt={startedAt} />}
        </div>
        <div ref={endRef} className="h-px" />
      </main>

      <footer className="sticky bottom-0 z-10 border-t border-line bg-bg/85 backdrop-blur">
        <div className="mx-auto w-full max-w-3xl px-4 py-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] sm:px-6">
          <Composer disabled={busy} onSend={send} onStop={stop} />
        </div>
      </footer>
    </div>
  );
}
