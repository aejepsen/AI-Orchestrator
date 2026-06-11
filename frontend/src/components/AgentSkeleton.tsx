import { domainMeta } from "../lib/domains";

/**
 * Placeholder de AgentCard exibido durante o fan-out: mesma estrutura
 * e borda colorida do domínio, com shimmer enquanto o agente trabalha.
 */
export function AgentSkeleton({ domain }: { domain: string }) {
  const meta = domainMeta(domain);
  return (
    <article
      aria-busy="true"
      className={`animate-fade-up rounded-lg border border-line border-l-2 bg-surface/80 px-4 py-3 ${meta.border}`}
    >
      <header className="mb-1.5 flex items-center gap-2">
        <span className={`animate-pulse-dot size-1.5 rounded-full ${meta.dot}`} />
        <span className={`font-mono text-[11px] font-medium tracking-wide uppercase ${meta.text}`}>{meta.label}</span>
        <span className="text-[11px] text-faint">consultando…</span>
      </header>
      <div className="space-y-2">
        <div className="animate-shimmer h-3 w-full rounded bg-raised" />
        <div className="animate-shimmer h-3 w-4/5 rounded bg-raised" style={{ animationDelay: "0.15s" }} />
        <div className="animate-shimmer h-3 w-3/5 rounded bg-raised" style={{ animationDelay: "0.3s" }} />
      </div>
    </article>
  );
}
