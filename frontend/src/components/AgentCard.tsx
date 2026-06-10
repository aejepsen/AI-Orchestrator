import { domainMeta } from "../lib/domains";

export function AgentCard({ domain, answer }: { domain: string; answer: string }) {
  const meta = domainMeta(domain);
  return (
    <article
      className={`animate-fade-up rounded-lg border border-line border-l-2 bg-surface/80 px-4 py-3 ${meta.border}`}
    >
      <header className="mb-1.5 flex items-center gap-2">
        <span className={`size-1.5 rounded-full ${meta.dot}`} />
        <span className={`font-mono text-[11px] font-medium tracking-wide uppercase ${meta.text}`}>{meta.label}</span>
        <span className="text-[11px] text-faint">agente concluído</span>
      </header>
      <p className="text-sm leading-relaxed whitespace-pre-wrap text-ink/90">{answer}</p>
    </article>
  );
}
