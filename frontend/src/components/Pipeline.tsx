import { useEffect, useState } from "react";

export type Stage = "sanitize" | "route" | "agents" | "synthesize";

const STAGES: { id: Stage; label: string }[] = [
  { id: "sanitize", label: "sanitize" },
  { id: "route", label: "roteamento" },
  { id: "agents", label: "agentes" },
  { id: "synthesize", label: "síntese" },
];

function formatElapsed(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

export function Pipeline({ stage, startedAt }: { stage: Stage; startedAt: number }) {
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const elapsed = Math.max(0, Math.floor((now - startedAt) / 1000));
  const currentIndex = STAGES.findIndex((s) => s.id === stage);

  return (
    <div className="animate-fade-up rounded-xl border border-line bg-surface/60 px-4 py-3">
      <div className="flex flex-wrap items-center gap-x-1 gap-y-2">
        {STAGES.map((s, i) => {
          const isCurrent = i === currentIndex;
          const isDone = i < currentIndex;
          return (
            <span key={s.id} className="flex items-center gap-1">
              <span
                className={[
                  "font-mono text-[11px] tracking-wide uppercase transition-colors duration-500",
                  isCurrent ? "text-ink" : isDone ? "text-muted" : "text-faint",
                ].join(" ")}
              >
                {isCurrent && (
                  <span className="animate-pulse-dot mr-1.5 inline-block size-1.5 rounded-full bg-ink align-middle" />
                )}
                {s.label}
              </span>
              {i < STAGES.length - 1 && <span className="px-1 text-faint">→</span>}
            </span>
          );
        })}
        <span className="ml-auto font-mono text-[11px] tabular-nums text-muted">{formatElapsed(elapsed)}</span>
      </div>
      <p className="mt-2 text-xs leading-relaxed text-faint">
        Modelo local (30B MoE em GPU consumer) — consultas multi-domínio levam alguns minutos.
      </p>
    </div>
  );
}
