import { useCallback, useEffect, useRef, useState } from "react";

/* ───────── Types ───────── */

interface LatencyMetrics {
  avg_ms: number;
  p50_ms: number;
  p95_ms: number;
  sample_size: number;
}

interface TokenMetrics {
  input: number;
  output: number;
  total: number;
}

interface RoutingMetrics {
  semantic: number;
  llm: number;
  unclassified: number;
}

interface Metrics {
  available: boolean;
  stale: boolean;
  total_traces: number;
  latency: LatencyMetrics;
  tokens: TokenMetrics;
  routing: RoutingMetrics;
  injection_blocks: number;
  error_count: number;
}

const TOKEN_KEY = "aio:access-token";
const REFRESH_INTERVAL_MS = 30_000;

/* ───────── Helpers ───────── */

function fmt(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return n.toLocaleString("pt-BR");
}

function fmtMs(n: number): string {
  if (n >= 1000) return `${(n / 1000).toFixed(2)}s`;
  return `${n.toFixed(0)}ms`;
}

function pct(value: number, total: number): number {
  if (total === 0) return 0;
  return Math.round((value / total) * 100);
}

/* ───────── Skeleton ───────── */

function CardSkeleton() {
  return (
    <div className="rounded-2xl border border-line bg-surface/50 p-5">
      <div className="mb-3 h-3 w-20 rounded animate-shimmer bg-raised" />
      <div className="h-8 w-28 rounded animate-shimmer bg-raised" />
    </div>
  );
}

function DashboardSkeleton() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {Array.from({ length: 8 }).map((_, i) => (
        <CardSkeleton key={i} />
      ))}
    </div>
  );
}

/* ───────── Stat Card ───────── */

function StatCard({
  label,
  value,
  sub,
  accent,
}: {
  label: string;
  value: string;
  sub?: string;
  accent?: string;
}) {
  return (
    <div className="group relative overflow-hidden rounded-2xl border border-line bg-surface/40 p-5 backdrop-blur-sm transition-all duration-300 hover:border-line-strong hover:bg-surface/60">
      {/* Glassmorphism glow */}
      <div
        className="pointer-events-none absolute -right-6 -top-6 h-24 w-24 rounded-full opacity-0 blur-2xl transition-opacity duration-500 group-hover:opacity-100"
        style={{ background: accent || "rgba(52, 211, 153, 0.08)" }}
      />
      <p className="font-mono text-[11px] tracking-widest text-faint uppercase">{label}</p>
      <p className="mt-2 text-2xl font-semibold tracking-tight text-ink">{value}</p>
      {sub && <p className="mt-1 text-xs text-muted">{sub}</p>}
    </div>
  );
}

/* ───────── Routing Bar ───────── */

function RoutingBar({ routing }: { routing: RoutingMetrics }) {
  const total = routing.semantic + routing.llm + routing.unclassified;
  if (total === 0) {
    return (
      <div className="col-span-full rounded-2xl border border-line bg-surface/40 p-5 backdrop-blur-sm">
        <p className="font-mono text-[11px] tracking-widest text-faint uppercase">Routing Layer</p>
        <p className="mt-3 text-sm text-muted">Sem dados de roteamento</p>
      </div>
    );
  }

  const segments = [
    { label: "Semantic", value: routing.semantic, color: "#34d399" },
    { label: "LLM", value: routing.llm, color: "#818cf8" },
    { label: "Outros", value: routing.unclassified, color: "#64748b" },
  ].filter((s) => s.value > 0);

  return (
    <div className="col-span-full rounded-2xl border border-line bg-surface/40 p-5 backdrop-blur-sm">
      <p className="font-mono text-[11px] tracking-widest text-faint uppercase">Routing Layer</p>
      {/* Bar */}
      <div className="mt-4 flex h-3 w-full overflow-hidden rounded-full bg-raised">
        {segments.map((seg) => (
          <div
            key={seg.label}
            className="transition-all duration-700 ease-out"
            style={{
              width: `${pct(seg.value, total)}%`,
              backgroundColor: seg.color,
              minWidth: seg.value > 0 ? "4px" : 0,
            }}
          />
        ))}
      </div>
      {/* Legend */}
      <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1">
        {segments.map((seg) => (
          <div key={seg.label} className="flex items-center gap-2 text-xs text-muted">
            <span className="inline-block h-2 w-2 rounded-full" style={{ backgroundColor: seg.color }} />
            {seg.label}: {fmt(seg.value)} ({pct(seg.value, total)}%)
          </div>
        ))}
      </div>
    </div>
  );
}

/* ───────── Latency Detail ───────── */

function LatencyDetail({ latency }: { latency: LatencyMetrics }) {
  const bars = [
    { label: "Avg", value: latency.avg_ms, color: "#34d399" },
    { label: "P50", value: latency.p50_ms, color: "#60a5fa" },
    { label: "P95", value: latency.p95_ms, color: "#f59e0b" },
  ];
  const max = Math.max(...bars.map((b) => b.value), 1);

  return (
    <div className="col-span-full rounded-2xl border border-line bg-surface/40 p-5 backdrop-blur-sm sm:col-span-2">
      <p className="font-mono text-[11px] tracking-widest text-faint uppercase">Distribuicao de Latencia</p>
      <div className="mt-4 space-y-3">
        {bars.map((bar) => (
          <div key={bar.label} className="flex items-center gap-3">
            <span className="w-8 text-right font-mono text-xs text-muted">{bar.label}</span>
            <div className="flex-1">
              <div className="h-5 w-full rounded-md bg-raised">
                <div
                  className="flex h-full items-center rounded-md px-2 text-[10px] font-medium text-bg transition-all duration-700 ease-out"
                  style={{
                    width: `${Math.max((bar.value / max) * 100, 8)}%`,
                    backgroundColor: bar.color,
                  }}
                >
                  {fmtMs(bar.value)}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11px] text-faint">{latency.sample_size} amostras</p>
    </div>
  );
}

/* ───────── Main Dashboard ───────── */

export function Dashboard() {
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchMetrics = useCallback(async () => {
    try {
      const headers: Record<string, string> = {};
      const token = localStorage.getItem(TOKEN_KEY);
      if (token) headers["X-Access-Token"] = token;

      const res = await fetch("/metrics", { headers });
      if (!res.ok) {
        if (res.status === 401) {
          setError("Nao autorizado. Verifique o token de acesso.");
          return;
        }
        throw new Error(`HTTP ${res.status}`);
      }
      const data: Metrics = await res.json();
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError("Falha ao conectar com o gateway.");
      // Mantém último estado válido
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchMetrics();
    intervalRef.current = setInterval(fetchMetrics, REFRESH_INTERVAL_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchMetrics]);

  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-8 sm:px-6">
      {/* Header */}
      <div className="mb-8 flex items-baseline justify-between">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-ink">Observabilidade</h2>
          <p className="mt-1 text-xs text-muted">Metricas agregadas via Langfuse</p>
        </div>
        <div className="flex items-center gap-3">
          {metrics && !metrics.available && (
            <span className="rounded-full bg-amber-400/10 px-2.5 py-1 text-[11px] font-medium text-amber-400">
              Langfuse offline
            </span>
          )}
          {metrics?.stale && (
            <span className="rounded-full bg-amber-400/10 px-2.5 py-1 text-[11px] font-medium text-amber-300">
              Dados em cache
            </span>
          )}
          {metrics?.available && !metrics.stale && (
            <span className="rounded-full bg-emerald-400/10 px-2.5 py-1 text-[11px] font-medium text-emerald-400">
              Live
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="mb-6 rounded-xl border border-red-400/20 bg-red-400/5 px-4 py-3">
          <p className="text-sm text-red-300">{error}</p>
        </div>
      )}

      {loading ? (
        <DashboardSkeleton />
      ) : metrics ? (
        <div className="animate-fade-up grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Total Requests"
            value={fmt(metrics.total_traces)}
            sub="Ultimas 200 traces"
            accent="rgba(52, 211, 153, 0.1)"
          />
          <StatCard
            label="Latencia Media"
            value={fmtMs(metrics.latency.avg_ms)}
            sub={`P50: ${fmtMs(metrics.latency.p50_ms)} · P95: ${fmtMs(metrics.latency.p95_ms)}`}
            accent="rgba(96, 165, 250, 0.1)"
          />
          <StatCard
            label="Tokens Consumidos"
            value={fmt(metrics.tokens.total)}
            sub={`In: ${fmt(metrics.tokens.input)} · Out: ${fmt(metrics.tokens.output)}`}
            accent="rgba(129, 140, 248, 0.1)"
          />
          <StatCard
            label="Erros"
            value={fmt(metrics.error_count)}
            sub={
              metrics.total_traces > 0
                ? `${pct(metrics.error_count, metrics.total_traces)}% do total`
                : undefined
            }
            accent="rgba(248, 113, 113, 0.1)"
          />
          <StatCard
            label="Injection Blocks"
            value={fmt(metrics.injection_blocks)}
            sub="Tentativas bloqueadas"
            accent="rgba(251, 191, 36, 0.1)"
          />
          <StatCard
            label="Roteamento Semantico"
            value={fmt(metrics.routing.semantic)}
            sub={
              metrics.total_traces > 0
                ? `${pct(metrics.routing.semantic, metrics.total_traces)}% do total`
                : undefined
            }
            accent="rgba(52, 211, 153, 0.1)"
          />
          <StatCard
            label="Roteamento LLM"
            value={fmt(metrics.routing.llm)}
            sub={
              metrics.total_traces > 0
                ? `${pct(metrics.routing.llm, metrics.total_traces)}% do total`
                : undefined
            }
            accent="rgba(129, 140, 248, 0.1)"
          />
          <StatCard
            label="Uptime"
            value={metrics.available ? "Online" : "Offline"}
            sub="Status do Langfuse"
            accent={metrics.available ? "rgba(52, 211, 153, 0.1)" : "rgba(248, 113, 113, 0.1)"}
          />

          {/* Full-width sections */}
          <RoutingBar routing={metrics.routing} />
          <LatencyDetail latency={metrics.latency} />
        </div>
      ) : null}

      {/* Auto-refresh indicator */}
      <p className="mt-8 text-center text-[11px] text-faint">
        Atualiza automaticamente a cada 30s
      </p>
    </div>
  );
}
