import { useCallback, useEffect, useRef, useState } from "react";

/* ───────── Types ───────── */

interface DomainBreakdown {
  total: number;
  correct: number;
  accuracy: number;
}

interface RoutingRun {
  timestamp: string;
  file: string;
  model: string;
  total: number;
  hits: number;
  accuracy: number;
  passed: boolean;
  gate: number;
  domain_breakdown: Record<string, DomainBreakdown>;
  failed_queries: { question: string; expected: string[]; got: string[] }[];
  layer_counts: Record<string, number>;
}

interface ConfusionMatrix {
  tp: number;
  fp: number;
  fn: number;
  tn: number;
}

interface InjectionRun {
  timestamp: string;
  file: string;
  total: number;
  blocked: number;
  leaks: number;
  gate_pass: boolean;
  precision: number;
  recall: number;
  f1: number;
  confusion: ConfusionMatrix;
  leaked_cases: { id: string; question: string; bad_route: string[] }[];
  avg_elapsed_s: number;
}

interface ModelComparison {
  model: string;
  avg_routing_accuracy?: number;
  last_routing_accuracy?: number;
  routing_runs?: number;
  avg_injection_f1?: number;
  injection_runs?: number;
}

interface EvalData {
  available: boolean;
  stale?: boolean;
  routing: {
    runs: RoutingRun[];
    avg_accuracy: number;
    domain_breakdown: Record<string, DomainBreakdown>;
  };
  injection: {
    runs: InjectionRun[];
    avg_f1: number;
    total_blocked: number;
    total_leaked: number;
  };
  models: ModelComparison[];
  total_runs: number;
}

const TOKEN_KEY = "aio:access-token";
const REFRESH_MS = 60_000;

/* ───────── Helpers ───────── */

function pctStr(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function scoreColor(v: number): string {
  if (v >= 0.95) return "#34d399";
  if (v >= 0.85) return "#fbbf24";
  return "#f87171";
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

function EvalSkeleton() {
  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {Array.from({ length: 6 }).map((_, i) => (
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

/* ───────── Domain Bar Chart ───────── */

function DomainChart({ domains }: { domains: Record<string, DomainBreakdown> }) {
  const entries = Object.entries(domains);
  if (entries.length === 0) return null;

  const domainColors: Record<string, string> = {
    financas: "#34d399",
    rh: "#818cf8",
    estoque: "#f59e0b",
    vendas: "#60a5fa",
  };

  return (
    <div className="col-span-full rounded-2xl border border-line bg-surface/40 p-5 backdrop-blur-sm">
      <p className="font-mono text-[11px] tracking-widest text-faint uppercase">
        Accuracy por Dominio (Agregado)
      </p>
      <div className="mt-4 space-y-3">
        {entries.map(([domain, stats]) => {
          const color = domainColors[domain] || "#64748b";
          return (
            <div key={domain} className="flex items-center gap-3">
              <span className="w-20 text-right font-mono text-xs text-muted capitalize">
                {domain}
              </span>
              <div className="flex-1">
                <div className="h-6 w-full rounded-md bg-raised">
                  <div
                    className="flex h-full items-center rounded-md px-2 text-[10px] font-semibold transition-all duration-700 ease-out"
                    style={{
                      width: `${Math.max(stats.accuracy * 100, 6)}%`,
                      backgroundColor: color,
                      color: "#0f172a",
                    }}
                  >
                    {pctStr(stats.accuracy)}
                  </div>
                </div>
              </div>
              <span className="w-16 text-right font-mono text-[11px] text-muted">
                {stats.correct}/{stats.total}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ───────── Confusion Matrix ───────── */

function ConfusionMatrixCard({ matrix }: { matrix: ConfusionMatrix }) {
  const cells = [
    { label: "TP", value: matrix.tp, color: "rgba(52, 211, 153, 0.25)", border: "border-emerald-400/30", text: "text-emerald-300" },
    { label: "FP", value: matrix.fp, color: "rgba(251, 191, 36, 0.15)", border: "border-amber-400/30", text: "text-amber-300" },
    { label: "FN", value: matrix.fn, color: "rgba(248, 113, 113, 0.2)", border: "border-red-400/30", text: "text-red-300" },
    { label: "TN", value: matrix.tn, color: "rgba(100, 116, 139, 0.15)", border: "border-line", text: "text-muted" },
  ];

  return (
    <div className="rounded-2xl border border-line bg-surface/40 p-5 backdrop-blur-sm">
      <p className="font-mono text-[11px] tracking-widest text-faint uppercase">
        Confusion Matrix (Injection)
      </p>
      <div className="mt-4 grid grid-cols-2 gap-2">
        {cells.map((cell) => (
          <div
            key={cell.label}
            className={`flex flex-col items-center justify-center rounded-xl border ${cell.border} p-4 transition-all duration-300`}
            style={{ backgroundColor: cell.color }}
          >
            <span className="font-mono text-[10px] tracking-wider text-faint uppercase">
              {cell.label}
            </span>
            <span className={`mt-1 text-2xl font-bold ${cell.text}`}>{cell.value}</span>
          </div>
        ))}
      </div>
      <div className="mt-3 grid grid-cols-2 gap-2 text-center">
        <p className="text-[10px] text-faint">Predicted Positive</p>
        <p className="text-[10px] text-faint">Predicted Negative</p>
      </div>
    </div>
  );
}

/* ───────── Runs Table ───────── */

function RunsTable({
  routingRuns,
  injectionRuns,
}: {
  routingRuns: RoutingRun[];
  injectionRuns: InjectionRun[];
}) {
  type Row = { timestamp: string; type: string; score: number; scoreLabel: string; model: string; passed: boolean };

  const rows: Row[] = [
    ...routingRuns.map((r) => ({
      timestamp: r.timestamp,
      type: "Routing",
      score: r.accuracy,
      scoreLabel: pctStr(r.accuracy),
      model: r.model,
      passed: r.passed,
    })),
    ...injectionRuns.map((r) => ({
      timestamp: r.timestamp,
      type: "Injection",
      score: r.f1,
      scoreLabel: pctStr(r.f1),
      model: "-",
      passed: r.gate_pass,
    })),
  ].sort((a, b) => b.timestamp.localeCompare(a.timestamp));

  if (rows.length === 0) return null;

  return (
    <div className="col-span-full rounded-2xl border border-line bg-surface/40 p-5 backdrop-blur-sm">
      <p className="font-mono text-[11px] tracking-widest text-faint uppercase mb-4">
        Historico de Runs
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-line text-left">
              <th className="pb-2 pr-4 font-mono text-[11px] text-faint uppercase">Data</th>
              <th className="pb-2 pr-4 font-mono text-[11px] text-faint uppercase">Tipo</th>
              <th className="pb-2 pr-4 font-mono text-[11px] text-faint uppercase">Score</th>
              <th className="pb-2 pr-4 font-mono text-[11px] text-faint uppercase">Modelo</th>
              <th className="pb-2 font-mono text-[11px] text-faint uppercase">Status</th>
            </tr>
          </thead>
          <tbody>
            {rows.slice(0, 15).map((row, i) => (
              <tr key={i} className="border-b border-line/50 transition-colors hover:bg-surface/30">
                <td className="py-2.5 pr-4 font-mono text-xs text-muted">{row.timestamp}</td>
                <td className="py-2.5 pr-4">
                  <span
                    className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${
                      row.type === "Routing"
                        ? "bg-emerald-400/10 text-emerald-400"
                        : "bg-amber-400/10 text-amber-400"
                    }`}
                  >
                    {row.type}
                  </span>
                </td>
                <td className="py-2.5 pr-4">
                  <span className="font-semibold" style={{ color: scoreColor(row.score) }}>
                    {row.scoreLabel}
                  </span>
                </td>
                <td className="py-2.5 pr-4 font-mono text-xs text-muted max-w-[180px] truncate">
                  {row.model}
                </td>
                <td className="py-2.5">
                  <span
                    className={`inline-block h-2 w-2 rounded-full ${
                      row.passed ? "bg-emerald-400" : "bg-red-400"
                    }`}
                  />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ───────── Model Comparison ───────── */

function ModelComparisonCard({ models }: { models: ModelComparison[] }) {
  if (models.length < 1) return null;

  return (
    <div className="col-span-full rounded-2xl border border-line bg-surface/40 p-5 backdrop-blur-sm">
      <p className="font-mono text-[11px] tracking-widest text-faint uppercase mb-4">
        Comparacao de Modelos
      </p>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {models.map((m) => (
          <div
            key={m.model}
            className="rounded-xl border border-line/60 bg-raised/30 p-4 transition-all hover:border-line-strong"
          >
            <p className="font-mono text-xs text-ink truncate mb-3" title={m.model}>
              {m.model}
            </p>
            <div className="space-y-2">
              {m.last_routing_accuracy !== undefined && (
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-muted">Routing (ultimo run)</span>
                  <span
                    className="font-semibold text-sm"
                    style={{ color: scoreColor(m.last_routing_accuracy) }}
                  >
                    {pctStr(m.last_routing_accuracy)}
                  </span>
                </div>
              )}
              {m.avg_routing_accuracy !== undefined && (
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-muted">Routing (media {m.routing_runs ?? 0} runs)</span>
                  <span
                    className="text-xs"
                    style={{ color: scoreColor(m.avg_routing_accuracy) }}
                  >
                    {pctStr(m.avg_routing_accuracy)}
                  </span>
                </div>
              )}
              {m.avg_injection_f1 !== undefined && (
                <div className="flex items-center justify-between">
                  <span className="text-[11px] text-muted">Injection F1</span>
                  <span
                    className="font-semibold text-sm"
                    style={{ color: scoreColor(m.avg_injection_f1) }}
                  >
                    {pctStr(m.avg_injection_f1)}
                  </span>
                </div>
              )}
              <div className="flex items-center justify-between">
                <span className="text-[11px] text-muted">Runs</span>
                <span className="text-xs text-muted">
                  {(m.routing_runs || 0) + (m.injection_runs || 0)}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ───────── Main Component ───────── */

export function EvalDashboard() {
  const [data, setData] = useState<EvalData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchData = useCallback(async () => {
    try {
      const headers: Record<string, string> = {};
      const token = localStorage.getItem(TOKEN_KEY);
      if (token) headers["X-Access-Token"] = token;

      const res = await fetch("/eval-results", { headers });
      if (!res.ok) {
        if (res.status === 401) {
          setError("Nao autorizado. Verifique o token de acesso.");
          return;
        }
        throw new Error(`HTTP ${res.status}`);
      }
      const json: EvalData = await res.json();
      setData(json);
      setError(null);
    } catch {
      setError("Falha ao conectar com o gateway.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    intervalRef.current = setInterval(fetchData, REFRESH_MS);
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [fetchData]);

  // Latest injection confusion matrix (aggregate across latest run)
  const latestInjection = data?.injection?.runs?.[0];
  const aggConfusion: ConfusionMatrix = data?.injection?.runs
    ? data.injection.runs.reduce(
        (acc, r) => ({
          tp: acc.tp + r.confusion.tp,
          fp: acc.fp + r.confusion.fp,
          fn: acc.fn + r.confusion.fn,
          tn: acc.tn + r.confusion.tn,
        }),
        { tp: 0, fp: 0, fn: 0, tn: 0 },
      )
    : { tp: 0, fp: 0, fn: 0, tn: 0 };

  return (
    <div className="mx-auto w-full max-w-5xl px-4 py-8 sm:px-6">
      {/* Header */}
      <div className="mb-8 flex items-baseline justify-between">
        <div>
          <h2 className="text-lg font-semibold tracking-tight text-ink">Evaluation Metrics</h2>
          <p className="mt-1 text-xs text-muted">Routing accuracy, injection defense, model comparison</p>
        </div>
        <div className="flex items-center gap-3">
          {data && !data.available && (
            <span className="rounded-full bg-amber-400/10 px-2.5 py-1 text-[11px] font-medium text-amber-400">
              Sem dados
            </span>
          )}
          {data?.stale && (
            <span className="rounded-full bg-amber-400/10 px-2.5 py-1 text-[11px] font-medium text-amber-300">
              Cache
            </span>
          )}
          {data?.available && !data.stale && (
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
        <EvalSkeleton />
      ) : data?.available ? (
        <div className="animate-fade-up grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {/* Hero Cards */}
          <StatCard
            label="Routing Accuracy"
            value={pctStr(data.routing.avg_accuracy)}
            sub={`${data.routing.runs.length} runs`}
            accent="rgba(52, 211, 153, 0.1)"
          />
          <StatCard
            label="Injection F1"
            value={pctStr(data.injection.avg_f1)}
            sub={`${data.injection.total_blocked} bloqueados, ${data.injection.total_leaked} vazamentos`}
            accent={data.injection.total_leaked > 0 ? "rgba(248, 113, 113, 0.1)" : "rgba(52, 211, 153, 0.1)"}
          />
          <StatCard
            label="Total Runs"
            value={String(data.total_runs)}
            sub={`${data.routing.runs.length} routing + ${data.injection.runs.length} injection`}
            accent="rgba(129, 140, 248, 0.1)"
          />

          {/* Latest run highlights */}
          {data.routing.runs[0] && (
            <StatCard
              label="Ultimo Routing"
              value={pctStr(data.routing.runs[0].accuracy)}
              sub={`${data.routing.runs[0].hits}/${data.routing.runs[0].total} — ${data.routing.runs[0].timestamp}`}
              accent="rgba(96, 165, 250, 0.1)"
            />
          )}
          {latestInjection && (
            <StatCard
              label="Ultimo Injection"
              value={latestInjection.gate_pass ? "PASS" : "FAIL"}
              sub={`F1: ${pctStr(latestInjection.f1)} — ${latestInjection.timestamp}`}
              accent={latestInjection.gate_pass ? "rgba(52, 211, 153, 0.1)" : "rgba(248, 113, 113, 0.1)"}
            />
          )}
          {latestInjection && (
            <StatCard
              label="Tempo Medio (Injection)"
              value={`${latestInjection.avg_elapsed_s.toFixed(1)}s`}
              sub={`${latestInjection.total} tentativas`}
              accent="rgba(251, 191, 36, 0.1)"
            />
          )}

          {/* Domain Breakdown */}
          <DomainChart domains={data.routing.domain_breakdown} />

          {/* Confusion Matrix */}
          {(aggConfusion.tp > 0 || aggConfusion.fn > 0) && (
            <ConfusionMatrixCard matrix={aggConfusion} />
          )}

          {/* Model Comparison */}
          <ModelComparisonCard models={data.models} />

          {/* Runs Table */}
          <RunsTable routingRuns={data.routing.runs} injectionRuns={data.injection.runs} />
        </div>
      ) : (
        <div className="mt-16 text-center">
          <p className="text-sm text-muted">Nenhum resultado de avaliacao encontrado em evals/results/</p>
        </div>
      )}

      <p className="mt-8 text-center text-[11px] text-faint">Atualiza automaticamente a cada 60s</p>
    </div>
  );
}
