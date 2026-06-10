import { domainMeta } from "../lib/domains";
import type { RoutePlan } from "../lib/sse";

export function RouteChips({ route }: { route: RoutePlan }) {
  return (
    <div className="animate-fade-up space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <span className="font-mono text-[11px] tracking-wide text-faint uppercase">rota</span>
        {route.domains.map((domain) => {
          const meta = domainMeta(domain);
          return (
            <span
              key={domain}
              className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-xs font-medium ${meta.chip}`}
            >
              <span className={`size-1.5 rounded-full ${meta.dot}`} />
              {meta.label}
            </span>
          );
        })}
      </div>
      {route.plan && <p className="text-xs leading-relaxed text-muted">{route.plan}</p>}
    </div>
  );
}
