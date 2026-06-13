interface ConfirmCardProps {
  domains: string[];
  plan: string;
  onApprove: () => void;
  onReject: () => void;
  disabled?: boolean;
}

export function ConfirmCard({ domains, plan, onApprove, onReject, disabled }: ConfirmCardProps) {
  return (
    <div className="animate-fade-up rounded-lg border border-amber-400/30 bg-amber-400/5 px-4 py-4">
      <p className="font-mono text-[11px] tracking-wide text-amber-300 uppercase">
        confirmação necessária
      </p>
      <p className="mt-2 text-sm leading-relaxed text-ink/90">
        {plan || `Executar agentes: ${domains.join(", ")}`}
      </p>
      <div className="mt-3 flex gap-3">
        <button
          onClick={onApprove}
          disabled={disabled}
          className="rounded-lg bg-emerald-500/20 border border-emerald-400/30 px-4 py-1.5 text-xs font-medium text-emerald-300 transition-colors hover:bg-emerald-500/30 disabled:opacity-40"
        >
          Aprovar
        </button>
        <button
          onClick={onReject}
          disabled={disabled}
          className="rounded-lg bg-red-500/10 border border-red-400/20 px-4 py-1.5 text-xs font-medium text-red-300 transition-colors hover:bg-red-500/20 disabled:opacity-40"
        >
          Rejeitar
        </button>
      </div>
    </div>
  );
}
