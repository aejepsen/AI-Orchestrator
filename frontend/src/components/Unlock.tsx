import { useState, type FormEvent } from "react";

export function Unlock({ error, onUnlock }: { error: string | null; onUnlock: (token: string) => void }) {
  const [token, setToken] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    onUnlock(token.trim());
  }

  return (
    <main className="flex min-h-dvh items-center justify-center px-6">
      <form onSubmit={submit} className="animate-fade-up w-full max-w-sm space-y-6">
        <header className="space-y-1.5">
          <h1 className="text-2xl font-semibold tracking-[-0.03em]">AI-Orchestrator</h1>
          <p className="text-sm leading-relaxed text-muted">
            Gateway multi-agente on-premise. Informe o token de acesso para continuar.
          </p>
        </header>
        <div className="space-y-3">
          <input
            type="password"
            value={token}
            onChange={(e) => setToken(e.target.value)}
            placeholder="Token de acesso"
            autoFocus
            aria-label="Token de acesso"
            className="w-full rounded-xl border border-line bg-surface px-4 py-3 text-sm text-ink placeholder:text-faint focus:border-line-strong focus:ring-1 focus:ring-white/10 focus:outline-none"
          />
          {error && <p className="text-xs text-red-400">{error}</p>}
          <button
            type="submit"
            className="w-full rounded-xl bg-accent py-3 text-sm font-medium text-bg transition-opacity hover:opacity-90"
          >
            Entrar
          </button>
          <p className="text-center text-[11px] text-faint">Sem token configurado no servidor, o acesso é aberto.</p>
        </div>
      </form>
    </main>
  );
}
