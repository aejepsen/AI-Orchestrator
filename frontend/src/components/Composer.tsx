import { useState, type FormEvent } from "react";

export function Composer({ disabled, onSend }: { disabled: boolean; onSend: (question: string) => void }) {
  const [value, setValue] = useState("");

  function submit(event: FormEvent) {
    event.preventDefault();
    const question = value.trim();
    if (!question || disabled) return;
    setValue("");
    onSend(question);
  }

  return (
    <form onSubmit={submit} className="flex items-end gap-2">
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit(e);
          }
        }}
        rows={1}
        maxLength={4000}
        disabled={disabled}
        placeholder={disabled ? "Aguardando resposta do gateway…" : "Pergunte sobre finanças, RH, estoque ou vendas…"}
        aria-label="Pergunta"
        className="max-h-40 min-h-[2.75rem] flex-1 resize-none rounded-xl border border-line bg-surface px-4 py-3 text-sm text-ink placeholder:text-faint focus:border-line-strong focus:outline-none disabled:opacity-50"
      />
      <button
        type="submit"
        disabled={disabled || !value.trim()}
        className="h-11 shrink-0 rounded-xl bg-accent px-4 text-sm font-medium text-bg transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-30"
      >
        Enviar
      </button>
    </form>
  );
}
