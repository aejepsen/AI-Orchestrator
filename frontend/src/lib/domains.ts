/**
 * Identidade visual por domínio. Classes completas (não interpoladas)
 * para que o Tailwind v4 as detecte estaticamente no build.
 */

export interface DomainMeta {
  label: string;
  chip: string;
  border: string;
  dot: string;
  text: string;
}

export const DOMAIN_META: Record<string, DomainMeta> = {
  financas: {
    label: "Finanças",
    chip: "border-emerald-400/20 bg-emerald-400/10 text-emerald-300",
    border: "border-l-emerald-400/60",
    dot: "bg-emerald-400",
    text: "text-emerald-300",
  },
  rh: {
    label: "RH",
    chip: "border-violet-400/20 bg-violet-400/10 text-violet-300",
    border: "border-l-violet-400/60",
    dot: "bg-violet-400",
    text: "text-violet-300",
  },
  estoque: {
    label: "Estoque",
    chip: "border-amber-400/20 bg-amber-400/10 text-amber-300",
    border: "border-l-amber-400/60",
    dot: "bg-amber-400",
    text: "text-amber-300",
  },
  vendas: {
    label: "Vendas",
    chip: "border-sky-400/20 bg-sky-400/10 text-sky-300",
    border: "border-l-sky-400/60",
    dot: "bg-sky-400",
    text: "text-sky-300",
  },
};

const FALLBACK: DomainMeta = {
  label: "Domínio",
  chip: "border-line-strong bg-raised text-muted",
  border: "border-l-line-strong",
  dot: "bg-muted",
  text: "text-muted",
};

export function domainMeta(domain: string): DomainMeta {
  return DOMAIN_META[domain] ?? { ...FALLBACK, label: domain };
}
