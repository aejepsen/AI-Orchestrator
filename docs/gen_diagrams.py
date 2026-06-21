"""Gera diagramas do AI-Orchestrator: 4 cards de agentes, mapa de microserviços e fluxo completo."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BG = "#0a0a0b"
PANEL = "#111114"
BORDER = "#26262c"
TXT = "#e8e8ec"
MUT = "#8a8a94"
GREEN = "#34d399"
BLUE = "#60a5fa"
AMBER = "#fbbf24"
RED = "#f87171"
VIOLET = "#a78bfa"
PINK = "#f472b6"

DOCS = "/home/aejepsen/Documentos/projeto-portifolio/AI-Orchestrator/docs"


def box(ax, x, y, w, h, fc=PANEL, ec=BORDER, lw=1.2, r=0.025):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=2)
    ax.add_patch(p)


def arrow(ax, x1, y1, x2, y2, color=MUT, lw=1.4, style="-|>"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                        color=color, lw=lw, zorder=3)
    ax.add_patch(a)


def base_ax(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=160)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ----------------------------------------------------------------------------
# 1) Cards dos agentes
# ----------------------------------------------------------------------------
AGENTS = {
    "financas": dict(
        accent=GREEN, port="8101", title="Agente Finanças",
        scope="Contas a pagar/receber · fluxo de caixa projetado · despesas · alçada de aprovação",
        tools=[
            "list_accounts / create_account — contas a pagar e receber",
            "settle_account — pay (pagar) / receive (receber)",
            "cashflow — projeção de caixa por período",
        ],
        rules=[
            "≤ R$ 5.000 → auto-aprovada",
            "≤ R$ 50.000 → exige aprovador 'gerente'",
            "> R$ 50.000 → exige aprovador 'diretor'",
            "pay só em conta 'pagar' aberta; receive só em 'receber'",
        ],
    ),
    "rh": dict(
        accent=BLUE, port="8102", title="Agente RH",
        scope="Funcionários · férias CLT · reembolsos · headcount",
        tools=[
            "list_employees / get_employee — cadastro e salários",
            "vacation_balance / request_vacation — férias por ano aquisitivo",
            "create_reimbursement — viagem, refeição, home office",
        ],
        rules=[
            "30 dias de férias após 12 meses de admissão",
            "fracionamento limitado a N períodos por ano aquisitivo",
            "reembolso: viagem ≤ R$ 3.000 · refeição ≤ R$ 100/dia",
            "home office ≤ R$ 500/mês",
        ],
    ),
    "estoque": dict(
        accent=AMBER, port="8103", title="Agente Estoque",
        scope="Produtos por SKU · saldo disponível · reservas · ponto de reposição",
        tools=[
            "list_products / get_product — catálogo e saldo por SKU",
            "create_reservation / release — reserva de unidades",
            "replenishment — sugestão de compra",
        ],
        rules=[
            "available = on_hand − reserved",
            "reserva nunca excede o disponível",
            "reposição quando available ≤ reorder_point",
            "sugestão de compra = reorder_point×2 − available",
        ],
    ),
    "vendas": dict(
        accent=VIOLET, port="8104", title="Agente Vendas",
        scope="Pedidos de venda · política de desconto por papel · comissão",
        tools=[
            "create_order — pedido com itens, papel e desconto",
            "list_orders / get_order — consulta de pedidos",
        ],
        rules=[
            "desconto ≤ 10% 'vendedor' · ≤ 20% 'gerente'",
            "nunca acima de 20% (hard limit)",
            "comissão: 2% até R$ 10.000 líquido · 3,5% acima",
        ],
    ),
}

COMMON = [
    "Mesmo LLM residente (Ollama) com system prompt escopado ao domínio",
    "Tools geradas do OpenAPI do microserviço (ToolRegistry, least-privilege)",
    "Loop de tool-calling: máx. 6 iterações + deadline; 422/404 reinjetado como observação",
    "Toda afirmação numérica vem do retorno de uma tool — nunca inventa valores",
    "Chamadas HTTP autenticadas com X-Internal-Key",
]

for name, a in AGENTS.items():
    fig, ax = base_ax((10, 7.2))
    ac = a["accent"]
    box(ax, 0.03, 0.03, 0.94, 0.94, fc=PANEL, ec=ac, lw=1.6)
    ax.plot([0.03, 0.97], [0.855, 0.855], color=BORDER, lw=1)
    ax.text(0.07, 0.925, a["title"], color=TXT, fontsize=19, fontweight="bold", va="center")
    ax.text(0.07, 0.878, a["scope"], color=MUT, fontsize=9.5, va="center")
    ax.text(0.93, 0.925, f"serviço :{a['port']}", color=ac, fontsize=10, ha="right",
            va="center", family="monospace")

    ax.text(0.07, 0.80, "FERRAMENTAS (OpenAPI → tools)", color=ac, fontsize=9,
            fontweight="bold")
    y = 0.755
    for t in a["tools"]:
        ax.text(0.085, y, "▸ " + t, color=TXT, fontsize=9.5)
        y -= 0.045

    ax.text(0.07, y - 0.02, "REGRAS DE NEGÓCIO (vivem na API, não no prompt)",
            color=ac, fontsize=9, fontweight="bold")
    y -= 0.065
    for r in a["rules"]:
        ax.text(0.085, y, "▸ " + r, color=TXT, fontsize=9.5)
        y -= 0.045

    ax.text(0.07, y - 0.02, "COMPORTAMENTO DO SUBAGENTE", color=MUT, fontsize=9,
            fontweight="bold")
    y -= 0.065
    for c in COMMON:
        ax.text(0.085, y, "· " + c, color=MUT, fontsize=8.8)
        y -= 0.038

    fig.savefig(f"{DOCS}/agent-{name}.png", facecolor=BG, bbox_inches="tight")
    plt.close(fig)

# ----------------------------------------------------------------------------
# 2) Mapa dos microserviços
# ----------------------------------------------------------------------------
fig, ax = base_ax((14, 8.5))
ax.text(0.5, 0.965, "AI-Orchestrator — Microserviços de Domínio",
        color=TXT, fontsize=17, fontweight="bold", ha="center")
ax.text(0.5, 0.925, "FastAPI + SQLite (volume próprio) · regra de negócio na API · "
        "OpenAPI consumido pelo gateway · auth X-Internal-Key",
        color=MUT, fontsize=9.5, ha="center")

MS = [
    ("financas :8101", GREEN, "Finanças",
     ["Contas a pagar e a receber", "Fluxo de caixa projetado",
      "Aprovação de despesa por alçada", "≤5k auto · ≤50k gerente", "· >50k diretor"]),
    ("rh :8102", BLUE, "RH",
     ["Cadastro funcionários / headcount", "Férias CLT por ano aquisitivo",
      "Reembolsos por categoria", "viagem 3k · refeição 100/dia", "· home office 500/mês"]),
    ("estoque :8103", AMBER, "Estoque",
     ["Catálogo de produtos por SKU", "Saldo on_hand / reservado /", "  disponível",
      "Reservas de unidades", "Ponto de reposição + sugestão"]),
    ("vendas :8104", VIOLET, "Vendas",
     ["Pedidos de venda multi-item", "Desconto por papel", "(10% / 20% máx.)",
      "Comissão 2% / 3,5% s/ líquido", "Catálogo de clientes"]),
]

gw_y = 0.72
box(ax, 0.30, gw_y, 0.40, 0.115, fc="#15151c", ec="#3a3a55", lw=1.6)
ax.text(0.5, gw_y + 0.078, "gateway :8100", color=TXT, fontsize=12.5,
        fontweight="bold", ha="center")
ax.text(0.5, gw_y + 0.036, "ToolRegistry: GET /openapi.json de cada serviço → tools do agente",
        color=MUT, fontsize=8.8, ha="center")

w, gap = 0.215, 0.028
x0 = (1 - 4 * w - 3 * gap) / 2
for i, (port, ac, title, lines) in enumerate(MS):
    x = x0 + i * (w + gap)
    y, h = 0.16, 0.42
    box(ax, x, y, w, h, fc=PANEL, ec=ac, lw=1.5)
    ax.text(x + w / 2, y + h - 0.045, title, color=TXT, fontsize=12,
            fontweight="bold", ha="center")
    ax.text(x + w / 2, y + h - 0.085, port, color=ac, fontsize=9, ha="center",
            family="monospace")
    yy = y + h - 0.14
    for ln in lines:
        cont = ln.startswith(("·", " ", "("))
        ax.text(x + 0.012 + (0.018 if cont else 0), yy,
                ln if cont else "▸ " + ln, color=TXT, fontsize=7.4)
        yy -= 0.05
    box(ax, x + 0.03, y - 0.075, w - 0.06, 0.05, fc="#0d0d12", ec=BORDER)
    ax.text(x + w / 2, y - 0.05, "SQLite WAL (volume)", color=MUT, fontsize=7.5,
            ha="center", family="monospace")
    arrow(ax, 0.5, gw_y, x + w / 2, y + h, color=ac, lw=1.3)
    ax.text(x + w / 2, y - 0.075 - 0.012, "", color=MUT, fontsize=7)

ax.text(0.5, 0.045, "HTTP interno (rede do compose) · header X-Internal-Key obrigatório · "
        "erros 422/404 estruturados com a regra violada → reinjetados no loop do agente",
        color=MUT, fontsize=9, ha="center")
fig.savefig(f"{DOCS}/microservices.png", facecolor=BG, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# 3) Fluxo completo
# ----------------------------------------------------------------------------
fig, ax = base_ax((13, 12))
ax.text(0.5, 0.978, "AI-Orchestrator — Fluxo Completo", color=TXT, fontsize=18,
        fontweight="bold", ha="center")
ax.text(0.5, 0.955, "suasalada.com.br · POST /chat → SSE (route · agent · final)",
        color=MUT, fontsize=10, ha="center")


def node(x, y, w, h, title, sub, ec=BORDER, fc=PANEL, ts=10.5, ss=7.6):
    box(ax, x, y, w, h, fc=fc, ec=ec, lw=1.5)
    ax.text(x + w / 2, y + h - 0.012, title, color=TXT, fontsize=ts,
            fontweight="bold", ha="center", va="top")
    ax.text(x + w / 2, y + h - 0.033, sub, color=MUT, fontsize=ss,
            ha="center", va="top", linespacing=1.4)


# ---- camada de entrada (horizontal, topo) ----
ty = 0.855
node(0.045, ty, 0.185, 0.062, "Browser", "React + SSE · Unlock\n(ACCESS_TOKEN)", ec=GREEN)
node(0.285, ty, 0.185, 0.062, "Cloudflare Tunnel", "cloudflared\nsuasalada.com.br", ec="#f59e0b")
node(0.525, ty, 0.215, 0.062, "Gateway FastAPI :8100", "auth Bearer · rate limit/h\nPOST /chat → SSE", ec="#3a3a55")
node(0.79, ty, 0.17, 0.062, "Ollama :11434", "chat: MODEL (qwen)\ninferência local GPU/CPU", ec="#7c3aed")
arrow(ax, 0.23, ty + 0.031, 0.285, ty + 0.031, color=MUT)
arrow(ax, 0.47, ty + 0.031, 0.525, ty + 0.031, color=MUT)
arrow(ax, 0.74, ty + 0.031, 0.79, ty + 0.031, color="#7c3aed")
ax.text(0.875, ty - 0.012, "usado por classify · agentes · synthesize",
        color=MUT, fontsize=7, ha="center", va="top")

# ---- fluxo vertical (LangGraph) ----
cx, nw, nh = 0.5, 0.26, 0.062
node(cx - nw / 2, 0.745, nw, nh, "sanitize", "BERT injection detector\nregex fallback · normaliza input", ec=GREEN)
# SBERT embedder (à direita do sanitize)
node(0.76, 0.745, 0.205, nh, "SBERT Embedder", "MiniLM-L12 (384d, CPU)\nembeddings locais", ec=PINK)
arrow(ax, 0.82, 0.745, 0.62, 0.697, color=PINK, lw=1.0)
ax.text(0.78, 0.738, "embed query", color=PINK, fontsize=7)
node(cx - nw / 2, 0.635, nw, nh, "classify", "semântico (Qdrant kNN) → LLM\n→ léxico · guards determinísticos", ec=GREEN)
# Qdrant — banco vetorial do semantic router (à esquerda do classify)
node(0.045, 0.635, 0.205, nh, "Qdrant :6333", "routing_examples (384d)\nAPI key auth · cosine kNN", ec="#22d3ee")
arrow(ax, 0.25, 0.666, cx - nw / 2, 0.666, color="#22d3ee")
ax.text(0.31, 0.676, "kNN ≥ 0.92 + consenso", color="#22d3ee", fontsize=7)
node(cx - nw / 2, 0.525, nw, nh, "dispatch", "fan-out paralelo por domínio\nThreadPoolExecutor", ec=BLUE)
arrow(ax, 0.6325, ty, cx, 0.807, color=MUT)
arrow(ax, cx, 0.745, cx, 0.697, color=GREEN)
arrow(ax, cx, 0.635, cx, 0.587, color=GREEN)
ax.text(cx + 0.012, 0.608, "domains ≥ 1", color=MUT, fontsize=7.5)
ax.text(cx - 0.145, 0.608, "SSE: route", color=GREEN, fontsize=7.5, ha="right")

# clarification (ramo à direita, isolado)
node(0.76, 0.635, 0.205, nh, "respond_clarification", "fora de escopo →\npede esclarecimento", ec=AMBER)
node(0.76, 0.525, 0.205, 0.052, "END", "SSE: final (clarification)", ec=RED)
arrow(ax, cx + nw / 2, 0.666, 0.76, 0.666, color=AMBER)
ax.text(0.69, 0.676, "clarification", color=AMBER, fontsize=7.5, ha="center")
arrow(ax, 0.8625, 0.635, 0.8625, 0.577, color=AMBER)

# ---- agentes (linha horizontal) + serviços abaixo ----
colors = [GREEN, BLUE, AMBER, VIOLET]
names = ["financas", "rh", "estoque", "vendas"]
ports = ["8101", "8102", "8103", "8104"]
aw, agap = 0.21, 0.035
ax0 = (1 - 4 * aw - 3 * agap) / 2
ay, sy = 0.36, 0.245
for i, (n, c, p) in enumerate(zip(names, colors, ports)):
    x = ax0 + i * (aw + agap)
    node(x, ay, aw, 0.068, f"agente {n}", "tool-loop ≤ 6 iters · deadline\nprompt + tools escopados", ec=c, ts=9.8, ss=7.2)
    node(x, sy, aw, 0.062, f"{n} :{p}", "FastAPI + SQLite\nregras de negócio na API", ec=c, ts=9.8, ss=7.2)
    arrow(ax, x + aw / 2, ay, x + aw / 2, sy + 0.062, color=c)
    ax.text(x + aw / 2 + 0.008, (ay + sy + 0.062) / 2, "X-Internal-Key",
            color=MUT, fontsize=6.2, va="center")
    arrow(ax, cx, 0.525, x + aw / 2, ay + 0.068, color=c, lw=1.2)

ax.text(0.5, 0.452, "SSE: agent (um evento por domínio concluído)",
        color=BLUE, fontsize=7.5, ha="center")

# ---- synthesize + END ----
node(cx - nw / 2, 0.125, nw, nh, "synthesize", "1 domínio: passthrough\nN domínios: síntese via LLM", ec=GREEN)
node(cx - nw / 2, 0.033, nw, 0.052, "END", "SSE: final — resposta + traces", ec=RED)
for i in range(4):
    x = ax0 + i * (aw + agap) + aw / 2
    arrow(ax, x, sy, cx + (i - 1.5) * 0.045, 0.187, color=colors[i], lw=1.0)
ax.text(cx + 0.155, 0.21, "answers", color=MUT, fontsize=7.5)
arrow(ax, cx, 0.125, cx, 0.085, color=GREEN)

ax.text(0.5, 0.005, "Segurança: BERT injection detector · SBERT embeddings (384d) · Qdrant API key auth · "
        "tools least-privilege · X-Internal-Key · rate limit · ACCESS_TOKEN · SQLite WAL",
        color=MUT, fontsize=8.5, ha="center")

fig.savefig(f"{DOCS}/architecture-flow.png", facecolor=BG, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# 4) LangGraph StateGraph (grafo do gateway)
# ----------------------------------------------------------------------------
fig, ax = base_ax((13, 8.5))
ax.text(0.5, 0.965, "AI-Orchestrator — LangGraph StateGraph (gateway/graph.py)",
        color=TXT, fontsize=17, fontweight="bold", ha="center")
ax.text(0.5, 0.93, "POST /chat  →  SSE: route · agent · final", color=MUT,
        fontsize=10.5, ha="center")


def gnode(x, y, w, h, title, lines, ec=BORDER, tc=TXT):
    box(ax, x, y, w, h, fc=PANEL, ec=ec, lw=1.5)
    ax.text(x + w / 2, y + h - 0.018, title, color=tc, fontsize=12,
            fontweight="bold", ha="center", va="top")
    ax.text(x + w / 2, y + h - 0.062, "\n".join(lines), color=MUT, fontsize=8.6,
            ha="center", va="top", linespacing=1.6)


ax.scatter([0.06], [0.79], s=420, color=GREEN, zorder=4)
ax.text(0.06, 0.745, "START", color=GREEN, fontsize=10, fontweight="bold", ha="center")

gnode(0.10, 0.73, 0.135, 0.12, "sanitize",
      ["BERT + regex", "anti-injection"], ec=BORDER)
gnode(0.245, 0.73, 0.125, 0.12, "enrich",
      ["Camada A", "ctx + KG", "(opt-in)"], ec=GREEN, tc=GREEN)
gnode(0.38, 0.70, 0.23, 0.15, "classify",
      ["semântico (Qdrant kNN ≥ 0.92)", "→ LLM → léxico · guards", "→ RoutePlan {domains, plan}"], ec=BORDER)
ax.text(0.495, 0.675, "emite evento SSE “route”", color=MUT, fontsize=8.6,
        ha="center", style="italic")

arrow(ax, 0.08, 0.79, 0.10, 0.79, color=GREEN)
arrow(ax, 0.235, 0.79, 0.245, 0.79, color=GREEN)
arrow(ax, 0.37, 0.79, 0.38, 0.79, color=MUT)

# SBERT embedder (coluna esquerda, abaixo de sanitize com gap claro)
gnode(0.02, 0.57, 0.22, 0.095, "SBERT Embedder",
      ["MiniLM-L12 (384d)", "CPU · local"], ec=PINK, tc=PINK)
arrow(ax, 0.24, 0.635, 0.40, 0.73, color=PINK, lw=1.0)
ax.text(0.34, 0.70, "embed", color=PINK, fontsize=8, style="italic")
# Qdrant — banco vetorial (abaixo de SBERT)
gnode(0.02, 0.43, 0.22, 0.105, "Qdrant :6333",
      ["routing_examples (384d)", "cosine · API key auth"], ec="#22d3ee", tc="#22d3ee")
arrow(ax, 0.24, 0.50, 0.40, 0.70, color="#22d3ee")
ax.text(0.34, 0.60, "kNN + consenso", color="#22d3ee", fontsize=8, style="italic")

gnode(0.69, 0.745, 0.27, 0.14, "respond_clarification",
      ["pergunta ambígua →", "devolve pedido de esclarecimento", "(final_answer direto)"], ec=AMBER, tc=AMBER)
arrow(ax, 0.61, 0.81, 0.69, 0.825, color=AMBER)
ax.text(0.645, 0.845, "sim\nclarification?", color=AMBER, fontsize=8.6,
        ha="center", style="italic")
ax.scatter([0.95], [0.69], s=420, color=RED, zorder=4)
ax.text(0.95, 0.645, "END", color=RED, fontsize=10, fontweight="bold", ha="center")
arrow(ax, 0.88, 0.745, 0.945, 0.70, color=AMBER)

gnode(0.38, 0.44, 0.23, 0.145, "dispatch",
      ["fan-out paralelo", "ThreadPoolExecutor", "(max_workers = nº domínios)"], ec=BORDER)
ax.text(0.495, 0.415, "emite “agent” por domínio", color=MUT, fontsize=8.6,
        ha="center", style="italic")
arrow(ax, 0.495, 0.665, 0.495, 0.585, color=MUT)
ax.text(0.51, 0.635, "não → dispatch", color=MUT, fontsize=8.8, style="italic")

agents = [("agente finanças", "8101"), ("agente rh", "8102"),
          ("agente estoque", "8103"), ("agente vendas", "8104")]
for i, (nm, p) in enumerate(agents):
    y = 0.50 - i * 0.125
    gnode(0.69, y, 0.27, 0.105, nm,
          [f"tool-loop LLM ↔ microserviço :{p}", "(OpenAPI via registry, X-Internal-Key)"],
          ec="#3b5bb5", tc=BLUE)
    arrow(ax, 0.61, 0.51, 0.69, y + 0.052, color=BLUE, lw=1.1)
    arrow(ax, 0.69, y + 0.03, 0.55, 0.225, color=BLUE, lw=0.8, style="-|>")

# Neo4j — Camada B (Knowledge Graph) como tool virtual dos agentes
gnode(0.02, 0.21, 0.26, 0.12, "Neo4j :7687",
      ["Camada B — Knowledge Graph", "tool virtual expand_context (1-hop)"], ec=AMBER, tc=AMBER)
arrow(ax, 0.28, 0.30, 0.69, 0.305, color=AMBER, lw=1.0)
ax.text(0.47, 0.315, "expand_context (least-privilege)", color=AMBER, fontsize=7.8, style="italic")
arrow(ax, 0.13, 0.33, 0.30, 0.73, color=AMBER, lw=0.8, style="-|>")
ax.text(0.165, 0.52, "KG→enrich", color=AMBER, fontsize=7.2, style="italic", rotation=68)

gnode(0.38, 0.13, 0.23, 0.115, "synthesize",
      ["LLM consolida respostas", "dos agentes → final_answer"], ec=BORDER)
ax.text(0.62, 0.085, "emite evento SSE “final”", color=MUT, fontsize=8.6,
        ha="left", style="italic")
ax.scatter([0.495], [0.045], s=420, color=RED, zorder=4)
ax.text(0.535, 0.04, "END", color=RED, fontsize=10, fontweight="bold", ha="left")
arrow(ax, 0.495, 0.105, 0.495, 0.062, color=MUT)

fig.savefig(f"{DOCS}/langgraph-flow.png", facecolor=BG, bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------------------
# 5) Semiose — pipeline contextual (Camadas A / B / C + S1/S3/S6)
# ----------------------------------------------------------------------------
fig, ax = base_ax((14, 8.6))
ax.text(0.5, 0.965, "AI-Orchestrator — Semiose: contexto antes do significado",
        color=TXT, fontsize=17, fontweight="bold", ha="center")
ax.text(0.5, 0.928, "Signo → Objeto → Interpretante (Peirce) mapeado no grafo · "
        "cada camada é opt-in e degrada graceful (Harness antes de Model)",
        color=MUT, fontsize=9.5, ha="center")


def snode(x, y, w, h, title, lines, ec=BORDER, tc=TXT, ts=12, ss=8.4):
    box(ax, x, y, w, h, fc=PANEL, ec=ec, lw=1.6)
    ax.text(x + w / 2, y + h - 0.02, title, color=tc, fontsize=ts,
            fontweight="bold", ha="center", va="top")
    ax.text(x + w / 2, y + h - 0.075, "\n".join(lines), color=MUT, fontsize=ss,
            ha="center", va="top", linespacing=1.6)


# fluxo base
y0 = 0.70
ax.scatter([0.045], [y0 + 0.075], s=360, color=GREEN, zorder=4)
ax.text(0.045, y0 + 0.015, "query", color=GREEN, fontsize=9, ha="center", fontweight="bold")
snode(0.10, y0, 0.175, 0.155, "sanitize",
      ["anti-injection", "(BERT + regex)"], ec=BORDER)
snode(0.31, y0, 0.205, 0.155, "enrich — Camada A",
      ["signo contextualizado", "regex/spaCy + _last_route", "+ vizinhos do KG (opt-in)"], ec=GREEN, tc=GREEN)
snode(0.565, y0, 0.205, 0.155, "classify — Camada C",
      ["Qdrant kNN + consenso", "boost contextual +0,05", "cross-encoder desempata (S3)"], ec=VIOLET, tc=VIOLET)
snode(0.80, y0, 0.165, 0.155, "dispatch",
      ["fan-out por domínio", "agentes + KG tool"], ec=BLUE, tc=BLUE)
arrow(ax, 0.07, y0 + 0.077, 0.10, y0 + 0.077, color=GREEN)
arrow(ax, 0.275, y0 + 0.077, 0.31, y0 + 0.077, color=MUT)
arrow(ax, 0.515, y0 + 0.077, 0.565, y0 + 0.077, color=MUT)
arrow(ax, 0.77, y0 + 0.077, 0.80, y0 + 0.077, color=MUT)

# Camada B — Knowledge Graph (Neo4j) alimentando enrich e agentes
snode(0.305, 0.40, 0.46, 0.20, "Camada B — Knowledge Graph (Neo4j)",
      ["tool virtual expand_context · travessia 1-hop dirigida por domínio",
       "entidades: produto · fornecedor · funcionário · cliente · despesa · cargo",
       "relações: EMITE · REQUER_APROVACAO (alçada) · ABASTECE · COMPROU · VENDEU_PARA",
       "realimenta o enrich (KG_ENRICH_ENABLED) e o loop dos agentes (least-privilege)"],
      ec=AMBER, tc=AMBER, ts=12.5, ss=8.6)
arrow(ax, 0.40, 0.60, 0.40, y0, color=AMBER, lw=1.3)
ax.text(0.345, 0.66, "KG→enrich", color=AMBER, fontsize=7.8, ha="center", style="italic")
arrow(ax, 0.66, 0.60, 0.84, y0, color=AMBER, lw=1.3)
ax.text(0.78, 0.66, "expand_context", color=AMBER, fontsize=7.8, ha="center", style="italic")

# coluna direita: índice contextual (S1) + embedder
snode(0.80, 0.43, 0.165, 0.14, "Índice contextual",
      ["SBERT MiniLM-L12", "(384d, CPU)", "S1: prefixa domínio", "antes de embedar"],
      ec=PINK, tc=PINK, ts=11, ss=8)
arrow(ax, 0.8825, 0.57, 0.70, y0, color=PINK, lw=1.0)

# faixa inferior: resultados medidos
ax.text(0.5, 0.315, "Resultados medidos (eval)", color=TXT, fontsize=11,
        fontweight="bold", ha="center")
cards = [
    (GREEN, "Roteamento multi-domínio", "88,9% → 93,7%", "decomposição conceito→domínio\n(gate ≥90% PASS)"),
    (AMBER, "Knowledge Graph", "0 órfãos · 0,929", "fornecedores conectados\nRelation Validity@5"),
    (GREEN, "Camada A — Enricher", "F1 0,973 · FER 0,026", "Entity Propagation\n150 casos, 4/4 gates"),
    (VIOLET, "Eval Semiose (S6)", "12 métricas", "BERTScore + Routing\nFailure Rate"),
]
cw, cgap = 0.215, 0.025
cx0 = (1 - 4 * cw - 3 * cgap) / 2
for i, (ac, ttl, big, sub) in enumerate(cards):
    x = cx0 + i * (cw + cgap)
    box(ax, x, 0.045, cw, 0.235, fc=PANEL, ec=ac, lw=1.5)
    ax.text(x + cw / 2, 0.255, ttl, color=MUT, fontsize=8.8, ha="center", fontweight="bold")
    ax.text(x + cw / 2, 0.185, big, color=ac, fontsize=15.5, ha="center", fontweight="bold")
    ax.text(x + cw / 2, 0.115, sub, color=MUT, fontsize=8, ha="center", linespacing=1.5)

fig.savefig(f"{DOCS}/semiose-flow.png", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("ok: 8 imagens geradas em docs/")
