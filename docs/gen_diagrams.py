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
    ax.text(x + w / 2, y - 0.05, "SQLite (volume)", color=MUT, fontsize=7.5,
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
node(0.79, ty, 0.17, 0.062, "Ollama :11434", "MODEL (qwen)\nkeep_alive=-1 · serializa", ec="#7c3aed")
arrow(ax, 0.23, ty + 0.031, 0.285, ty + 0.031, color=MUT)
arrow(ax, 0.47, ty + 0.031, 0.525, ty + 0.031, color=MUT)
arrow(ax, 0.74, ty + 0.031, 0.79, ty + 0.031, color="#7c3aed")
ax.text(0.875, ty - 0.012, "usado por classify · agentes · synthesize",
        color=MUT, fontsize=7, ha="center", va="top")

# ---- fluxo vertical (LangGraph) ----
cx, nw, nh = 0.5, 0.26, 0.062
node(cx - nw / 2, 0.745, nw, nh, "sanitize", "normaliza e neutraliza injeção\nde markup/controle", ec=GREEN)
node(cx - nw / 2, 0.635, nw, nh, "classify", "LLM roteador → RoutePlan\nretry → fallback léxico", ec=GREEN)
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

ax.text(0.5, 0.005, "Segurança: sanitize anti-injeção · classifier ignora comandos injetados · "
        "tools least-privilege por domínio · X-Internal-Key · rate limit · ACCESS_TOKEN público",
        color=MUT, fontsize=8.5, ha="center")

fig.savefig(f"{DOCS}/architecture-flow.png", facecolor=BG, bbox_inches="tight")
plt.close(fig)
print("ok: 6 imagens geradas em docs/")
