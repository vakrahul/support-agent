"""Generate report/figures/integrations.png showing Twitter Bot and MCP Server architecture.
Pure white-and-white background, executive publication grade.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle

OUT = Path(__file__).resolve().parents[1] / "report" / "figures" / "integrations.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

fig_w, fig_h = 13.0, 6.2
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title and Subtitle
ax.text(50, 95.5, "Next-Week Integration Architecture: Live Twitter Bot & Model Context Protocol (MCP) Server",
        ha="center", va="center", fontsize=12.5, weight="bold", color="#0f172a", fontfamily="sans-serif")
ax.text(50, 91.5, "Production Roadmap: Autonomous Twitter Streaming Ingestion and Standardized Tooling for External AI Agents",
        ha="center", va="center", fontsize=8.5, color="#64748b", fontfamily="sans-serif")

# Helper to draw rounded cards
def draw_card(x, y, w, h, bg="#ffffff", border="#cbd5e1", radius=1.5, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={radius}",
                       facecolor=bg, edgecolor=border, linewidth=lw, zorder=2)
    ax.add_patch(p)

def draw_pill(x, y, w, h, text, bg="#f1f5f9", border="#cbd5e1", text_color="#1e293b", weight="bold", fontsize=7.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=1.0",
                       facecolor=bg, edgecolor=border, linewidth=1.0, zorder=4)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=text_color, fontfamily="sans-serif", zorder=5)

# ==========================================
# LEFT PANEL: LIVE TWITTER / X WEBHOOK BOT
# ==========================================
# Container Card
draw_card(3, 8, 45, 78, bg="#f8fafc", border="#cbd5e1", radius=2.0)
draw_pill(5, 80.5, 41, 4.0, "TRACK A: PRODUCTION TWITTER / X WEBHOOK BOT", bg="#eff6ff", border="#93c5fd", text_color="#1d4ed8", fontsize=8.0)

# Step 1: Twitter Event Ingest
draw_card(6, 64, 39, 13, bg="#ffffff", border="#e2e8f0", radius=1.2)
ax.text(8, 73.5, "1. Twitter Account Activity API (Webhook)", fontsize=8.5, weight="bold", color="#0f172a", fontfamily="sans-serif")
ax.text(8, 68.0, "• Real-time webhook stream for @AmazonHelp mentions & DMs\n• Payload normalization, de-duplication, language filter (EN)",
        fontsize=7.2, color="#475569", fontfamily="sans-serif")

# Arrow 1 -> 2
ax.annotate("", xy=(25.5, 60.5), xytext=(25.5, 63.8),
            arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.8))

# Step 2: Support Agent Core Execution
draw_card(6, 44.5, 39, 15.5, bg="#ffffff", border="#3b82f6", radius=1.2, lw=1.5)
ax.text(8, 56.5, "2. Autonomous Support Agent Pipeline", fontsize=8.5, weight="bold", color="#1d4ed8", fontfamily="sans-serif")
ax.text(8, 48.0, "• Few-Shot Intent Classification (6 intents)\n• Vector Precedent Retrieval (Qdrant top-3 non-deflections)\n• Grounded RAG Generation + 6-Point Policy Safety Gate",
        fontsize=7.2, color="#334155", fontfamily="sans-serif")

# Decision Split Arrows
ax.annotate("", xy=(15.5, 36.5), xytext=(20.0, 44.2),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=1.8))
ax.annotate("", xy=(35.5, 36.5), xytext=(31.0, 44.2),
            arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8))

# Step 3A: Auto-Handle Dispatch
draw_card(5.5, 12, 19.5, 24, bg="#f0fdf4", border="#86efac", radius=1.2)
draw_pill(7.5, 31, 15.5, 3.5, "OK_AUTO (17% Cases)", bg="#dcfce7", border="#4ade80", text_color="#15803d", fontsize=7.2)
ax.text(15.25, 27.0, "Direct Auto-Reply", ha="center", fontsize=8.0, weight="bold", color="#14532d", fontfamily="sans-serif")
ax.text(7.5, 16.5, "• Dispatches tweet via\n  Twitter REST v2 API\n• Zero human touch\n• Cited policy links\n• Sub-second reply time",
        fontsize=7.0, color="#166534", fontfamily="sans-serif")

# Step 3B: Escalate to Human Copilot
draw_card(26.5, 12, 19.5, 24, bg="#fef2f2", border="#fca5a5", radius=1.2)
draw_pill(28.5, 31, 15.5, 3.5, "ESCALATE (83% Cases)", bg="#fee2e2", border="#f87171", text_color="#b91c1c", fontsize=7.2)
ax.text(36.25, 27.0, "Human Agent Queue", ha="center", fontsize=8.0, weight="bold", color="#7f1d1d", fontfamily="sans-serif")
ax.text(28.5, 16.5, "• Webhook to Zendesk/Slack\n• Pre-drafted reply attached\n• Cited precedent IDs\n• Specific reason code\n• 1-click human approve",
        fontsize=7.0, color="#991b1b", fontfamily="sans-serif")


# ==========================================
# RIGHT PANEL: MODEL CONTEXT PROTOCOL (MCP)
# ==========================================
# Container Card
draw_card(52, 8, 45, 78, bg="#f8fafc", border="#cbd5e1", radius=2.0)
draw_pill(54, 80.5, 41, 4.0, "TRACK B: MODEL CONTEXT PROTOCOL (MCP) SERVER", bg="#fdf4ff", border="#f0abfc", text_color="#a21caf", fontsize=8.0)

# Step 1: External AI Agents
draw_card(55, 64, 39, 13, bg="#ffffff", border="#e2e8f0", radius=1.2)
ax.text(57, 73.5, "1. External AI Agents & Copilots", fontsize=8.5, weight="bold", color="#0f172a", fontfamily="sans-serif")
ax.text(57, 68.0, "• Claude Desktop, Cursor AI, LangChain, Multi-Agent Swarms\n• Standard MCP Client connecting over stdio or SSE transport",
        fontsize=7.2, color="#475569", fontfamily="sans-serif")

# Bidirectional Arrow 1 <-> 2
ax.annotate("", xy=(74.5, 60.5), xytext=(74.5, 63.8),
            arrowprops=dict(arrowstyle="<->", color="#c026d3", lw=1.8))

# Step 2: MCP Server Layer
draw_card(55, 44.5, 39, 15.5, bg="#ffffff", border="#c026d3", radius=1.2, lw=1.5)
ax.text(57, 56.5, "2. support-agent-mcp Server Interface", fontsize=8.5, weight="bold", color="#86198f", fontfamily="sans-serif")
ax.text(57, 48.0, "• Exposes standardized tool schema over JSON-RPC\n• Enforces safety invariants, rate limits, and audit telemetry\n• Decouples AI reasoning client from domain grounding data",
        fontsize=7.2, color="#334155", fontfamily="sans-serif")

# Arrow 2 -> Tools
ax.annotate("", xy=(74.5, 37.0), xytext=(74.5, 44.2),
            arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=1.8))

# Step 3: Four Atomic MCP Tools
draw_card(54.5, 12, 40, 24, bg="#faf5ff", border="#d8b4fe", radius=1.2)
ax.text(74.5, 32.5, "Standardized MCP Tool Suite for Customer Resolution", ha="center", fontsize=8.0, weight="bold", color="#581c87", fontfamily="sans-serif")

tools = [
    ("classify_intent", "Categorize raw multi-line query into 6 intents with confidence"),
    ("retrieve_precedents", "Cosine search over 3,334 verified historical resolutions"),
    ("draft_grounded_reply", "Synthesize reply strictly conditioned on retrieved precedent text"),
    ("evaluate_policy_gate", "Audit draft against 6 policy rules; emit OK_AUTO or ESCALATE")
]

for idx, (tname, tdesc) in enumerate(tools):
    ty = 27.5 - (idx * 4.8)
    draw_pill(56.5, ty, 15.0, 3.4, tname, bg="#ffffff", border="#c084fc", text_color="#6b21a8", fontsize=6.8)
    ax.text(73.0, ty + 1.7, tdesc, va="center", fontsize=6.5, color="#4c1d95", fontfamily="sans-serif")

fig.tight_layout()
fig.savefig(OUT, dpi=220, facecolor="#ffffff", edgecolor="none")
plt.close(fig)
print(f"[integrations] Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
