"""Generate report/figures/architecture.png with crisp, publication-grade white aesthetic.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "report" / "figures" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

fig_w, fig_h = 13.0, 5.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
fig.patch.set_facecolor("#ffffff")
ax.set_facecolor("#ffffff")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")

# Title and subtitle
ax.text(50, 93.5, "AmazonHelp AI Support Agent: End-to-End Pipeline & 6-Point Policy Safety Gate",
        ha="center", va="center", fontsize=12, weight="bold", color="#0f172a", fontfamily="sans-serif")
ax.text(50, 88.0, "Zero-Network Replay Reproducible ($0) • Grounded Retrieval-Augmented Generation • Strict Deterministic Escalation",
        ha="center", va="center", fontsize=8.0, color="#64748b", fontfamily="sans-serif")

def draw_card(x, y, w, h, bg="#ffffff", border="#cbd5e1", radius=1.2, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.2,rounding_size={radius}",
                       facecolor=bg, edgecolor=border, linewidth=lw, zorder=2)
    ax.add_patch(p)

def draw_pill(x, y, w, h, text, bg="#f1f5f9", border="#cbd5e1", text_color="#1e293b", weight="bold", fontsize=7.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.8",
                       facecolor=bg, edgecolor=border, linewidth=1.0, zorder=4)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fontsize, weight=weight, color=text_color, fontfamily="sans-serif", zorder=5)

# Stage 1: Input
draw_card(2, 28, 14, 48, bg="#f8fafc", border="#cbd5e1")
draw_pill(3.5, 69.5, 11, 4.0, "INPUT", bg="#e2e8f0", border="#94a3b8", text_color="#334155")
ax.text(9, 62, "Customer Tweet", ha="center", fontsize=8.5, weight="bold", color="#0f172a", fontfamily="sans-serif")
ax.text(9, 44, "• Raw incoming\n  Twitter message\n• Language check\n  (English filter)\n• Pre-cleaning &\n  de-duplication",
        ha="center", fontsize=7.2, color="#475569", fontfamily="sans-serif")

# Arrow 1 -> 2
ax.annotate("", xy=(18.5, 52), xytext=(16.2, 52), arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.6))

# Stage 2: Intent Classifier
draw_card(19, 28, 17, 48, bg="#ffffff", border="#93c5fd")
draw_pill(20.5, 69.5, 14, 4.0, "STAGE 1: INTENT", bg="#eff6ff", border="#bfdbfe", text_color="#1d4ed8")
ax.text(27.5, 62, "Few-Shot Classifier", ha="center", fontsize=8.5, weight="bold", color="#1e40af", fontfamily="sans-serif")
ax.text(27.5, 43, "• 5 operational classes\n  + other_unclear\n• Few-shot JSON schema\n• Strict parse validator\n• Raw score ≥ 0.70\n  (tuned on cal-50)",
        ha="center", fontsize=7.0, color="#334155", fontfamily="sans-serif")

# Arrow 2 -> 3
ax.annotate("", xy=(38.5, 52), xytext=(36.2, 52), arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.6))

# Stage 3: Retrieval & Grounding
draw_card(39, 28, 18, 48, bg="#ffffff", border="#86efac")
draw_pill(40.5, 69.5, 15, 4.0, "STAGE 2: RETRIEVE", bg="#f0fdf4", border="#bbf7d0", text_color="#15803d")
ax.text(48, 62, "Qdrant Vector RAG", ha="center", fontsize=8.5, weight="bold", color="#166534", fontfamily="sans-serif")
ax.text(48, 43, "• 3,334 historical pairs\n• Non-deflection filtered\n• all-MiniLM-L6-v2 (384d)\n• Top-3 precedent match\n• Cosine similarity score\n• Precedent intent tags",
        ha="center", fontsize=7.0, color="#334155", fontfamily="sans-serif")

# Arrow 3 -> 4
ax.annotate("", xy=(59.5, 52), xytext=(57.2, 52), arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.6))

# Stage 4: Risk & Draft
draw_card(60, 28, 17, 48, bg="#ffffff", border="#fde047")
draw_pill(61.5, 69.5, 14, 4.0, "STAGE 3: DRAFT", bg="#fefce8", border="#fef08a", text_color="#854d0e")
ax.text(68.5, 62, "Grounded Drafter", ha="center", fontsize=8.5, weight="bold", color="#713f12", fontfamily="sans-serif")
ax.text(68.5, 43, "• Evidence-conditioned\n  LLM generation\n• Risk keyword scanner\n  (fraud, legal, safety)\n• Code validator checks:\n  - Cited IDs exist\n  - Verbatim promise rules",
        ha="center", fontsize=7.0, color="#334155", fontfamily="sans-serif")

# Arrow 4 -> 5
ax.annotate("", xy=(79.5, 52), xytext=(77.2, 52), arrowprops=dict(arrowstyle="->", color="#3b82f6", lw=1.6))

# Stage 5: 6-Point Policy Gate
draw_card(80, 24, 18, 56, bg="#ffffff", border="#c084fc", lw=1.5)
draw_pill(81.5, 73.5, 15, 4.0, "STAGE 4: POLICY GATE", bg="#faf5ff", border="#e9d5ff", text_color="#6b21a8")
ax.text(89, 66.5, "6-Point Verification", ha="center", fontsize=8.5, weight="bold", color="#581c87", fontfamily="sans-serif")
ax.text(89, 47, "1. Intent ≠ other_unclear\n2. Model conf ≥ 0.70\n3. Retrieval sim ≥ 0.60\n4. Intent agreement ≥ 2/3\n5. Risk flags == 0\n6. Code validator == pass",
        ha="center", fontsize=6.8, color="#3b0764", fontfamily="sans-serif")

# Gate Split Arrows
ax.annotate("", xy=(84, 15), xytext=(85, 23.8), arrowprops=dict(arrowstyle="->", color="#059669", lw=1.8))
ax.annotate("", xy=(94, 15), xytext=(93, 23.8), arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.8))

# Output Pills
draw_pill(77, 9, 14, 5.5, "AUTO-HANDLE\n(OK_AUTO: 17%)", bg="#ecfdf5", border="#34d399", text_color="#065f46", fontsize=7.0)
draw_pill(92, 9, 7.5, 5.5, "ESCALATE\n(83%)", bg="#fef2f2", border="#f87171", text_color="#991b1b", fontsize=7.0)

fig.tight_layout()
fig.savefig(OUT, dpi=220, facecolor="#ffffff", edgecolor="none")
plt.close(fig)
print(f"[architecture] Wrote {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")
