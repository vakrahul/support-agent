"""Generate report/figures/architecture.png (no external deps beyond matplotlib)."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUT = Path(__file__).resolve().parents[1] / "report" / "figures" / "architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(12, 4))
ax.set_xlim(0, 12)
ax.set_ylim(0, 4)
ax.axis("off")

boxes = [
    (0.2, 1.2, 1.5, 1.6, "Twitter\nmessage", "#e3f2fd"),
    (2.1, 1.2, 1.5, 1.6, "Classify\nLLM few-shot\n6 intents", "#fff3e0"),
    (4.0, 1.2, 1.7, 1.6, "Retrieve\nQdrant embedded\ntop-3, local emb", "#e8f5e9"),
    (6.1, 1.2, 1.7, 1.6, "Draft RAG\n+ grounding\nvalidator", "#fce4ec"),
    (8.2, 1.2, 1.5, 1.6, "Gate\nAUTO / HUMAN\n+ reason code", "#f3e5f5"),
    (10.1, 1.2, 1.5, 1.6, "Eval\nmetrics + judge\nCIs, curve", "#e0f7fa"),
]

for x, y, w, h, label, color in boxes:
    ax.add_patch(patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                        facecolor=color, edgecolor="#444"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8)

for i in range(len(boxes) - 1):
    x1 = boxes[i][0] + boxes[i][2]
    x2 = boxes[i + 1][0]
    y = 2.0
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="->", color="#444"))

ax.text(6, 3.4, "AmazonHelp support agent — classify / retrieve / draft / gate / eval", ha="center", fontsize=10, weight="bold")
ax.text(6, 0.5, "Thresholds from cal-50 only. Test-100 frozen. Replay cache pins numbers.", ha="center", fontsize=7, style="italic")
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
