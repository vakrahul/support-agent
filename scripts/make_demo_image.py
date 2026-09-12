"""Render a sleek macOS dark-mode terminal window mockup from a REAL pipeline run.

Uses real pipeline execution (replay cache, $0, deterministic) via
scripts/demo.run_pipeline and draws a high-fidelity macOS terminal window.

Usage:
    python scripts/make_demo_image.py                 # Default: amz-003 (Auto-handle)
    python scripts/make_demo_image.py --id amz-001    # Precedent conflict (Escalate)
    python scripts/make_demo_image.py --id amz-000    # High risk (Escalate)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

from demo import run_pipeline
from src.retrieve.index import Retriever
from src.classify.classifier import LLMClassifier


def calculate_calibrated_conf(raw_conf: float) -> float:
    """Platt scaling fitted on frozen cal-50 (see outputs/calibration_results.json)."""
    a = 0.4597
    b = -0.4094
    p = max(min(raw_conf, 1.0 - 1e-4), 1e-4)
    log_odds = math.log(p / (1.0 - p))
    logits = a * log_odds + b
    return round(1.0 / (1.0 + math.exp(-logits)), 3)


def render_terminal_demo(
    sample_id: str = "amz-003",
    output_path: Path | None = None,
    mode: str = "replay",
) -> Path:
    if output_path is None:
        output_path = ROOT / "report" / "figures" / "demo_run.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Run real pipeline on real golden sample
    gold_path = ROOT / "data" / "golden" / "golden.jsonl"
    gold = {
        r["id"]: r
        for r in (json.loads(line) for line in open(gold_path, encoding="utf-8") if line.strip())
    }
    if sample_id not in gold:
        raise ValueError(f"Unknown sample ID {sample_id!r}. Must be in golden.jsonl.")

    g = gold[sample_id]
    ret = Retriever()
    clf = LLMClassifier()
    res = run_pipeline(g["text"], ret, clf, mode=mode)

    cust_text = res["text"].replace("\n", " ").strip()
    cust_display = "".join(ch for ch in cust_text if ord(ch) < 0xFFFF)

    cal_conf = calculate_calibrated_conf(res["score"])
    is_auto = (res["decision"] == "auto")
    pred_intent = res["intent"].upper()

    mono_font = "Consolas"
    try:
        from matplotlib import font_manager
        available = {f.name for f in font_manager.fontManager.ttflist}
        if "Consolas" in available:
            mono_font = "Consolas"
        elif "DejaVu Sans Mono" in available:
            mono_font = "DejaVu Sans Mono"
        elif "Menlo" in available:
            mono_font = "Menlo"
    except Exception:
        mono_font = "monospace"

    # Dimensions tuned to wrap tightly with zero wasted space at the bottom
    fig_w, fig_h = 13.0, 7.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    fig.patch.set_facecolor("#080c14")  # outer dark canvas backdrop
    ax.set_facecolor("#080c14")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    win_x, win_y, win_w, win_h = 3.5, 4.0, 93.0, 92.0

    # Window Drop Shadow
    shadow = FancyBboxPatch(
        (win_x + 0.6, win_y - 0.8),
        win_w, win_h,
        boxstyle="round,pad=0.8,rounding_size=2.0",
        facecolor="#020408",
        edgecolor="none",
        alpha=0.75,
        zorder=1,
    )
    ax.add_patch(shadow)

    # Main Terminal Body
    term_box = FancyBboxPatch(
        (win_x, win_y),
        win_w, win_h,
        boxstyle="round,pad=0.8,rounding_size=2.0",
        facecolor="#0d111a",
        edgecolor="#1e293b",
        linewidth=1.4,
        zorder=2,
    )
    ax.add_patch(term_box)

    # Header: macOS Traffic Light Dots
    header_y = win_y + win_h - 5.0
    dot_r = 0.85
    ax.add_patch(Circle((win_x + 3.8, header_y), dot_r, facecolor="#ff5f56", edgecolor="#e0443e", linewidth=0.6, zorder=5))
    ax.add_patch(Circle((win_x + 6.6, header_y), dot_r, facecolor="#ffbd2e", edgecolor="#dea123", linewidth=0.6, zorder=5))
    ax.add_patch(Circle((win_x + 9.4, header_y), dot_r, facecolor="#27c93f", edgecolor="#1aab29", linewidth=0.6, zorder=5))

    # Centered Header Title
    window_title = "Commerce Support AI Agent — demo.py"
    ax.text(
        50.0, header_y, window_title,
        color="#94a3b8", fontsize=10.2, fontfamily=mono_font, fontweight="500",
        ha="center", va="center", zorder=5
    )

    left_x = win_x + 3.8
    right_x = win_x + win_w - 3.8

    def draw_divider(y_pos, style="dash"):
        if style == "double":
            ax.plot([left_x, right_x], [y_pos + 0.35, y_pos + 0.35], color="#1e293b", linewidth=1.1, zorder=4)
            ax.plot([left_x, right_x], [y_pos - 0.35, y_pos - 0.35], color="#1e293b", linewidth=1.1, zorder=4)
        else:
            ax.plot([left_x, right_x], [y_pos, y_pos], color="#1e293b", linestyle="--", linewidth=1.1, zorder=4)

    # 1. Command Prompt
    cur_y = 85.2
    ax.text(left_x, cur_y, "$ ", color="#38bdf8", fontsize=11.2, fontfamily=mono_font, fontweight="bold", zorder=4)
    ax.text(left_x + 2.4, cur_y, f"python demo.py --preset {sample_id}", color="#f1f5f9", fontsize=11.2, fontfamily=mono_font, zorder=4)

    # Divider 1
    cur_y -= 4.2
    draw_divider(cur_y, style="double")

    # 2. Customer Query Block
    cur_y -= 4.6
    prefix = f"Customer Query [{sample_id}]: "
    ax.text(left_x, cur_y, prefix, color="#38bdf8", fontsize=9.8, fontfamily=mono_font, fontweight="bold", zorder=4)

    full_query_str = f'"{cust_display}"'
    # Start immediately after the prefix (16.2 units from left_x)
    first_line_wrap = 62
    q_lines = textwrap.wrap(full_query_str, width=first_line_wrap)
    for idx, ql in enumerate(q_lines):
        if idx == 0:
            ax.text(left_x + 16.5, cur_y, ql, color="#f8fafc", fontsize=9.6, fontfamily=mono_font, zorder=4)
        else:
            cur_y -= 3.6
            ax.text(left_x, cur_y, ql, color="#f8fafc", fontsize=9.6, fontfamily=mono_font, zorder=4)

    # Divider 2
    cur_y -= 4.2
    draw_divider(cur_y, style="dash")

    # 3. Model & Diagnostics Signals
    # Line A: Predicted Intent
    cur_y -= 4.6
    ax.text(left_x, cur_y, "Predicted Intent : ", color="#38bdf8", fontsize=9.5, fontfamily=mono_font, fontweight="bold", zorder=4)
    ax.text(left_x + 15.5, cur_y, pred_intent, color="#f59e0b", fontsize=9.8, fontfamily=mono_font, fontweight="bold", zorder=4)
    intent_len = len(pred_intent) * 1.08
    ax.text(
        left_x + 16.0 + intent_len, cur_y,
        f"(raw conf: {res['score']:.2f} → calibrated: {cal_conf:.3f})",
        color="#94a3b8", fontsize=9.2, fontfamily=mono_font, zorder=4
    )

    # Line B: Retrieval Metric
    cur_y -= 4.0
    ax.text(left_x, cur_y, "Retrieval Metric : ", color="#38bdf8", fontsize=9.5, fontfamily=mono_font, fontweight="bold", zorder=4)
    ret_str = f"Top-1 Sim: {res['max_sim']:.3f} | Precedent Agreement: {res['agreement']:.3f} ({res['matching_hits']}/{res['total_hits']} precedents agree)"
    ax.text(left_x + 15.5, cur_y, ret_str, color="#cbd5e1", fontsize=9.2, fontfamily=mono_font, zorder=4)

    # Line C: Risk Triggers
    cur_y -= 4.0
    ax.text(left_x, cur_y, "Risk Triggers    : ", color="#38bdf8", fontsize=9.5, fontfamily=mono_font, fontweight="bold", zorder=4)
    if res["risk"]:
        risk_str = f"TRIGGERED ({res['risk']!r} keyword detected)"
        ax.text(left_x + 15.5, cur_y, risk_str, color="#ef4444", fontsize=9.2, fontfamily=mono_font, fontweight="bold", zorder=4)
    else:
        risk_str = "None detected (legal: clean, fraud: clean, safety: clean, distress: clean)"
        ax.text(left_x + 15.5, cur_y, risk_str, color="#cbd5e1", fontsize=9.2, fontfamily=mono_font, zorder=4)

    # Line D: Grounding Check
    cur_y -= 4.0
    ax.text(left_x, cur_y, "Grounding Check  : ", color="#38bdf8", fontsize=9.5, fontfamily=mono_font, fontweight="bold", zorder=4)
    if res["grounding_passed"]:
        ax.text(left_x + 15.5, cur_y, "PASSED ", color="#22c55e", fontsize=9.5, fontfamily=mono_font, fontweight="bold", zorder=4)
        ax.text(left_x + 22.0, cur_y, f"(score: {res['grounding_score']:.2f} / 5.0 — verified against historical precedents)", color="#cbd5e1", fontsize=9.2, fontfamily=mono_font, zorder=4)
    else:
        ax.text(left_x + 15.5, cur_y, "FAILED ", color="#ef4444", fontsize=9.5, fontfamily=mono_font, fontweight="bold", zorder=4)
        ax.text(left_x + 22.0, cur_y, f"(score: {res['grounding_score']:.2f} / 5.0 — {res['grounding_reason']})", color="#cbd5e1", fontsize=9.2, fontfamily=mono_font, zorder=4)

    # Divider 3
    cur_y -= 4.2
    draw_divider(cur_y, style="dash")

    # 4. Routing Decision Block
    cur_y -= 4.6
    ax.text(left_x, cur_y, "Routing Decision : ", color="#38bdf8", fontsize=9.5, fontfamily=mono_font, fontweight="bold", zorder=4)

    badge_bg = "#064e3b" if is_auto else "#450a0a"
    badge_border = "#10b981" if is_auto else "#ef4444"
    badge_text_col = "#34d399" if is_auto else "#f87171"
    badge_text = "[AUTO-HANDLE]" if is_auto else "[ESCALATE TO HUMAN]"
    badge_w = 15.8 if is_auto else 22.5

    badge = FancyBboxPatch(
        (left_x + 15.5, cur_y - 1.5),
        badge_w, 3.2,
        boxstyle="round,pad=0.2,rounding_size=0.6",
        facecolor=badge_bg,
        edgecolor=badge_border,
        linewidth=1.2,
        zorder=5,
    )
    ax.add_patch(badge)
    ax.text(
        left_x + 15.5 + (badge_w / 2.0), cur_y + 0.1, badge_text,
        color=badge_text_col, fontsize=9.3, fontfamily=mono_font, fontweight="bold",
        ha="center", va="center", zorder=6
    )

    # Line E: Reason Code
    cur_y -= 4.2
    ax.text(left_x, cur_y, "Reason Code      : ", color="#38bdf8", fontsize=9.5, fontfamily=mono_font, fontweight="bold", zorder=4)
    ax.text(left_x + 15.5, cur_y, res["reason_code"], color="#f1f5f9", fontsize=9.2, fontfamily=mono_font, fontweight="bold", zorder=4)

    # Line F: Policy Reason
    cur_y -= 4.0
    ax.text(left_x, cur_y, "Policy Reason    : ", color="#38bdf8", fontsize=9.5, fontfamily=mono_font, fontweight="bold", zorder=4)
    ax.text(left_x + 15.5, cur_y, res["reason"][:76], color="#cbd5e1", fontsize=9.2, fontfamily=mono_font, zorder=4)

    # Divider 4
    cur_y -= 4.2
    draw_divider(cur_y, style="dash")

    # 5. Grounded Drafted Reply
    cur_y -= 4.6
    ax.text(
        left_x, cur_y,
        "Drafted Resolution Reply (≤280 chars, URL guardrail verified):",
        color="#38bdf8", fontsize=9.5, fontfamily=mono_font, fontweight="bold", zorder=4
    )

    cur_y -= 3.6
    draft_clean = res["draft"].replace("\n", " ").strip()
    reply_lines = textwrap.wrap(f'"{draft_clean}"', width=75)
    
    quote_height = len(reply_lines) * 3.4 + 1.2
    bar_top = cur_y + 1.2
    bar_bottom = bar_top - quote_height

    # Cyan Left Accent Bar
    accent_bar = FancyBboxPatch(
        (left_x + 0.6, bar_bottom),
        0.5, quote_height,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor="#0284c7",
        edgecolor="none",
        zorder=5,
    )
    ax.add_patch(accent_bar)

    reply_y = cur_y
    for rline in reply_lines:
        ax.text(
            left_x + 2.8, reply_y, rline,
            color="#cbd5e1", fontsize=9.3, fontfamily=mono_font, zorder=4
        )
        reply_y -= 3.3

    # Bottom Terminal Divider
    cur_y = reply_y - 2.0
    draw_divider(cur_y, style="double")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none", dpi=220)
    plt.close(fig)
    print(f"Generated real pipeline terminal mockup for {sample_id} -> {output_path}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render terminal demo image from real pipeline runs")
    parser.add_argument("--id", "--preset", dest="id", default="amz-003", help="Golden sample ID (default: amz-003)")
    parser.add_argument("--out", default=str(ROOT / "report" / "figures" / "demo_run.png"), help="Output PNG path")
    parser.add_argument("--mode", default="replay", choices=["replay", "live"], help="LLM mode")
    args = parser.parse_args()

    render_terminal_demo(sample_id=args.id, output_path=Path(args.out), mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
