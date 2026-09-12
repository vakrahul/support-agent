"""Render a pristine white-background execution inspector from REAL pipeline runs.

Uses real pipeline execution (replay cache, $0, deterministic) via
scripts/demo.run_pipeline and draws high-fidelity white-and-white execution cards.

Usage:
    python scripts/make_demo_image.py
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
    """Platt scaling fitted on frozen cal-50."""
    a = 0.4597
    b = -0.4094
    p = max(min(raw_conf, 1.0 - 1e-4), 1e-4)
    log_odds = math.log(p / (1.0 - p))
    logits = a * log_odds + b
    return round(1.0 / (1.0 + math.exp(-logits)), 3)


def render_white_demo(
    sample_ids: list[str] | None = None,
    output_path: Path | None = None,
    mode: str = "replay",
) -> Path:
    if sample_ids is None:
        sample_ids = ["amz-003", "amz-000"]
    if output_path is None:
        output_path = ROOT / "report" / "figures" / "demo_run.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gold_path = ROOT / "data" / "golden" / "golden.jsonl"
    gold = {
        r["id"]: r
        for r in (json.loads(line) for line in open(gold_path, encoding="utf-8") if line.strip())
    }

    ret = Retriever()
    clf = LLMClassifier()
    runs = []
    for sid in sample_ids:
        if sid in gold:
            g = gold[sid]
            res = run_pipeline(g["text"], ret, clf, mode=mode)
            runs.append((sid, g, res))

    fig_w, fig_h = 13.0, 7.5
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=220)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Overall Header
    ax.text(50, 96.5, "Real Customer Query Demonstrations: Automated vs. Escalated Decision Execution",
            ha="center", va="center", fontsize=11.5, weight="bold", color="#0f172a", fontfamily="sans-serif")
    ax.text(50, 92.5, "Zero-Network Replay Execution ($0) • Grounded Precedent Evidence • Deterministic Policy Verification",
            ha="center", va="center", fontsize=8.0, color="#64748b", fontfamily="sans-serif")

    card_width = 46.5
    card_height = 87.0

    for col_idx, (sid, g, res) in enumerate(runs[:2]):
        cx = 2.5 if col_idx == 0 else 51.0
        cy = 3.5

        is_auto = (res["decision"] == "auto")
        card_border = "#86efac" if is_auto else "#fca5a5"
        card_bg = "#ffffff"
        header_bg = "#f0fdf4" if is_auto else "#fef2f2"
        badge_bg = "#dcfce7" if is_auto else "#fee2e2"
        badge_border = "#4ade80" if is_auto else "#f87171"
        badge_text = "#15803d" if is_auto else "#b91c1c"
        badge_label = "DECISION: AUTO-HANDLE (OK_AUTO)" if is_auto else f"DECISION: ESCALATE ({res['reason_code']})"

        # Main Card Body
        card = FancyBboxPatch(
            (cx, cy), card_width, card_height,
            boxstyle="round,pad=0.2,rounding_size=1.5",
            facecolor=card_bg, edgecolor=card_border, linewidth=1.4, zorder=2
        )
        ax.add_patch(card)

        # Header Bar
        hbar = FancyBboxPatch(
            (cx, cy + card_height - 6.5), card_width, 6.5,
            boxstyle="round,pad=0.2,rounding_size=1.5",
            facecolor=header_bg, edgecolor=card_border, linewidth=1.0, zorder=3
        )
        ax.add_patch(hbar)

        # Traffic light dots
        dot_y = cy + card_height - 3.2
        dot_r = 0.65
        ax.add_patch(Circle((cx + 2.5, dot_y), dot_r, facecolor="#ff5f56", edgecolor="#e0443e", linewidth=0.5, zorder=5))
        ax.add_patch(Circle((cx + 4.5, dot_y), dot_r, facecolor="#ffbd2e", edgecolor="#dea123", linewidth=0.5, zorder=5))
        ax.add_patch(Circle((cx + 6.5, dot_y), dot_r, facecolor="#27c93f", edgecolor="#1aab29", linewidth=0.5, zorder=5))

        case_type = "Case 1: Safe Routine Automation" if is_auto else "Case 2: Safety Gate Intercept (Risk/Escalate)"
        ax.text(cx + 8.5, dot_y, f"{sid} • {case_type}", fontsize=8.0, weight="bold", color="#1e293b", va="center", fontfamily="sans-serif", zorder=5)

        # Decision Badge
        p_y = cy + card_height - 11.5
        badge_patch = FancyBboxPatch(
            (cx + 2.5, p_y), card_width - 5.0, 3.8,
            boxstyle="round,pad=0.1,rounding_size=0.8",
            facecolor=badge_bg, edgecolor=badge_border, linewidth=1.0, zorder=4
        )
        ax.add_patch(badge_patch)
        ax.text(cx + card_width/2, p_y + 1.9, badge_label,
                ha="center", va="center", fontsize=7.8, weight="bold", color=badge_text, fontfamily="sans-serif", zorder=5)

        # Customer Query Box
        q_y = p_y - 17.5
        q_box = FancyBboxPatch(
            (cx + 2.5, q_y), card_width - 5.0, 16.0,
            boxstyle="round,pad=0.1,rounding_size=0.8",
            facecolor="#f8fafc", edgecolor="#e2e8f0", linewidth=1.0, zorder=4
        )
        ax.add_patch(q_box)
        ax.text(cx + 4.0, q_y + 13.8, "Incoming Customer Query:", fontsize=7.2, weight="bold", color="#475569", fontfamily="sans-serif", zorder=5)

        clean_text = g["text"].replace("\n", " ").strip()
        lines = textwrap.wrap(f'"{clean_text}"', width=48)
        ty = q_y + 10.2
        for l in lines[:3]:
            ax.text(cx + 4.0, ty, l, fontsize=6.8, color="#0f172a", fontfamily="sans-serif", zorder=5)
            ty -= 3.2

        # Signals & Metrics Table Box
        s_y = q_y - 25.5
        s_box = FancyBboxPatch(
            (cx + 2.5, s_y), card_width - 5.0, 24.0,
            boxstyle="round,pad=0.1,rounding_size=0.8",
            facecolor="#ffffff", edgecolor="#e2e8f0", linewidth=1.0, zorder=4
        )
        ax.add_patch(s_box)
        ax.text(cx + 4.0, s_y + 21.5, "Pipeline Diagnostic Verification Signals:", fontsize=7.2, weight="bold", color="#334155", fontfamily="sans-serif", zorder=5)

        cal_conf = calculate_calibrated_conf(res["score"])
        diag_rows = [
            ("Predicted Intent", f"{res['intent'].upper()} (calibrated conf: {cal_conf:.3f})"),
            ("Retrieval Match", f"Sim {res['max_sim']:.3f} (top-3 non-deflection pool)"),
            ("Intent Agreement", f"{res['agreement']:.2f} ({res['matching_hits']}/{res['total_hits']} precedents match)"),
            ("Risk Keyword Flags", "None (Clean)" if not res["risk"] else f"TRIGGERED ({res['risk']})"),
            ("Grounding Validator", f"PASS ({res['grounding_score']:.1f}/5.0)" if res["grounding_passed"] else f"FAIL ({res['grounding_reason']})"),
        ]

        dy = s_y + 17.5
        for lbl, val in diag_rows:
            ax.text(cx + 4.0, dy, lbl, fontsize=6.5, color="#64748b", fontfamily="sans-serif", zorder=5)
            val_col = "#059669" if "PASS" in val or "Clean" in val or "OK" in val else ("#dc2626" if "TRIGGERED" in val or "FAIL" in val else "#0f172a")
            ax.text(cx + 19.5, dy, f": {val}", fontsize=6.5, weight="bold", color=val_col, fontfamily="sans-serif", zorder=5)
            dy -= 3.5

        # Grounded Draft / Escalation Summary Box
        d_y = s_y - 22.5
        d_box = FancyBboxPatch(
            (cx + 2.5, d_y), card_width - 5.0, 21.0,
            boxstyle="round,pad=0.1,rounding_size=0.8",
            facecolor="#f8fafc", edgecolor="#cbd5e1", linewidth=1.0, zorder=4
        )
        ax.add_patch(d_box)

        box_title = "Grounded Autonomous Reply (URL Guardrails Enforced):" if is_auto else "Escalation Routing Context to Human Copilot Queue:"
        ax.text(cx + 4.0, d_y + 18.5, box_title, fontsize=7.0, weight="bold", color="#334155", fontfamily="sans-serif", zorder=5)

        if is_auto:
            draft_text = res["draft"].replace("\n", " ").strip()
            dlines = textwrap.wrap(f'"{draft_text}"', width=50)
            dty = d_y + 14.8
            for dl in dlines[:4]:
                ax.text(cx + 4.0, dty, dl, fontsize=6.6, color="#0f172a", fontfamily="monospace", zorder=5)
                dty -= 3.3
        else:
            reason_text = f"Policy Action: {res['reason']}"
            rlines = textwrap.wrap(reason_text, width=50)
            dty = d_y + 14.8
            for rl in rlines[:4]:
                ax.text(cx + 4.0, dty, rl, fontsize=6.6, color="#7f1d1d", fontfamily="sans-serif", zorder=5)
                dty -= 3.3

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, facecolor="#ffffff", edgecolor="none")
    plt.close(fig)
    print(f"[demo_image] Wrote white-and-white demo {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Render white-and-white demo image from real pipeline runs")
    parser.add_argument("--out", default=str(ROOT / "report" / "figures" / "demo_run.png"), help="Output PNG path")
    parser.add_argument("--mode", default="replay", choices=["replay", "live"], help="LLM mode")
    args = parser.parse_args()
    render_white_demo(output_path=Path(args.out), mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
