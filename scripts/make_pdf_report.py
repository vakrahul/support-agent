"""Generate a publication-grade PDF report using Python-Markdown and headless Edge.
Strict 6-page layout, zero double-replacement artifacts, pure white styling.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "report" / "REPORT.md"
OUT_HTML = ROOT / "report" / "report.html"
OUT_PDF = ROOT / "report" / "REPORT.pdf"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

@page {
    size: A4;
    margin: 10mm 12mm 10mm 12mm;
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-family: 'Inter', sans-serif;
        font-size: 7.2pt;
        color: #64748b;
    }
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 8.3pt;
    line-height: 1.34;
    color: #0f172a;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 14.5pt;
    font-weight: 700;
    color: #0f172a;
    margin-top: 0;
    margin-bottom: 3px;
    padding-bottom: 4px;
    border-bottom: 2px solid #2563eb;
    letter-spacing: -0.02em;
}

.subtitle {
    font-size: 8.0pt;
    color: #475569;
    margin-bottom: 8px;
    font-weight: 500;
}

.meta-badge {
    display: inline-block;
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 7.2pt;
    font-weight: 600;
    color: #334155;
    margin-right: 4px;
}

h2 {
    font-size: 10.5pt;
    font-weight: 700;
    color: #1e293b;
    margin-top: 9px;
    margin-bottom: 3px;
    padding-bottom: 2px;
    border-bottom: 1px solid #e2e8f0;
    letter-spacing: -0.01em;
}

h3 {
    font-size: 9.0pt;
    font-weight: 600;
    color: #334155;
    margin-top: 7px;
    margin-bottom: 2px;
}

p {
    margin-top: 0;
    margin-bottom: 4px;
    text-align: justify;
}

ul, ol {
    margin-top: 1px;
    margin-bottom: 4px;
    padding-left: 16px;
}

li {
    margin-bottom: 2px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 5px 0 7px 0;
    font-size: 7.6pt;
    page-break-inside: avoid;
}

th, td {
    padding: 3px 5px;
    text-align: left;
    border: 1px solid #cbd5e1;
}

th {
    background-color: #f8fafc;
    font-weight: 600;
    color: #0f172a;
}

tr:nth-child(even) td {
    background-color: #f8fafc;
}

code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 7.6pt;
    background: #f1f5f9;
    padding: 1px 3px;
    border-radius: 3px;
    color: #0f172a;
    border: 1px solid #e2e8f0;
}

pre {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 4px;
    font-size: 7.0pt;
    line-height: 1.20;
    overflow-x: auto;
    margin: 4px 0;
}

pre code {
    background: none;
    border: none;
    padding: 0;
}

blockquote {
    border-left: 3px solid #2563eb;
    background: #eff6ff;
    padding: 3px 8px;
    margin: 4px 0;
    font-size: 7.8pt;
    color: #1e40af;
}

.figure-container {
    text-align: center;
    margin: 5px 0 6px 0;
    page-break-inside: avoid;
}

.figure-container img {
    max-width: 100%;
    max-height: 70mm;
    height: auto;
    display: block;
    margin: 0 auto;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    background-color: #ffffff;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
}

.figure-caption {
    font-size: 7.2pt;
    color: #475569;
    margin-top: 3px;
    font-weight: 600;
    letter-spacing: 0.01em;
}

.page-break {
    page-break-before: always;
    break-before: page;
    clear: both;
    height: 0;
    margin: 0;
    padding: 0;
}
"""


def build_report_html():
    raw_md = REPORT_MD.read_text(encoding="utf-8")

    # Paths to figures (absolute posix for browser file loading)
    brand_img = (ROOT / "report" / "figures" / "brand_selection.png").as_posix()
    arch_img = (ROOT / "report" / "figures" / "architecture.png").as_posix()
    demo_img = (ROOT / "report" / "figures" / "demo_run.png").as_posix()
    integrations_img = (ROOT / "report" / "figures" / "integrations.png").as_posix()

    # Anchor mappings with exact replacements (no double-substring collisions)
    replacements = {
        "<!-- FIGURE_1_BRAND_SELECTION -->": (
            f'<div class="figure-container">'
            f'<img src="{brand_img}" alt="Empirical Brand Selection Across 108 Brands">'
            f'<div class="figure-caption">Figure 1: Empirical Brand Selection Across 108 Brands (Volume vs. Resolution Rate)</div>'
            f'</div>'
        ),
        "<!-- FIGURE_2_ARCHITECTURE -->": (
            f'<div class="figure-container">'
            f'<img src="{arch_img}" alt="System Architecture & Policy Gate">'
            f'<div class="figure-caption">Figure 2: End-to-End System Pipeline & 6-Point Policy Safety Gate Architecture</div>'
            f'</div>'
        ),
        "<!-- FIGURE_3_DEMO_RUN -->": (
            f'<div class="figure-container">'
            f'<img src="{demo_img}" alt="Real Customer Query Execution Demonstrations">'
            f'<div class="figure-caption">Figure 3: Real Customer Executions: Safe Routine Automation (amz-003) vs. Policy Gate Intercept (amz-000)</div>'
            f'</div>'
        ),
        "<!-- FIGURE_4_INTEGRATIONS -->": (
            f'<div class="figure-container">'
            f'<img src="{integrations_img}" alt="Integration Roadmap">'
            f'<div class="figure-caption">Figure 4: Integration Roadmap: Production Twitter/X Webhook Bot & Model Context Protocol (MCP) Server</div>'
            f'</div>'
        ),
        "<!-- PAGE_BREAK -->": '<div class="page-break"></div>',
    }

    for anchor, html_replacement in replacements.items():
        raw_md = raw_md.replace(anchor, html_replacement)

    html_content = markdown.markdown(
        raw_md,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI Customer Support Agent: Engineering Technical Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="subtitle">
    <span class="meta-badge">Engineering Technical Report</span>
    <span class="meta-badge">Target Domain: AmazonHelp (Twitter / X)</span>
    <span class="meta-badge">Benchmark: Frozen test-100 (Calibrated on cal-50)</span>
    <span class="meta-badge">Status: Production-Verified & Replayable ($0)</span>
</div>
{html_content}
</body>
</html>
"""
    OUT_HTML.write_text(full_html, encoding="utf-8")
    print(f"[report] HTML compiled -> {OUT_HTML}")
    return OUT_HTML


def convert_html_to_pdf(html_path: Path, pdf_path: Path):
    edge_paths = [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    ]
    browser_bin = None
    for p in edge_paths:
        if Path(p).exists():
            browser_bin = p
            break

    if not browser_bin:
        print("[report] No headless browser found for PDF export.")
        return False

    cmd = [
        browser_bin,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        str(html_path.resolve())
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and pdf_path.exists():
        size_kb = pdf_path.stat().st_size / 1024
        print(f"[report] SUCCESS: PDF generated -> {pdf_path} ({size_kb:.1f} KB)")
        return True
    else:
        print(f"[report] Failed to generate PDF: {res.stderr}")
        return False


if __name__ == "__main__":
    h = build_report_html()
    convert_html_to_pdf(h, OUT_PDF)
