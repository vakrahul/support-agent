"""Generate a polished, publication-grade PDF report using Python-Markdown and headless Edge.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
REPORT_MD = ROOT / "report" / "REPORT.md"
DECISIONS_MD = ROOT / "DECISIONS.md"
OUT_HTML = ROOT / "report" / "report.html"
OUT_PDF = ROOT / "report" / "REPORT.pdf"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
    @bottom-right {
        content: counter(page) " / " counter(pages);
        font-family: 'Inter', sans-serif;
        font-size: 8pt;
        color: #666;
    }
}

body {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 9.5pt;
    line-height: 1.45;
    color: #1a1a1a;
    background-color: #ffffff;
    margin: 0;
    padding: 0;
}

h1 {
    font-size: 17pt;
    font-weight: 700;
    color: #0f172a;
    margin-top: 0;
    margin-bottom: 6px;
    padding-bottom: 8px;
    border-bottom: 2px solid #2563eb;
    letter-spacing: -0.02em;
}

.subtitle {
    font-size: 10pt;
    color: #475569;
    margin-bottom: 18px;
    font-weight: 500;
}

.meta-badge {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #cbd5e1;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 8pt;
    font-weight: 600;
    color: #334155;
    margin-right: 6px;
}

h2 {
    font-size: 12pt;
    font-weight: 700;
    color: #1e293b;
    margin-top: 16px;
    margin-bottom: 6px;
    padding-bottom: 3px;
    border-bottom: 1px solid #e2e8f0;
    letter-spacing: -0.01em;
}

h3 {
    font-size: 10.5pt;
    font-weight: 600;
    color: #334155;
    margin-top: 12px;
    margin-bottom: 4px;
}

p {
    margin-top: 0;
    margin-bottom: 8px;
    text-align: justify;
}

ul, ol {
    margin-top: 2px;
    margin-bottom: 8px;
    padding-left: 20px;
}

li {
    margin-bottom: 3px;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin: 10px 0 12px 0;
    font-size: 8.5pt;
}

th, td {
    padding: 6px 8px;
    text-align: left;
    border: 1px solid #cbd5e1;
}

th {
    background-color: #f8fafc;
    font-weight: 600;
    color: #1e293b;
}

tr:nth-child(even) td {
    background-color: #f8fafc;
}

code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 8.5pt;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    color: #0f172a;
    border: 1px solid #e2e8f0;
}

pre {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
    padding: 8px;
    font-size: 8pt;
    line-height: 1.35;
    overflow-x: auto;
    margin: 8px 0;
}

pre code {
    background: none;
    border: none;
    padding: 0;
}

blockquote {
    border-left: 3px solid #2563eb;
    background: #eff6ff;
    padding: 6px 12px;
    margin: 8px 0;
    font-size: 9pt;
    color: #1e40af;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 10px auto;
    border: 1px solid #e2e8f0;
    border-radius: 4px;
}

.page-break {
    page-break-before: always;
}

.highlight-box {
    background: #fdf2f8;
    border: 1px solid #fbcfe8;
    border-left: 3px solid #db2777;
    border-radius: 4px;
    padding: 8px 12px;
    margin: 10px 0;
    font-size: 9pt;
}
"""


def build_report_html():
    raw_md = REPORT_MD.read_text(encoding="utf-8")
    decisions_md = DECISIONS_MD.read_text(encoding="utf-8")

    # Replace decision log placeholder with actual decisions text
    if "DECISIONS.md holds exactly 15 entries" in raw_md:
        dec_body = decisions_md.split("\n", 2)[2] if "\n" in decisions_md else decisions_md
        raw_md = raw_md.replace(
            "`DECISIONS.md` holds exactly 15 entries (WHAT/WHY/alternatives/tradeoff) —\nthe ones above, compressed. Overflow detail lives in GOLDEN_NOTE.md.",
            dec_body
        )

    # Path to brand selection image
    img_path = (ROOT / "report" / "figures" / "brand_selection.png").as_posix()
    raw_md = raw_md.replace("`report/figures/brand_selection.png`", f"![Brand Selection]({img_path})")

    html_content = markdown.markdown(
        raw_md,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"]
    )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hiver SDE Take-Home: AI Customer Support Agent Report</title>
<style>{CSS}</style>
</head>
<body>
<div class="subtitle">
    <span class="meta-badge">Candidate Take-Home Report</span>
    <span class="meta-badge">Target: AmazonHelp</span>
    <span class="meta-badge">Frozen Evaluation (test-100)</span>
    <span class="meta-badge">Status: Complete</span>
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
