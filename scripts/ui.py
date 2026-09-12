"""Real-time web UI for the AmazonHelp agent (stdlib only, no new deps).

Run:      python scripts/ui.py            (then open http://localhost:8000)
Pipeline: classify -> retrieve -> draft -> gate, live per request.
Modes:    replay (default; golden ids only, $0, deterministic)
          live (needs GEMINI_API_KEY; any text)
"""
from __future__ import annotations
import json
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from demo import run_pipeline, load_golden_lookup
from src import config

GOLDEN = load_golden_lookup()
INTENT_COLORS = {
    "delivery_delay": "#f59e0b", "missing_parcel_tracking": "#10b981",
    "refund_return": "#3b82f6", "device_app_account": "#8b5cf6",
    "order_status_general": "#06b6d4", "other_unclear": "#94a3b8",
}

PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AmazonHelp Agent</title>
<style>
*{box-sizing:border-box} body{font-family:'Segoe UI',system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px} .sub{color:#94a3b8;font-size:12px;margin-bottom:18px}
.panel{max-width:880px;margin:0 auto}
textarea{width:100%;height:70px;background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:10px;font-size:14px;resize:vertical}
.row{display:flex;gap:8px;margin-top:8px;align-items:center;flex-wrap:wrap}
button{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:10px 22px;font-size:14px;font-weight:600;cursor:pointer}
button:hover{background:#3b82f6} select{background:#1e293b;color:#e2e8f0;border:1px solid #334155;border-radius:8px;padding:9px;font-size:13px}
.mode{font-size:12px;color:#94a3b8}
#out{margin-top:18px;display:none}
.card{background:#1e293b;border-radius:12px;padding:14px;margin-bottom:12px;border:1px solid #334155}
.bubble{background:#e0f2fe;color:#0c4a6e;border-radius:12px;padding:10px 14px;font-size:14px}
.chip{display:inline-block;padding:5px 12px;border-radius:99px;font-size:12px;font-weight:700;color:#fff;margin:6px 4px 0 0}
.check{font-family:Consolas,monospace;font-size:12px;padding:3px 0}
.pass{color:#4ade80} .trip{color:#f87171}
.draft{background:#dcfce7;color:#14532d;border-radius:12px;padding:10px 14px;font-size:13px;white-space:pre-wrap}
.banner{border-radius:10px;padding:12px;font-weight:800;font-size:16px;color:#fff;text-align:center}
.auto{background:#16a34a} .human{background:#dc2626}
.reason{font-size:12px;color:#94a3b8;margin-top:6px;text-align:center}
.ev{font-size:12px;color:#94a3b8;margin-top:4px}
.spin{display:none;font-size:13px;color:#93c5fd}
</style></head><body><div class="panel">
<h1>AmazonHelp AI Support Agent</h1>
<div class="sub">classify &rarr; retrieve (3,334 precedents) &rarr; draft (grounded) &rarr; gate (auto/escalate + reason) &middot; __MODE_LABEL__</div>
<textarea id="q" placeholder="Type a customer message, or pick a golden example below">__PREFILL__</textarea>
<div class="row">
  <button onclick="go()">Run agent</button>
  <select id="gs" onchange="document.getElementById('q').value=this.value">
    <option value="">— golden example —</option>__OPTIONS__
  </select>
  <span class="mode" id="mode"></span>
  <span class="spin" id="sp">running pipeline (classify &rarr; retrieve &rarr; draft &rarr; gate)…</span>
</div>
<div id="out">
  <div class="card"><div class="bubble" id="qout"></div>
    <span class="chip" id="chip"></span><span class="chip" id="score" style="background:#475569"></span>
    <div class="ev" id="ev"></div></div>
  <div class="card" id="checks"></div>
  <div class="card"><div class="draft" id="draft"></div><div class="ev" id="ground"></div></div>
  <div class="card"><div class="banner" id="dec"></div><div class="reason" id="why"></div></div>
</div>
</div>
<script>
const COLORS = __COLORS__;
async function go() {
  const q = document.getElementById('q').value.trim();
  if (!q) return;
  document.getElementById('sp').style.display='inline';
  document.getElementById('out').style.display='none';
  try {
    const r = await fetch('/api?text=' + encodeURIComponent(q));
    const d = await r.json();
    if (d.error) { alert(d.error); return; }
    document.getElementById('mode').textContent = d.mode + ' mode · ' + d.elapsed_ms + ' ms';
    document.getElementById('qout').textContent = d.text;
    const c = document.getElementById('chip');
    c.textContent = d.intent; c.style.background = COLORS[d.intent] || '#64748b';
    document.getElementById('score').textContent = 'score ' + d.score.toFixed(2);
    document.getElementById('ev').textContent =
      'precedent agreement ' + d.agreement.toFixed(2) + ' · max similarity ' + d.max_sim.toFixed(3) +
      ' · evidence: ' + d.evidence_status;
    document.getElementById('checks').innerHTML = d.policy_checks.map(p =>
      '<div class="check"><span class="' + (p[0]?'pass':'trip') + '">' + (p[0]?'PASS':'TRIP') +
      '</span>  ' + p[1] + ': ' + p[2] + '</div>').join('');
    document.getElementById('draft').textContent = d.draft;
    document.getElementById('ground').textContent =
      'grounding validator: ' + (d.grounding_passed ? 'PASS' : 'FAIL') + ' (' + d.grounding_reason + ')';
    const b = document.getElementById('dec');
    b.textContent = d.decision === 'auto' ? 'AUTO-HANDLE' : 'ESCALATE TO HUMAN';
    b.className = 'banner ' + (d.decision === 'auto' ? 'auto' : 'human');
    document.getElementById('why').textContent = d.reason_code + ' — ' + d.reason;
    document.getElementById('out').style.display='block';
  } catch(e) { alert('request failed: ' + e); }
  finally { document.getElementById('sp').style.display='none'; }
}
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            opts = "".join(
                f'<option value="{t}">{k} — {t[:40]}</option>'
                for k, t in _examples())
            # token substitution (never .format(): JS/CSS braces break it)
            html = (PAGE
                    .replace("__MODE_LABEL__", _mode_label())
                    .replace("__OPTIONS__", opts)
                    .replace("__PREFILL__", GOLDEN["amz-003"]["text"] if GOLDEN else "")
                    .replace("__COLORS__", json.dumps(INTENT_COLORS)))
            self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api":
            import time
            text = urllib.parse.unquote(urllib.parse.parse_qs(parsed.query).get("text", [""])[0])
            mode = "live" if _has_key() else "replay"
            t0 = time.time()
            try:
                res = run_pipeline(text, _ret(), _clf(), mode=mode)
                payload = {k: res[k] for k in (
                    "text", "intent", "score", "top3", "agreement", "max_sim",
                    "evidence_status", "draft", "grounding_passed", "grounding_reason",
                    "policy_checks", "decision", "reason_code", "reason")}
                payload["policy_checks"] = [list(p) for p in res["policy_checks"]]
                payload["mode"] = mode
                payload["elapsed_ms"] = int((time.time() - t0) * 1000)
                self._send(200, json.dumps(payload).encode("utf-8"),
                          "application/json; charset=utf-8")
            except Exception as e:
                msg = str(e)
                hint = ("Replay cache miss — this text was never recorded. "
                        "Use a golden example, or set GEMINI_API_KEY and restart "
                        "for live mode." if "CacheMiss" in type(e).__name__
                        else f"{type(e).__name__}: {msg[:300]}")
                self._send(400, json.dumps({"error": hint}).encode("utf-8"),
                           "application/json; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain")

    def log_message(self, *a):
        pass


_STATE = {}


def _ret():
    if "ret" not in _STATE:
        from src.retrieve.index import Retriever
        _STATE["ret"] = Retriever()
    return _STATE["ret"]


def _clf():
    if "clf" not in _STATE:
        from src.classify.classifier import LLMClassifier
        _STATE["clf"] = LLMClassifier()
    return _STATE["clf"]


def _has_key() -> bool:
    return bool(config.GEMINI_API_KEY)


def _mode_label() -> str:
    return "live mode (Gemini key detected)" if _has_key() else "replay mode ($0, golden examples cached)"


def _examples():
    feat = [("amz-003", "auto-handle (refund)"), ("amz-000", "escalate (fraud)"),
            ("amz-001", "escalate (no agreement)"), ("amz-005", "delay complaint"),
            ("amz-007", "vague query"), ("amz-067", "cancelled order")]
    return [(desc, GOLDEN[i]["text"]) for i, desc in feat if i in GOLDEN]


def main() -> int:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"UI on http://localhost:{port}  ({_mode_label()})")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    raise SystemExit(main())
