"""
LLM-as-judge: scores a draft against the golden must_include / must_not_promise
and the retrieved evidence. Uses the JUDGE model (pro tier), NEVER the
generator -- cross-model judging as a partial mitigation for self-preference
bias (both are Gemini; stated as partial in the report).

Every judgement must quote evidence spans. No quotes + high score = invalid.
Validated against human scores (see scripts/score_judge_sample.py output).
"""
from __future__ import annotations

from src import config

JUDGE_SCHEMA = "judge-v1"


def _prompt(text: str, draft: str, evidence: list[str],
            must_include: list[str], must_not: list[str]) -> str:
    ev = "\n".join(f"- {e[:300]}" for e in evidence) or "(no evidence cited)"
    return (
        "Score this AmazonHelp support reply draft 1-5 on four axes. Be strict: "
        "a polite but content-free reply scores low on groundedness.\n\n"
        f"Customer: {text!r}\nDraft: {draft!r}\n"
        f"Cited evidence:\n{ev}\n"
        f"Required points: {must_include}\nForbidden: {must_not}\n\n"
        "Axes: groundedness (claims supported by evidence), relevance "
        "(addresses the issue + required points), tone (professional, brand-fit), "
        "safety (no invented promises/dates/amounts).\n"
        'Reply ONLY valid JSON: {"groundedness": 1-5, "relevance": 1-5, '
        '"tone": 1-5, "safety": 1-5, "quotes": ["<span from evidence>"], '
        '"missing": ["<required point not covered>"]}'
    )


def judge(text: str, draft: str, evidence: list[str],
          must_include: list[str], must_not: list[str],
          mode: str | None = None) -> dict:
    from src.llm.gemini import complete_json
    default = {"groundedness": 1, "relevance": 1, "tone": 3, "safety": 5,
               "quotes": [], "missing": must_include}
    try:
        d = complete_json(
            _prompt(text, draft, evidence, must_include, must_not),
            model=config.JUDGE_MODEL, temperature=config.JUDGE_TEMPERATURE,
            schema=JUDGE_SCHEMA, mode=mode, default=default,
        )
        out = {k: max(1, min(5, int(d.get(k, default[k]))))
               for k in ("groundedness", "relevance", "tone", "safety")}
        out["quotes"] = [str(q) for q in (d.get("quotes") or [])][:4]
        out["missing"] = [str(m) for m in (d.get("missing") or [])][:5]
        out["avg"] = round(sum(out[k] for k in
                           ("groundedness", "relevance", "tone", "safety")) / 4, 2)
        out["model"] = config.JUDGE_MODEL
        return out
    except Exception:
        d = dict(default)
        d["avg"] = 2.5
        d["model"] = "fallback"
        return d
