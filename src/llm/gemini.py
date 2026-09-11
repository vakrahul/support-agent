"""
Gemini client with a deterministic replay cache.

WHY THIS EXISTS
---------------
The brief requires a reviewer to reproduce the headline results in under 15
minutes. If the eval calls a paid API, that promise quietly depends on the
reviewer owning a Gemini key, having quota, and tolerating non-determinism.

So every LLM call goes through `complete()`, which is keyed on
(model, prompt, temperature, response_schema) and backed by a committed
JSONL cache:

    replay (default) -- read from cache/llm_cache.jsonl, never touch the network
    live             -- call Gemini, append new entries to the cache

In replay mode a cache MISS is a hard error, never a silent fallback: a
fabricated or skipped response would corrupt every number in the report.

The cache is JSONL rather than one-file-per-call so it stays greppable and
diffable in git -- a reviewer can read exactly what the model was asked and
what it said.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src import config

CACHE_FILE = config.CACHE_DIR / "llm_cache.jsonl"


class CacheMiss(RuntimeError):
    """Raised when replay mode is asked for a prompt that was never recorded."""


@dataclass
class LLMResponse:
    text: str
    model: str
    cached: bool
    key: str

    def json(self) -> Any:
        """Parse the response as JSON, tolerating the usual model noise."""
        return extract_json(self.text)


# --------------------------------------------------------------------------
# Robust JSON extraction
# --------------------------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def extract_json(text: str) -> Any:
    """Pull a JSON object/array out of a model response.

    Models wrap JSON in markdown fences, prepend "Here's the result:", or emit
    trailing commas. Rather than crashing the eval on a formatting quirk, we
    try progressively more forgiving strategies. If all fail we raise, because
    silently returning a default would show up as a fake metric.
    """
    if text is None:
        raise ValueError("empty LLM response")
    s = text.strip()

    # 1. straight parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2. fenced block
    m = _FENCE_RE.search(s)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            s = m.group(1).strip()

    # 3. first balanced {...} or [...]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = s.find(open_ch)
        if start == -1:
            continue
        depth, in_str, esc = 0, False, False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == open_ch:
                depth += 1
            elif ch == close_ch:
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        # 4. strip trailing commas, retry once
                        cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
                        try:
                            return json.loads(cleaned)
                        except Exception:
                            break
    raise ValueError(f"could not parse JSON from response: {text[:250]!r}")


# --------------------------------------------------------------------------
# Cache
# --------------------------------------------------------------------------
def make_key(prompt: str, model: str, temperature: float, schema: str = "") -> str:
    """Stable cache key. Any change to prompt/model/temp/schema is a new key,
    which means stale results can never masquerade as fresh ones."""
    payload = json.dumps(
        {"p": prompt, "m": model, "t": round(float(temperature), 4), "s": schema},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class LLMCache:
    """In-memory dict backed by an append-only JSONL file."""

    def __init__(self, path: Path | str = CACHE_FILE):
        self.path = Path(path)
        self._store: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated final line
                if "key" in rec:
                    self._store[rec["key"]] = rec  # later entries win

    def get(self, key: str) -> str | None:
        rec = self._store.get(key)
        return rec["response"] if rec else None

    def put(self, key: str, prompt: str, response: str, model: str, temperature: float) -> None:
        rec = {
            "key": key,
            "model": model,
            "temperature": temperature,
            "prompt": prompt,
            "response": response,
            "ts": int(time.time()),
        }
        with self._lock:
            self._store[key] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._store)


_cache: LLMCache | None = None


def get_cache() -> LLMCache:
    global _cache
    if _cache is None:
        _cache = LLMCache()
    return _cache


def reset_cache(path: Path | str | None = None) -> LLMCache:
    """Point the module at a different cache file (used by tests)."""
    global _cache
    _cache = LLMCache(path or CACHE_FILE)
    return _cache


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
# A backend is any callable: (prompt, model, temperature) -> str
Backend = Callable[[str, str, float], str]

_backend: Backend | None = None


def set_backend(fn: Backend | None) -> None:
    """Inject a fake backend. Tests use this to exercise the full call path
    without a network or an API key."""
    global _backend
    _backend = fn


def _gemini_backend(prompt: str, model: str, temperature: float) -> str:
    """Real Gemini call. Imported lazily so the eval path never needs the SDK."""
    try:
        import google.generativeai as genai
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "google-generativeai is not installed. `pip install -r requirements.txt`, "
            "or stay in replay mode (LLM_MODE=replay)."
        ) from e

    if not config.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")

    genai.configure(api_key=config.GEMINI_API_KEY)
    gm = genai.GenerativeModel(model)

    last: Exception | None = None
    for attempt in range(6):  # transient 429/503 are common on free tiers
        try:
            resp = gm.generate_content(
                prompt,
                generation_config={"temperature": temperature},
            )
            return (resp.text or "").strip()
        except Exception as e:  # noqa: BLE001
            last = e
            msg = str(e)
            # Honor the server's asked retry delay on 429s instead of
            # hammering a free-tier quota (measured: 5 parallel workers
            # turned 144/160 judge calls into fallbacks).
            import re as _re
            m = _re.search(r"retry in ([\d.]+)s", msg)
            wait = float(m.group(1)) + 2 if m else 2**attempt
            if attempt == 5:
                break
            time.sleep(min(wait, 90))
    raise RuntimeError(f"Gemini call failed after 6 attempts: {last}")


# --------------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------------
def complete(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    schema: str = "",
    mode: str | None = None,
) -> LLMResponse:
    """Single entrypoint for every LLM call in the project.

    `schema` is not sent to the model -- it is a version tag folded into the
    cache key, so changing the expected output shape invalidates old entries
    instead of silently reusing them.
    """
    model = model or config.GENERATOR_MODEL
    temperature = config.GEN_TEMPERATURE if temperature is None else temperature
    mode = (mode or config.LLM_MODE).lower()

    key = make_key(prompt, model, temperature, schema)
    cache = get_cache()

    hit = cache.get(key)
    if hit is not None:
        return LLMResponse(text=hit, model=model, cached=True, key=key)

    if mode == "replay":
        raise CacheMiss(
            f"No cached response for key {key} (model={model}, temp={temperature}).\n"
            f"Prompt starts: {prompt[:160]!r}\n"
            "Replay mode never calls the network. Re-run with LLM_MODE=live "
            "(needs GEMINI_API_KEY) to record this call, then commit cache/llm_cache.jsonl."
        )

    backend = _backend or _gemini_backend
    text = backend(prompt, model, temperature)
    cache.put(key, prompt, text, model, temperature)
    return LLMResponse(text=text, model=model, cached=False, key=key)


def complete_json(
    prompt: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    schema: str = "",
    mode: str | None = None,
    default: Any = None,
) -> Any:
    """`complete()` + JSON parsing.

    `default` is returned only on a PARSE failure, never on a cache miss, and
    parse failures are counted so the report can state how often the model
    emitted unusable output instead of hiding it.
    """
    resp = complete(prompt, model=model, temperature=temperature, schema=schema, mode=mode)
    try:
        return extract_json(resp.text)
    except ValueError:
        PARSE_FAILURES.append({"key": resp.key, "text": resp.text[:500]})
        if default is not None:
            return default
        raise


PARSE_FAILURES: list[dict] = []
TRANSPORT_FAILURES: list[dict] = []


def parse_failure_rate(total_calls: int) -> float:
    return len(PARSE_FAILURES) / total_calls if total_calls else 0.0
