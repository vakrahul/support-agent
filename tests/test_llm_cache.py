"""Tests for the cached LLM client.

These exercise the full call path with an injected fake backend -- no network,
no API key. The cache is the mechanism the "<15 minute reproduction" promise
rests on, so its failure modes are pinned hard, especially:
  * replay mode must NEVER silently invent a response on a miss
  * changing the prompt/model/temp/schema must invalidate the key
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from src.llm.gemini import (
    CacheMiss,
    LLMCache,
    complete,
    complete_json,
    extract_json,
    make_key,
    reset_cache,
    set_backend,
)


def _tmp_cache() -> Path:
    d = Path(tempfile.mkdtemp())
    p = d / "llm_cache.jsonl"
    reset_cache(p)
    return p


# ---------------------------------------------------------------- JSON parsing
def test_extract_plain_json():
    assert extract_json('{"intent": "billing"}') == {"intent": "billing"}


def test_extract_from_markdown_fence():
    raw = 'Here you go:\n```json\n{"intent": "billing", "confidence": 0.9}\n```\nHope that helps!'
    assert extract_json(raw)["intent"] == "billing"


def test_extract_with_prose_around_it():
    raw = 'Sure! The classification is {"intent": "delivery"} based on the text.'
    assert extract_json(raw) == {"intent": "delivery"}


def test_extract_tolerates_trailing_comma():
    assert extract_json('{"a": 1, "b": 2,}') == {"a": 1, "b": 2}


def test_extract_handles_braces_inside_strings():
    # a naive brace-counter breaks on this
    raw = '{"reply": "use the {order_id} placeholder", "ok": true}'
    assert extract_json(raw)["reply"] == "use the {order_id} placeholder"


def test_extract_array():
    assert extract_json('```\n[{"a":1},{"a":2}]\n```') == [{"a": 1}, {"a": 2}]


def test_extract_raises_on_garbage():
    try:
        extract_json("I'm sorry, I can't help with that.")
        raise AssertionError("should have raised")
    except ValueError:
        pass


# ---------------------------------------------------------------- cache keys
def test_key_is_stable_and_sensitive():
    base = make_key("hello", "gemini-2.0-flash", 0.2, "v1")
    assert base == make_key("hello", "gemini-2.0-flash", 0.2, "v1")   # stable
    assert base != make_key("hello!", "gemini-2.0-flash", 0.2, "v1")  # prompt
    assert base != make_key("hello", "gemini-2.5-pro", 0.2, "v1")     # model
    assert base != make_key("hello", "gemini-2.0-flash", 0.0, "v1")   # temperature
    assert base != make_key("hello", "gemini-2.0-flash", 0.2, "v2")   # schema version


# ---------------------------------------------------------------- replay/live
def test_replay_miss_raises_not_silently_defaults():
    # THE critical guarantee: a missing cache entry must never become a fake number.
    _tmp_cache()
    try:
        complete("never recorded", mode="replay")
        raise AssertionError("replay mode must raise on a cache miss")
    except CacheMiss as e:
        assert "LLM_MODE=live" in str(e)


def test_live_records_then_replay_serves():
    _tmp_cache()
    calls = []

    def fake(prompt, model, temperature):
        calls.append(prompt)
        return '{"intent": "billing"}'

    set_backend(fake)
    try:
        r1 = complete("classify this", mode="live")
        assert r1.cached is False and len(calls) == 1

        # same prompt in replay mode: served from cache, backend not touched
        r2 = complete("classify this", mode="replay")
        assert r2.cached is True and r2.text == r1.text
        assert len(calls) == 1
    finally:
        set_backend(None)


def test_cache_survives_reload_from_disk():
    path = _tmp_cache()
    set_backend(lambda p, m, t: "recorded")
    try:
        complete("persist me", mode="live")
    finally:
        set_backend(None)

    reset_cache(path)  # simulate a fresh process
    assert complete("persist me", mode="replay").text == "recorded"


def test_cache_file_is_readable_jsonl():
    # a reviewer should be able to read exactly what was asked and answered
    path = _tmp_cache()
    set_backend(lambda p, m, t: "the answer")
    try:
        complete("the question", mode="live")
    finally:
        set_backend(None)
    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["prompt"] == "the question" and rec["response"] == "the answer"


def test_corrupt_final_line_is_tolerated():
    path = _tmp_cache()
    good = {"key": "k1", "model": "m", "temperature": 0.0,
            "prompt": "p", "response": "r", "ts": 0}
    path.write_text(json.dumps(good) + "\n{ truncated...", encoding="utf-8")
    c = LLMCache(path)
    assert len(c) == 1 and c.get("k1") == "r"


def test_complete_json_roundtrip():
    _tmp_cache()
    set_backend(lambda p, m, t: '```json\n{"intent":"delivery","confidence":0.81}\n```')
    try:
        out = complete_json("x", mode="live")
    finally:
        set_backend(None)
    assert out["intent"] == "delivery" and out["confidence"] == 0.81


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} llm-cache tests passed")
