"""
Central configuration. Every tunable lives here so the live-interview answer to
"where does X come from?" is always the same file.

Model choice (see DECISIONS.md):
  - Generator = Gemini *flash* tier (cheap, fast, the realistic production choice)
  - Judge     = Gemini *pro*  tier (stronger, different tier)
  Using a different model to judge than to generate is a partial mitigation for
  self-preference bias. It is only PARTIAL because both are Gemini and share a
  training lineage -- we say so explicitly in the report's "what is misleading"
  section, and we validate the judge against human labels (Cohen's kappa).
"""
from __future__ import annotations

import os
from pathlib import Path

import warnings
try:  # optional; the eval path must not hard-require dotenv
    from dotenv import load_dotenv

    _ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
    try:
        _has_pairs = any("=" in ln for ln in _ENV_FILE.read_text(encoding="utf-8").splitlines())
    except OSError:
        _has_pairs = False
    if _has_pairs:
        # Only hand a KEY=value file to dotenv; bare-value files make it warn.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_dotenv()
except Exception:  # pragma: no cover
    pass


def _load_bare_env(path: Path) -> None:
    """Tolerate a .env that holds bare values (one per line, no KEY=).

    Recognised by shape, never by position: an https:// URL is the Qdrant
    Cloud endpoint, an eyJ... token is the Qdrant API key, an AIza... token
    is the Gemini key. Only fills variables that are still unset, so real
    environment variables and KEY=value entries always win.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for raw in lines:
        t = raw.strip()
        if not t or t.startswith("#") or "=" in t:
            continue
        if t.startswith("https://") and not os.getenv("QDRANT_URL"):
            os.environ["QDRANT_URL"] = t
        elif t.startswith("eyJ") and not os.getenv("QDRANT_API_KEY"):
            os.environ["QDRANT_API_KEY"] = t
        elif t.startswith("AIza") and not os.getenv("GEMINI_API_KEY"):
            os.environ["GEMINI_API_KEY"] = t

# --------------------------------------------------------------------------
# Paths -- all relative to the repo root so the repo is portable.
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw"          # gitignored: the ~500MB Kaggle dump
DATA_SAMPLE = ROOT / "data" / "sample"    # committed: the working subsample
DATA_GOLDEN = ROOT / "data" / "golden"    # committed: hand-labelled eval set
CACHE_DIR = ROOT / "cache"                # committed: replayable LLM responses
REPORT_DIR = ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"
OUTPUTS_DIR = ROOT / "outputs"

for _d in (DATA_RAW, DATA_SAMPLE, DATA_GOLDEN, CACHE_DIR, REPORT_DIR, FIGURES_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_load_bare_env(ROOT / ".env")


def _find_raw_csv() -> Path:
    """Locate twcs.csv without forcing the user to move a ~500MB file.

    Kaggle downloads land in different places depending on how they were
    extracted, so we check the usual spots and fall back to the canonical path
    (whose absence produces a clear error message downstream).
    """
    candidates = [
        DATA_RAW / "twcs.csv",
        ROOT / "data" / "twcs" / "twcs.csv",
        ROOT / "data" / "twcs.csv",
        DATA_RAW / "twcs" / "twcs.csv",
    ]
    env = os.getenv("TWCS_PATH")
    if env:
        candidates.insert(0, Path(env))
    for c in candidates:
        if c.exists():
            return c
    return DATA_RAW / "twcs.csv"


RAW_CSV = _find_raw_csv()

# --------------------------------------------------------------------------
# Determinism -- a fixed seed everywhere so results reproduce exactly.
# --------------------------------------------------------------------------
SEED = 42

# --------------------------------------------------------------------------
# Gemini. Names are env-overridable because model IDs churn; pin whatever you
# actually ran in the report so the numbers are attributable.
# --------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gemini-3.1-flash-lite")
# Generator & Judge pinned to gemini-3.1-flash-lite for speed, quota efficiency and reproducibility
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3.1-flash-lite")

GEN_TEMPERATURE = 0.2   # low: we want grounded, repeatable drafts
JUDGE_TEMPERATURE = 0.0  # judging must be as deterministic as the API allows

# --------------------------------------------------------------------------
# LLM call mode.
#   replay : read from cache/, never hit the network. Default, so a reviewer
#            reproduces the headline table with zero API spend.  <-- 15-min promise
#   live   : call Gemini and write new cache entries.
# --------------------------------------------------------------------------
LLM_MODE = os.getenv("LLM_MODE", "replay")

# --------------------------------------------------------------------------
# Retrieval. Embeddings are LOCAL (sentence-transformers) so rebuilding the
# index costs nothing and stays deterministic -- Gemini spend is reserved for
# generation + judging.
# --------------------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBED_DIM = 384
QDRANT_PATH = str(ROOT / "qdrant_storage")  # embedded mode -- no Docker required
QDRANT_COLLECTION = "brand_resolutions"
# Cloud override: when both are set (e.g. via the .env bare lines), the demo
# may use Qdrant Cloud instead. The eval path stays on embedded/local so a
# reviewer reproduces results with no account and no network.
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
RETRIEVAL_TOP_K = 4

# --------------------------------------------------------------------------
# Escalation policy.
# Cost asymmetry: auto-handling something that NEEDED a human is far worse than
# escalating something we could have handled. So we tune for recall on the
# must-escalate class and report the precision we trade away.
# --------------------------------------------------------------------------
SIM_THRESHOLD_ESCALATE = 0.45  # below this max-similarity => no precedent => escalate
LOW_CONFIDENCE_ESCALATE = 0.60  # classifier confidence below this => escalate

# Hard triggers: topics we never auto-handle regardless of retrieval quality.
ESCALATION_KEYWORDS = [
    # financial crime / disputes with the bank
    "chargeback", "fraud", "unauthorized", "stolen card", "identity theft",
    # legal / regulatory
    "lawyer", "legal", "sue", "lawsuit", "ombudsman", "gdpr", "complaint to",
    "small claims", "attorney", "trading standards", "ftc", "wire fraud",
    "class action", "court summons", "legal counsel", "ico",
    # account compromise
    "hacked", "account stolen", "compromised", "stolen account",
    "unauthorized order", "unauthorized charge",
    # human escalation demand
    "real person", "human agent", "speak to a human", "call me back",
    "supervisor", "manager", "executive",
    # broken promise / repeated contact
    "promised me", "agent sarah", "agent told me", "6th time", "fourth time",
    "fifth time", "third time contacting", "keep ignoring",
    # churn / reputational
    "close my account", "boycott",
    # safety / vulnerability
    "emergency", "stranded", "unsafe", "threat", "harass", "discriminat",
    # sarcasm / ghost-delivery marker
    "resident ghost", "ghost signed",
    # ambiguous multi-clause resolution demand
    "do i return or keep", "return or keep",
    # currency / billing mismatch
    "international fee", "charged in euros", "charged in gbp", "charged in dollars",
    "foreign transaction fee",
    # data rights
    "password reset", "locked out",
]

# Ultra-short messages are intrinsically ambiguous and must always escalate.
# A 12-char message like "No tracking???" cannot be reliably handled by auto.
SHORT_TEXT_ESCALATE_CHARS = 25  # threshold (non-whitespace character count)

# PII patterns that must always escalate (regex, case-insensitive)
import re as _re
_PII_PATTERNS = [
    (_re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b", _re.I),          "SSN pattern"),
    (_re.compile(r"\b4[0-9]{3}[\s-]?[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b", _re.I), "credit card (Visa)"),
    (_re.compile(r"\b5[1-5][0-9]{2}[\s-]?[0-9]{4}[\s-]?[0-9]{4}[\s-]?[0-9]{4}\b", _re.I), "credit card (MC)"),
    (_re.compile(r"\bcvv\b|\bcvc\b", _re.I),                            "CVV/CVC"),
    (_re.compile(r"\bpassword\s+is\b", _re.I),                          "password exposed"),
    (_re.compile(r"\b(?:my\s+)?(?:drivers?\s+licen[sc]e|dl)\s+(?:number\s+)?(?:is\s+)?[a-z0-9]{5,}", _re.I), "license number"),
    (_re.compile(r"\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b", _re.I), "email in public"),
    (_re.compile(r"(?:\+?1[-\s.]?)?\(?\d{3}\)?[-\s.]?\d{3}[-\s.]?\d{4}\b", _re.I), "phone number"),
    (_re.compile(r"\bssn\b|\bsocial security\b", _re.I),                "SSN mention"),
    (_re.compile(r"\b(?:exp(?:iry|iration)?|cvv|cvc|zip)\s*\d{2,6}\b", _re.I), "card detail"),
]

# --------------------------------------------------------------------------
# Brand selection (EDA). Phrases that indicate the brand deflected to a private
# channel rather than resolving in-thread. If a brand's public replies are
# mostly these, its "resolutions" are NOT in the dataset and it is a poor
# grounding corpus -- this is the core insight driving brand choice.
# --------------------------------------------------------------------------
# These are MINED, not guessed. The first version of this list was written from
# intuition and under-fired badly -- it scored AmazonHelp at a 96.9% resolution
# rate, which is nonsense. The list below comes from ranking the 3-grams that
# immediately precede a link across 270k real brand replies, then hand-reading
# the top ~60. Counts in comments are from that scan. See DECISIONS.md #22.
DEFLECTION_PATTERNS = [
    # -- explicit private channel
    r"\bd\.?m\.?s?\b", r"direct message", r"private(ly| message| secured)",
    r"\bpm us\b", r"inbox us", r"slide into",
    r"(follow|following) (us )?(back|and|so) (we|i|our team) can",
    r"we'?(ll| will) (continue|follow up|take (this|it)) (there|here|privately|offline|in dm)",
    # -- "leave your details somewhere else"  (~1,400 hits)
    r"(your|the) details (here|below|via|at|on|through)",
    r"(drop|send|share|provide|leave|enter|fill) (in |us )?(your|the|these) (details|info)",
    # -- "drop us a note <link>"  (~3,280 hits, Apple's house style)
    r"(drop|send|shoot|leave|write) (us )?a (note|message|line|mail)",
    # -- "contact our support team here <link>"  (~1,140 hits)
    r"(support|service|care) team (here|at|via|on|through|for)",
    r"(contact|reach|message|write|speak) (out )?(to )?(us|our team|them)\s*(here|via|at|on|through|directly)?",
    r"\breach (us|out) (here|via|at|on)",
    r"get in touch",
    # -- other channels
    r"(email|e-mail|call|phone|text|chat|whatsapp) (us|our|with us|to us)",
    r"give us a (call|ring|shout)",
    r"\b1[\s\-\.]?8\d\d[\s\-\.]?\d{3}[\s\-\.]?\d{4}\b",   # US support numbers
    r"(chat|talk) (with (us|an? (agent|advisor|expert)))? ?(here|via|at|online)",
    r"live chat", r"our (website|site|app|help ?(centre|center))",
    # -- ticketing / forms / async handoff
    r"fill (out |in )?(this|the|our) (form|survey)",
    r"submit a (ticket|request|case|claim|form)",
    r"(you'?ll|you will|we'?ll) (receive|send you) an? (email|e-mail|call|dm)",
    r"(raise|open|log) a (ticket|case|complaint)",
    r"preferred language (here|below)",     # Amazon's language-routing punt
    # -- holding replies: no resolution, just an acknowledgement of a handoff
    r"(we'?(ve| have)|i'?(ve| have)) (passed|forwarded|escalated|shared) (this|it|your)",
    r"(someone|a member|our team|an agent) will (be in touch|contact|reach|get back)",
    # "we'll reach out to YOU" -- the punt points the other way, but it is still
    # a punt: no answer now, contact deferred to a private channel later.
    r"(reach|get) (out |back )?to you",
    r"will (update|contact|call|email|revert to) you",
    r"look(ing)? into (this|it) and (get|come) back",
]

# NEGATIVE evidence. A link is not automatically a deflection -- pointing a
# customer at the fix ("the steps are here: <link>") IS a resolution, and one we
# want in the grounding corpus. Mining found these at real volume ("the steps
# here" 442, "more info here" 307, "be found here" 251), so treating every
# linked reply as a punt would have thrown away genuine self-service answers.
# These override a deflection match.
SELF_SERVICE_PATTERNS = [
    r"(the |these |following )?steps (are )?(here|below|outlined|at)",
    r"more info(rmation)? (here|below|at|on)",
    r"(can |may )?be found (here|below|at)",
    r"(have|take) a look at (this|the|our) (article|guide|page|doc)",
    r"this (article|guide|page|help page|support page|walkthrough)",
    r"(instructions|troubleshoot(ing)?|how[- ]to) (steps |guide |article )?(here|below|at)",
    r"(here'?s|here is) (a|the|our) (guide|article|link to)",
]

MIN_THREAD_LEN = 2       # need at least customer msg + brand reply
MAX_TEXT_LEN = 400       # twitter-ish; guards against junk rows

# --------------------------------------------------------------------------
# Subsampling -- reviewers explicitly said they will NOT run the full dataset.
# --------------------------------------------------------------------------
SAMPLE_PAIRS_PER_BRAND = 6000   # (customer -> brand reply) pairs kept per brand
EDA_SCAN_ROWS = 1_200_000       # rows scanned for brand selection (chunked)

# --------------------------------------------------------------------------
# Intent taxonomy. FROZEN 2026-09-10 from AmazonHelp k=6 English-only clustering
# (4,860 msgs, sentence-transformers 384d). Draft: data/sample/taxonomy_draft.json.
# Cluster -> intent mapping: delay/Prime 28% => delivery_delay;
# marked-delivered/tracking 25% => missing_parcel_tracking; refund/return 26% =>
# refund_return; Alexa/Echo/app/account-linking + "call me" 11% =>
# device_app_account; generic order-status 6% => order_status_general;
# 3% URL-noise cluster deleted. Non-English clusters excluded by construction
# (English filter before clustering). `other_unclear` is a rejection bucket --
# it routes to escalate, scored as detection recall, not a 6th F1 class.
# --------------------------------------------------------------------------
INTENTS: dict[str, str] = {
    "delivery_delay": "Order is late, delayed past the promised/Prime date, or delivery rescheduled. No claim it arrived.",
    "missing_parcel_tracking": "Marked delivered / handed to customer but not received; tracking stuck, lost parcel, carrier investigation needed.",
    "refund_return": "Refund, return, money-back, wrong item, double charge, compensation for a failed order.",
    "device_app_account": "Alexa/Echo/Fire/app/device setup, account linking, login, or 'call me' support-contact requests tied to a device/account issue.",
    "order_status_general": "Generic where-is-my-order / check-status with an order number and no specific delay, loss, or refund claim yet.",
    "other_unclear": "Does not fit a defined intent, non-English slip-through, or too vague to act on. Always escalates.",
}

ESCALATE_LABELS = ("escalate", "auto_handle")
