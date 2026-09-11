"""
Brand selection -- the analysis that opens the report.

THE INSIGHT
-----------
Most big brands on this dataset do not resolve anything publicly. They reply
"Sorry to hear that! Please DM us." and the actual resolution happens in a
private channel that is NOT in the data. If we pick such a brand, our retrieval
corpus is a pile of deflections, and a "grounded" reply generator will faithfully
learn to say... "please DM us". The system would score well on groundedness
while being commercially worthless.

So we do not pick the most famous brand. We measure, per brand:
    volume            -- enough pairs to retrieve from at all
    deflection_rate   -- share of public replies that punt to DM/email/phone
    resolution_rate   -- 1 - deflection_rate
    groundable_pairs  -- volume * resolution_rate  (the number that matters)
    specificity       -- do replies contain concrete instructions/entities?
    gratitude_rate    -- customer follow-ups saying thanks (weak "it worked")

and pick on `groundable_pairs` + specificity. This is a subsample-based
estimate, so it is directional, not exact -- stated as such in the report.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import pandas as pd

from src import config
from src.ingest.language import is_english

# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------
_DEFLECT_RE = re.compile("|".join(config.DEFLECTION_PATTERNS), re.I)
_SELF_SERVICE_RE = re.compile("|".join(config.SELF_SERVICE_PATTERNS), re.I)

# Concrete-instruction markers: steps, settings paths, versions, timeframes.
# A reply containing these is far more likely to encode a real resolution.
_SPECIFIC_RE = re.compile(
    r"(settings?\s*>|\bmenu\b|\btap\b|\bclick\b|\bselect\b|\btoggle\b|"
    r"\bv?\d+\.\d+|\breset\b|\brestart\b|\breinstall\b|\bupdate\b|"
    r"\b\d+\s*(-|to)\s*\d+\s*(business\s*)?(day|hour|week)|"
    r"\brefund(ed)?\b|\breship\b|\breplacement\b|\bcredited?\b)",
    re.I,
)

_GRATITUDE_RE = re.compile(
    r"\b(thank|thanks|thx|ty|cheers|appreciate|sorted|fixed it|that worked|"
    r"worked|perfect|legend|awesome)\b",
    re.I,
)

_STILL_BROKEN_RE = re.compile(
    r"\b(still (not|isn'?t|doesn'?t|no)|not fixed|didn'?t work|doesn'?t work|"
    r"same (problem|issue)|useless|nothing happened)\b",
    re.I,
)


def _as_text(text: object) -> str:
    """Coerce to str. Old cached CSVs can carry NaN floats; `x or ''` does not
    catch NaN (it is truthy), so guard by type explicitly."""
    return text if isinstance(text, str) else ""


def is_self_service(text: str) -> bool:
    """True if a link points at the FIX rather than at a different channel.

    "The steps are here: <URL>" is a resolution and belongs in the grounding
    corpus. "Contact our support team here: <URL>" is a handoff and does not.
    Both contain a link, so link-presence alone cannot separate them.
    """
    return bool(_SELF_SERVICE_RE.search(_as_text(text)))


def is_deflection(text: str) -> bool:
    """True if the brand punted to a private channel instead of resolving.

    Self-service beats deflection: a reply that both names a channel and points
    at documentation is still giving the customer the answer.
    """
    t = _as_text(text)
    if is_self_service(t):
        return False
    return bool(_DEFLECT_RE.search(t))


def is_specific(text: str) -> bool:
    return bool(_SPECIFIC_RE.search(_as_text(text)))


def is_grateful(text: str) -> bool:
    return bool(_GRATITUDE_RE.search(_as_text(text)))


def is_still_broken(text: str) -> bool:
    return bool(_STILL_BROKEN_RE.search(_as_text(text)))


# --------------------------------------------------------------------------
# Per-brand profile
# --------------------------------------------------------------------------
@dataclass
class BrandProfile:
    brand: str
    n_pairs: int
    n_customers: int
    english_rate: float
    deflection_rate: float
    resolution_rate: float
    resolution_ci_lo: float
    resolution_ci_hi: float
    groundable_pairs: float
    specificity_rate: float
    mean_reply_len: float
    followup_rate: float
    gratitude_rate: float
    still_broken_rate: float
    gratitude_lift: float = float("nan")
    n_rate_sample: int = 0
    score: float = 0.0


def _rate_ci(flags, seed: int = config.SEED, n_boot: int = 400):
    """95% bootstrap interval for a rate. Makes sampling error visible."""
    import numpy as np

    a = np.asarray(flags, dtype=float)
    if a.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, a.size, size=(n_boot, a.size))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def profile_brands(pairs: pd.DataFrame, min_pairs: int = 50,
                   rate_sample: int = 3000, seed: int = config.SEED) -> pd.DataFrame:
    """Compute the selection table. One row per brand, ranked by score.

    Two corrections relative to the first version, both found by disbelieving
    the output rather than by a test failing:

    1. Deflection is scored ONLY over English pairs. Every deflection pattern is
       an English regex, so a Spanish "please DM us" matches nothing and would
       be counted as a resolution. That bias is not uniform across brands --
       AmazonHelp is ~19% non-English, AppleSupport ~2% -- so it distorted the
       ranking, not just the absolute numbers.

    2. `groundable_pairs` now means what its name claims: pairs that are English
       AND not a deflection, i.e. material we could actually retrieve from.
       Multilingual brands are therefore discounted by construction rather than
       by an ad-hoc penalty term.

    RATES ARE ESTIMATED FROM A CAPPED SAMPLE (`rate_sample` pairs per brand).
    Brand selection is a ranking over rates, and a rate estimated from 3,000
    draws has a standard error under 1 point -- far smaller than the gaps that
    decide the ranking. `n_pairs` remains the true population count; only the
    rates are sampled. `resolution_ci` carries the 95% bootstrap interval so the
    precision of the estimate is visible rather than implied.
    """
    if pairs.empty:
        return pd.DataFrame()

    rows: list[BrandProfile] = []
    for brand, g_all in pairs.groupby("brand"):
        n = len(g_all)
        if n < min_pairs:
            continue
        # Sample for RATE estimation; n_pairs keeps the true population size.
        g = g_all.sample(rate_sample, random_state=seed) if n > rate_sample else g_all
        g = g.copy()

        # A pair is usable only if BOTH sides are English: we classify the
        # customer message and we ground on the brand reply.
        g["_english"] = g["brand_reply"].map(is_english) & g["customer_message"].map(is_english)
        en = g[g["_english"]]
        english_rate = float(g["_english"].mean())
        if en.empty:
            continue

        en = en.copy()
        en["_deflect"] = en["brand_reply"].map(is_deflection)
        en["_specific"] = en["brand_reply"].map(is_specific)
        fu_text = en["customer_followup"].fillna("")
        en["_has_followup"] = fu_text.str.len() > 0
        en["_grateful"] = fu_text.map(is_grateful)

        deflect = float(en["_deflect"].mean())
        resolution = 1.0 - deflect
        lo, hi = _rate_ci(en["_deflect"].to_numpy(), seed=seed)
        n_fu = int(en["_has_followup"].sum())
        fu = en[en["_has_followup"]]

        # Behavioural check on the detector itself: if `_deflect` tracks reality,
        # customers should thank RESOLUTIONS more than they thank deflections.
        # A lift near zero means the regex is not measuring what it claims to.
        lift = float("nan")
        if n_fu >= 30:
            g_res = fu.loc[~fu["_deflect"], "_grateful"]
            g_def = fu.loc[fu["_deflect"], "_grateful"]
            if len(g_res) >= 10 and len(g_def) >= 10:
                lift = float(g_res.mean() - g_def.mean())

        rows.append(
            BrandProfile(
                brand=str(brand),
                n_pairs=n,
                n_customers=int(g_all["customer_id"].nunique()),
                english_rate=english_rate,
                deflection_rate=deflect,
                resolution_rate=resolution,
                resolution_ci_lo=1.0 - hi,
                resolution_ci_hi=1.0 - lo,
                groundable_pairs=n * english_rate * resolution,
                specificity_rate=float(en["_specific"].mean()),
                mean_reply_len=float(en["brand_reply"].str.len().mean()),
                followup_rate=n_fu / len(en) if len(en) else 0.0,
                gratitude_rate=float(fu["_grateful"].mean()) if n_fu else 0.0,
                still_broken_rate=float(fu["customer_followup"].fillna("").map(is_still_broken).mean())
                if n_fu else 0.0,
                gratitude_lift=lift,
                n_rate_sample=len(g),
            )
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame([asdict(r) for r in rows])

    # Score = how much *usable grounding material* a brand gives us.
    # log-scale the volume so a mega-brand with 90% deflections cannot win on
    # size alone; specificity is the tie-breaker between similar brands.
    import numpy as np

    vol = np.log10(out["groundable_pairs"].clip(lower=1))
    vol_n = (vol - vol.min()) / (vol.max() - vol.min()) if vol.max() > vol.min() else vol * 0
    out["score"] = (
        0.45 * out["resolution_rate"]
        + 0.35 * vol_n
        + 0.20 * out["specificity_rate"]
    )
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def to_markdown(profiles: pd.DataFrame, top_n: int = 12) -> str:
    cols = [
        "brand", "n_pairs", "english_rate", "resolution_rate", "groundable_pairs",
        "specificity_rate", "gratitude_lift", "score",
    ]
    p = profiles.head(top_n)[cols].copy()
    for c in ["english_rate", "resolution_rate", "specificity_rate", "score"]:
        p[c] = p[c].map(lambda v: f"{v:.3f}")
    p["gratitude_lift"] = p["gratitude_lift"].map(lambda v: f"{v:+.3f}")
    p["groundable_pairs"] = p["groundable_pairs"].map(lambda v: f"{v:,.0f}")
    p["n_pairs"] = p["n_pairs"].map(lambda v: f"{v:,}")
    hdr = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join(str(v) for v in row) + " |\n" for row in p.values)
    return hdr + body


def plot_selection(profiles: pd.DataFrame, path=None, top_n: int = 15):
    """Scatter: volume vs resolution rate. The chart that justifies the pick."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = profiles.head(top_n)
    path = path or (config.FIGURES_DIR / "brand_selection.png")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(p["n_pairs"], p["resolution_rate"], s=p["specificity_rate"] * 600 + 40,
               alpha=0.65, edgecolor="black", linewidth=0.5)
    for _, r in p.iterrows():
        ax.annotate(r["brand"], (r["n_pairs"], r["resolution_rate"]),
                    fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("pairs in subsample (log scale)  →  volume")
    ax.set_ylabel("resolution rate  (1 − deflection rate)")
    ax.set_title("Brand selection: volume is not enough — we need in-thread resolutions\n"
                 "(bubble size = specificity of replies)", fontsize=11)
    ax.axhline(0.5, ls="--", c="grey", lw=0.8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
