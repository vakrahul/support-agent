"""
English detection, deliberately dependency-free.

WHY THIS EXISTS
---------------
It was not in the original plan. It was forced by the data: sampling AmazonHelp
replies turned up Portuguese ("Vai amar com certeza!"), Spanish ("Por favor
contáctanos") and French ("Nous faisons appel à votre patience"). Because every
deflection pattern is an English regex, a Spanish "please DM us" matched nothing
and was scored as a RESOLUTION. That is a silent, one-directional bias: it
inflates the resolution rate of exactly the brands with the most foreign-language
traffic. Measured over a 400-reply sample per brand, AmazonHelp is 21% non-English
against 1.8% for AppleSupport -- so the bug did not distort all brands equally,
it distorted the ranking itself.

`langdetect`/`fasttext` would be the obvious tools. Neither is installable in
this environment (no network), and both would add a dependency to the path that
has to run in under 15 minutes. The heuristic below is weaker but sufficient for
the job it does here, which is a corpus filter, not a linguistic claim.

WHY THIS IS A REJECT-FILTER, NOT AN ACCEPT-FILTER
-------------------------------------------------
The first version demanded that text PROVE it was English by containing enough
English function words. On a corpus that is ~90% English that is the wrong
default, and it showed immediately: "app keeps crashing every time i open the
payments tab" scored 0.11 and was rejected. Terse customer complaints are
content-word dense and function-word sparse, which describes most of this
dataset -- so an accept-filter deletes exactly the messages we most need.

It is inverted here. Text is English unless there is positive evidence of
another language: a non-Latin script, or a meaningful share of tokens that are
distinctively foreign function words. Foreign function words are as cheap to
detect as English ones and, unlike content words, they cannot be confused for
product names or brand jargon.

WHAT IT WILL GET WRONG (stated up front, and pinned by tests)
  - A foreign-language reply written without its own function words (very short
    ones, mostly) passes. The filter fails toward KEEPING data, which for a
    corpus filter is the cheaper error: a stray foreign row adds noise, whereas
    a deleted English row is invisible loss.
  - Code-switched text ("sorry por el problema") is judged by whichever side
    supplies more function words.
  - It cannot distinguish Spanish from Portuguese, and does not try to.
"""
from __future__ import annotations

import re

# Function words, not content words. Content words vary by brand and topic;
# function words are what make a sentence English regardless of subject.
_EN_FUNCTION_WORDS = frozenset("""
a an the and or but if then than that this these those there here
i me my we us our you your he she it they them their his her its
is am are was were be been being do does did done have has had
can could will would shall should may might must
to for with from into onto about of on in at by as so no not
what when where which who whom how why
please sorry thanks thank hi hello hey ok okay yes yeah
get got give sent send let know need want help see look take make
""".split())

# The mirror image: function words that are distinctively NOT English.
# Deliberately excludes tokens that collide with English ("a", "no", "son",
# "die", "in", "an", "we") -- a collision would make English text look foreign.
_FOREIGN_FUNCTION_WORDS = frozenset("""
el la los las un una unos unas del al este esta esto ese esa
que qué para por como con sin sobre pero porque cuando donde
su sus nuestro nuestra tu tus mi mis yo tú usted ustedes ellos
es son está están ser estar tiene tienen hacer puede pueden
gracias hola buenos buenas favor lamentamos disculpa disculpas ayudarte
os um uma uns umas dos das nao não você vocês seu sua
obrigado obrigada desculpe pode podem fazer com aqui qualquer coisa
muito muita mais bem bem já ainda todo toda todos todas algo nada
sempre também então agora depois antes cada outro outra mesmo tudo
muy bien ya aún algún alguna siempre también entonces ahora después
mismo cualquier quien quién cuanto cuánto adónde pues
je tu il elle nous vous ils elles le les des une dans pour avec
est sont être avoir votre vos notre nos ce cette ces qui quoi
merci bonjour désolé désolée excusons pouvez pouvons ici très déjà
encore tout toute toutes rien toujours aussi alors maintenant après
avant chaque autre même quel quelle combien où parce
der das und ist sind sein haben wir ihr ihre ihren mit für
nicht auch noch aber oder wenn dann bitte danke entschuldigung
hier sehr mehr schon alles nichts immer jetzt jede andere welche weil
lo gli una che non per sono siamo grazie ciao scusa
het van zij wij niet maar ook wel voor
""".split())

# Scripts that are simply not English. Cheap, and near-zero false positives.
_NON_LATIN = re.compile(
    r"[Ѐ-ӿ"   # Cyrillic
    r"؀-ۿ"    # Arabic
    r"ऀ-ॿ"    # Devanagari
    r"぀-ヿ"    # Hiragana / Katakana
    r"一-鿿"    # CJK
    r"가-힯"    # Hangul
    r"฀-๿"    # Thai
    r"֐-׿]"   # Hebrew
)

# Diacritics common in Romance languages and rare in English prose.
_ROMANCE_DIACRITIC = re.compile(r"[áàâãäéèêëíìîïóòôõöúùûüñçÁÀÂÃÉÈÊÍÎÓÔÕÚÜÑÇ]")

_WORD = re.compile(r"[a-zA-ZáàâãäéèêëíìîïóòôõöúùûüñçÁÀÂÃÉÈÊÍÎÓÔÕÚÜÑÇ']+")

MIN_TOKENS = 4           # below this, only script/diacritics are admissible
FOREIGN_THRESHOLD = 0.18  # share of tokens that are foreign function words
NON_LATIN_SHARE = 0.10    # share of characters in a non-Latin script


def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall(str(text or "").lower()) if len(t) > 1]


def english_score(text: str) -> float:
    """Share of word tokens that are English function words, in [0, 1]."""
    toks = _tokens(text)
    return sum(t in _EN_FUNCTION_WORDS for t in toks) / len(toks) if toks else 0.0


def foreign_score(text: str) -> float:
    """Share of word tokens that are distinctively non-English function words."""
    toks = _tokens(text)
    return sum(t in _FOREIGN_FUNCTION_WORDS for t in toks) / len(toks) if toks else 0.0


def is_english(text: str) -> bool:
    """Best-effort English check, biased toward KEEPING text.

    Order matters: script first (near-certain, and works on text too short to
    score), then foreign function words, then diacritics as a tie-breaker.
    """
    s = str(text or "")
    if not s.strip():
        return False

    letters = [c for c in s if c.isalpha()]
    if letters and sum(bool(_NON_LATIN.match(c)) for c in letters) / len(letters) > NON_LATIN_SHARE:
        return False

    toks = _tokens(s)
    en, fg = english_score(s), foreign_score(s)

    if len(toks) >= MIN_TOKENS and fg >= FOREIGN_THRESHOLD and fg > en:
        return False

    # Diacritics are admissible even when there is little other evidence --
    # "Gracias señor" is short but unambiguous. Require the English signal to be
    # weak too, so "café" in an English sentence does not trip it.
    if _ROMANCE_DIACRITIC.search(s) and en < 0.15 and len(toks) >= 2:
        return False

    return True
