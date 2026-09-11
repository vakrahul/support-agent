"""Tests for the language gate.

This module exists because of a measured bug, so the tests are written against
the actual strings that caused it -- real replies pulled from the corpus, not
invented examples.
"""
from __future__ import annotations

from src.ingest.language import english_score, foreign_score, is_english

# Pulled verbatim from twcs.csv brand replies (URLs and handles removed).
REAL_ENGLISH = [
    "I'm sorry for the trouble with your order. Can you fill out this form so one of my team can take a look?",
    "Sorry to hear that! Please DM us your account details and we'll take a look.",
    "That's not the experience we want you to have. We'd like to look into this further.",
    "Thanks for reaching out - the steps to reset your password are here.",
    "Oh no! Sorry about that.",
    "Apologies for the delay.",
]

REAL_NON_ENGLISH = [
    "Vai amar com certeza! Qualquer coisa e so chamar por aqui.",
    "Por favor contactanos por mensaje directo para poder ayudarte.",
    "Nous faisons appel a votre patience, notre equipe revient vers vous.",
    "Lamentamos o transtorno. Envie seus dados por mensagem privada.",
    "Hallo, bitte senden Sie uns eine private Nachricht mit Ihren Daten.",
]


def test_real_english_replies_pass():
    for t in REAL_ENGLISH:
        assert is_english(t), f"false negative: {t!r} (score={english_score(t):.2f})"


def test_real_non_english_replies_fail():
    for t in REAL_NON_ENGLISH:
        assert not is_english(t), f"false positive: {t!r} (score={english_score(t):.2f})"


def test_non_latin_scripts_rejected():
    for t in ["ご迷惑をおかけして申し訳ありません", "Приносим извинения за неудобства",
              "نعتذر عن الإزعاج", "불편을 드려 죄송합니다"]:
        assert not is_english(t)


def test_terse_customer_complaints_are_kept():
    """The false negative that forced the accept-filter -> reject-filter rewrite.

    "app keeps crashing every time i open the payments tab" contains exactly one
    English function word ("the") in nine tokens. Under the original
    prove-it-is-English design it scored 0.11 and was deleted. Terse,
    content-word-dense complaints are the bulk of the customer side of this
    corpus, so that design deleted the data the project runs on.
    """
    for t in [
        "app keeps crashing every time i open the payments tab",
        "order still not delivered after two weeks",
        "charged twice this month subscription refund please",
        "flight delayed again no compensation offered",
    ]:
        assert is_english(t), f"deleted real English: {t!r} (en={english_score(t):.2f})"


def test_foreign_score_separates_the_two_populations():
    """The filter is only meaningful if the score gap is wide, not marginal."""
    en = [foreign_score(t) for t in REAL_ENGLISH]
    fg = [foreign_score(t) for t in REAL_NON_ENGLISH]
    assert max(en) < min(fg), f"populations overlap: english≤{max(en):.2f} foreign≥{min(fg):.2f}"


def test_english_text_containing_a_loanword_survives():
    # "café"/"naïve" carry diacritics but the sentence is plainly English.
    assert is_english("We can refund the café order for you if you confirm the date.")


def test_short_text_fails_toward_keeping_data():
    """Below the evidence threshold we keep the row rather than guess.

    Documented as a known limitation: short foreign replies survive the filter.
    The alternative -- guessing -- would silently delete English data, which is
    the worse failure for a corpus filter.
    """
    assert is_english("Fixed now!")
    assert is_english("On it.")


def test_short_text_with_romance_diacritics_still_rejected():
    # Diacritics are admissible evidence even when the text is short.
    assert not is_english("Gracias señor")
    assert not is_english("Não é possível")


def test_emoji_and_punctuation_do_not_break_it():
    assert is_english("Sorry about that! 😞 We can help you fix it right away.")


def test_empty_and_none_safe():
    assert not is_english("") and not is_english(None) and not is_english("   ")
    assert english_score(None) == 0.0


def test_url_token_does_not_count_as_a_word():
    a = "Please contact our support team here <URL>"
    assert is_english(a)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} language tests passed")
