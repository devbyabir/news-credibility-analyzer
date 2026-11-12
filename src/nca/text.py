"""Text normalisation and tokenisation for Bangla news.

Design notes
- scikit-learn's default token pattern (``\\b\\w\\w+\\b``) splits Bangla words at
  every vowel sign / virama, because Python does not treat combining marks as
  word characters ("বিশ্ববিদ্যালয়ে" -> "বব", "লয"). We use an explicit
  Bengali-block tokenizer instead.
- Formatting artifacts that differ between the dataset's sources are
  neutralised so the model cannot learn typography instead of content
  (measured in docs/DATA.md): zero-width joiners/non-joiners and soft hyphens
  are removed, NBSP and other whitespace collapse to one space, ASCII digits are
  mapped to Bangla digits, and sentence punctuation (। ॥ ! ? , ; :) is spaced
  out so "।এ" (no space after a danda) and "। এ" look the same.
- No stemming: the legacy rule-based stemmer was noisy and slow, and n-gram
  TF-IDF does not need it.
"""

from __future__ import annotations

import re
import unicodedata

BENGALI_BLOCK = "\u0980-\u09ff"

_INVISIBLE = re.compile(r"[\u00ad\u200b\u200c\u200d\u2060\ufeff]")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"\s*([\u0964\u0965!?,;:])\s*")
_DIGITS = str.maketrans("0123456789", "\u09e6\u09e7\u09e8\u09e9\u09ea\u09eb\u09ec\u09ed\u09ee\u09ef")
_TOKEN = re.compile(rf"[{BENGALI_BLOCK}a-z0-9]+")
_BENGALI_LETTER = re.compile(rf"[{BENGALI_BLOCK}]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_KEY_STRIP = re.compile(rf"[^{BENGALI_BLOCK}a-z0-9]+")
_DIGIT_RUN = re.compile(r"[0-9\u09e6-\u09ef]+")
_SLASH_TOKEN = re.compile(r"\S*/\S*")
_BRACKETS = re.compile(r"[()\[\]{}]")
# "korona", "kovid" (COVID-era words appear only in fake-labelled articles)
_COVID = re.compile(
    r"(?:\u0995\u09b0\u09cb\u09a8\u09be|\u0995\u09cb\u09ad\u09bf\u09a1)\S*|covid\S*|corona\S*"
)
# tokens containing "bidi" (BD), e.g. "bidi-protidin" bylines and "bdnews24"-style sources
_BD_BYLINE = re.compile(r"\S*\u09ac\u09bf\u09a1\u09bf\S*")
# Gregorian month names (Bangla). Authentic articles cluster in September/October
# (their collection window), so months act as a collection-period cue.
_MONTHS = re.compile(
    r"(?<!\S)(?:\u09ab\u09c7\u09ac\u09cd\u09b0\u09c1\u09af\u09bc\u09be\u09b0\u09bf|\u09b8\u09c7\u09aa\u09cd\u099f\u09c7\u09ae\u09cd\u09ac\u09b0|\u099c\u09be\u09a8\u09c1\u09af\u09bc\u09be\u09b0\u09bf|\u09a1\u09bf\u09b8\u09c7\u09ae\u09cd\u09ac\u09b0|\u0985\u0995\u09cd\u099f\u09cb\u09ac\u09b0|\u09a8\u09ad\u09c7\u09ae\u09cd\u09ac\u09b0|\u098f\u09aa\u09cd\u09b0\u09bf\u09b2|\u09ae\u09be\u09b0\u09cd\u099a|\u099c\u09c1\u09b2\u09be\u0987|\u0986\u0997\u09b8\u09cd\u099f|\u0986\u0997\u09b7\u09cd\u099f|\u099c\u09c1\u09a8|\u09ae\u09c7)(?:\u09c7\u09b0|\u09b0|\u09c7)?(?=\s|$)"
)  # optional -er/-r/-e endings
# mis-encoded "motinidhi" -> "protinidhi" (correspondent); occurs only in some fake-labelled sources
_GARBLED = (
    "\u09ae\u09a4\u09bf\u09a8\u09bf\u09a7\u09bf",
    "\u09aa\u09cd\u09b0\u09a4\u09bf\u09a8\u09bf\u09a7\u09bf",
)


def normalize_text(text: object) -> str:
    """Unicode-normalise and clean a raw news string (idempotent)."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = _INVISIBLE.sub("", s)
    s = _URL.sub(" ", s)
    s = _EMAIL.sub(" ", s)
    s = s.lower().translate(_DIGITS)
    s = _PUNCT.sub(r" \1 ", s)
    return _WS.sub(" ", s).strip()


def normalize_robust(text: object) -> str:
    """``normalize_text`` plus masking of dataset-specific shortcut cues.

    BanFakeNews-2.0's authentic and fake articles come from different periods
    and portals, so dates, month names, COVID-era words, byline fragments (``x/y``, ``bd...``
    sources) and a mis-encoded word separate the classes without saying anything
    about credibility (see docs/DATA.md, "Shortcut cues"). This variant removes
    them so the model must rely more on the article body. It cannot remove every
    source signal.
    """
    s = normalize_text(text).replace(*_GARBLED)
    s = _SLASH_TOKEN.sub(" ", s)
    s = _BRACKETS.sub(" ", s)
    s = _COVID.sub(" ", s)
    s = _BD_BYLINE.sub(" ", s)
    s = _DIGIT_RUN.sub(" ", s)
    s = _MONTHS.sub(" ", s)
    return _WS.sub(" ", s).strip()


PREPROCESSORS = {"standard": normalize_text, "robust": normalize_robust}


def tokenize(text: str) -> list[str]:
    """Split normalised text into Bangla/Latin/digit tokens (punctuation dropped)."""
    return _TOKEN.findall(text)


def dedup_key(text: object) -> str:
    """Aggressive key for exact-duplicate detection (ignores spacing/punctuation)."""
    return _KEY_STRIP.sub("", normalize_text(text))


def bangla_ratio(text: str) -> float:
    """Share of Bengali-script letters among Bengali + Latin letters (0..1)."""
    bn = len(_BENGALI_LETTER.findall(text))
    latin = len(_LATIN_LETTER.findall(text))
    total = bn + latin
    return bn / total if total else 0.0


def bangla_char_count(text: str) -> int:
    return len(_BENGALI_LETTER.findall(text))
