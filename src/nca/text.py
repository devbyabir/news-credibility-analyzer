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

BENGALI_BLOCK = "ঀ-৿"

_INVISIBLE = re.compile(r"[­​‌‍⁠﻿]")
_URL = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"\s*([।॥!?,;:])\s*")
_DIGITS = str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯")
_TOKEN = re.compile(rf"[{BENGALI_BLOCK}a-z0-9]+")
_BENGALI_LETTER = re.compile(rf"[{BENGALI_BLOCK}]")
_LATIN_LETTER = re.compile(r"[A-Za-z]")
_KEY_STRIP = re.compile(rf"[^{BENGALI_BLOCK}a-z0-9]+")


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
