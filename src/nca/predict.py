"""Inference: input validation, scoring, uncertainty bands and explanations.

python -m nca.predict "বাংলা সংবাদের লেখা ..."
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import joblib

from nca.evaluate import BANDS, band_of
from nca.labels import AUTHENTIC, FAKE, LABEL_NAMES
from nca.text import bangla_char_count, bangla_ratio, normalize_text, tokenize

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "model"

MAX_CHARS = 20_000  # ~10x a typical article; longer input is rejected, not truncated
MIN_BANGLA_CHARS = 20  # below this there is nothing meaningful to score
SHORT_TEXT_TOKENS = 40  # training articles have a median of ~200 words
MIN_BANGLA_RATIO = 0.6

BAND_TEXT = {
    "strongly_authentic_like": "Strongly resembles authentic articles in the training data",
    "leans_authentic_like": "Leans towards authentic-like",
    "uncertain": "Uncertain: the model cannot separate this text well",
    "leans_fake_like": "Leans towards fake-like",
    "strongly_fake_like": "Strongly resembles fake articles in the training data",
}


class InputError(ValueError):
    """Raised for input the model should not score (empty, too long, not Bangla)."""


@dataclass
class Prediction:
    label: int
    label_name: str
    p_fake: float
    band: str
    band_text: str
    band_test_fake_rate: float | None
    band_test_n: int | None
    warnings: list[str] = field(default_factory=list)
    top_fake_terms: list[tuple[str, float]] = field(default_factory=list)
    top_authentic_terms: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def validate_text(text: str | None) -> tuple[str, list[str]]:
    """Return (stripped_text, warnings) or raise InputError."""
    if text is None or not str(text).strip():
        raise InputError("Please paste some Bangla news text.")
    text = str(text).strip()
    if len(text) > MAX_CHARS:
        raise InputError(f"Text is too long ({len(text):,} characters). The limit is {MAX_CHARS:,}.")
    n_letters = sum(ch.isalpha() for ch in text)
    if n_letters >= MIN_BANGLA_CHARS and bangla_ratio(text) < MIN_BANGLA_RATIO:
        raise InputError("This model only supports Bangla text. Most of the input is in another script.")
    if bangla_char_count(text) < MIN_BANGLA_CHARS:
        raise InputError("Text is too short to analyse. Please paste a full Bangla news article.")
    warnings = []
    n_tokens = len(tokenize(normalize_text(text)))
    if n_tokens < SHORT_TEXT_TOKENS:
        warnings.append(
            f"Short input ({n_tokens} words). The model was trained on full articles "
            "(median ~200 words); results on headlines or single sentences are unreliable."
        )
    return text, warnings


class Predictor:
    def __init__(self, pipeline, metadata: dict | None = None) -> None:
        self.pipeline = pipeline
        self.metadata = metadata or {}
        classes = list(pipeline.classes_)
        if sorted(classes) != [FAKE, AUTHENTIC]:
            raise ValueError(f"Model classes {classes} do not match labels {LABEL_NAMES}")
        self._fake_col = classes.index(FAKE)
        self._bands = {b["band"]: b for b in self.metadata.get("bands", [])}

    @classmethod
    def load(cls, model_dir: str | Path | None = None) -> Predictor:
        model_dir = Path(model_dir or os.environ.get("NCA_MODEL_DIR") or DEFAULT_MODEL_DIR)
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model at {model_path}. Train one with `python -m nca.train` "
                "or set NCA_MODEL_DIR to a directory containing model.joblib."
            )
        meta_path = model_dir / "metadata.json"
        metadata = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        return cls(joblib.load(model_path), metadata)

    def predict(self, text: str, explain: int = 6) -> Prediction:
        text, warnings = validate_text(text)
        p_fake = float(self.pipeline.predict_proba([text])[0, self._fake_col])
        label = FAKE if p_fake >= 0.5 else AUTHENTIC
        band = band_of(p_fake)
        stats = self._bands.get(band, {})
        fake_terms, auth_terms = self._explain(text, explain) if explain else ([], [])
        return Prediction(
            label=label,
            label_name=LABEL_NAMES[label],
            p_fake=p_fake,
            band=band,
            band_text=BAND_TEXT[band],
            band_test_fake_rate=stats.get("observed_fake_rate"),
            band_test_n=stats.get("n_test"),
            warnings=warnings,
            top_fake_terms=fake_terms,
            top_authentic_terms=auth_terms,
        )

    def _explain(self, text: str, k: int) -> tuple[list, list]:
        """Per-term contribution (tf-idf weight x coefficient) for linear models.

        Word models report their n-grams directly. Character n-gram models are
        attributed back to whole words: each n-gram's contribution is shared
        equally among its occurrences, and a word's score is the sum over the
        n-grams it contains. Positive score -> authentic-like.
        """
        steps = getattr(self.pipeline, "named_steps", {})
        vec, clf = steps.get("tfidf"), steps.get("clf")
        if vec is None or not hasattr(clf, "coef_"):
            return [], []
        x = vec.transform([text])
        coef = clf.coef_.ravel()  # positive -> authentic (class 1)
        contrib = dict(zip(x.indices.tolist(), (x.data * coef[x.indices]).tolist()))
        if vec.analyzer == "word":
            names = vec.get_feature_names_out()
            scores = {str(names[i]): c for i, c in contrib.items()}
        else:
            analyzer, vocab = vec.build_analyzer(), vec.vocabulary_
            occurrences = Counter(analyzer(text))
            words = Counter(tokenize(normalize_text(text)))
            scores = {}
            for word, n in words.items():
                s = 0.0
                for g in analyzer(word):
                    j = vocab.get(g)
                    if j is not None and j in contrib:
                        s += contrib[j] / occurrences[g]
                scores[word] = s * n
        ranked = sorted(scores.items(), key=lambda kv: kv[1])
        fake = [(t, -s) for t, s in ranked[:k] if s < 0]
        auth = [(t, s) for t, s in ranked[::-1][:k] if s > 0]
        return fake, auth


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score a Bangla news text.")
    ap.add_argument("text", nargs="?", help="text to score (reads stdin if omitted)")
    ap.add_argument("--model-dir", type=Path, default=None)
    args = ap.parse_args(argv)
    import sys

    text = args.text if args.text is not None else sys.stdin.read()
    try:
        result = Predictor.load(args.model_dir).predict(text)
    except InputError as exc:
        raise SystemExit(f"Input rejected: {exc}") from None
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))


__all__ = ["BANDS", "InputError", "Prediction", "Predictor", "validate_text"]

if __name__ == "__main__":
    main()
