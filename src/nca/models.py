"""Model and feature factories. Everything a model sees is fit inside a pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nca.text import normalize_text, tokenize

# Models whose scores we expose in the product must provide predict_proba.
PROBABILISTIC_MODELS = {"logreg", "complementnb", "dummy"}


def make_vectorizer(spec: dict[str, Any]) -> TfidfVectorizer:
    spec = dict(spec)
    analyzer = spec.pop("analyzer", "word")
    ngram = tuple(spec.pop("ngram_range", (1, 1)))
    common = dict(
        preprocessor=normalize_text,
        lowercase=False,
        ngram_range=ngram,
        dtype=np.float32,
        **spec,
    )
    if analyzer == "word":
        return TfidfVectorizer(analyzer="word", tokenizer=tokenize, token_pattern=None, **common)
    if analyzer in {"char", "char_wb"}:
        return TfidfVectorizer(analyzer=analyzer, **common)
    raise ValueError(f"Unknown analyzer {analyzer!r}")


def make_classifier(kind: str, seed: int, **params: Any) -> BaseEstimator:
    if kind == "dummy":
        return DummyClassifier(strategy="most_frequent")
    if kind == "logreg":
        return LogisticRegression(
            class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed, **params
        )
    if kind == "linearsvm":
        return LinearSVC(class_weight="balanced", random_state=seed, **params)
    if kind == "complementnb":
        return ComplementNB(**params)
    raise ValueError(f"Unknown classifier {kind!r}")


def make_pipeline(feature_spec: dict | None, kind: str, seed: int, **params: Any) -> Pipeline:
    steps = []
    if feature_spec is not None:
        steps.append(("tfidf", make_vectorizer(feature_spec)))
    steps.append(("clf", make_classifier(kind, seed, **params)))
    return Pipeline(steps)
