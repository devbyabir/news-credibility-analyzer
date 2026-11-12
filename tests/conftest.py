"""Synthetic Bangla fixtures so tests never need the 305 MB dataset."""

from __future__ import annotations

import json
import random

import joblib
import pandas as pd
import pytest

from nca.labels import AUTHENTIC, FAKE
from nca.models import make_pipeline

AUTH_WORDS = (
    "সরকার মন্ত্রণালয় জানিয়েছে প্রকল্প বাস্তবায়ন সভায় সিদ্ধান্ত বাজেট উন্নয়ন কর্মকর্তা "
    "বৃহস্পতিবার সংবাদ সম্মেলনে বলেন জেলা প্রশাসন উপজেলা নির্বাচন কমিশন"
).split()
FAKE_WORDS = (
    "অবিশ্বাস্য গোপন ফাঁস হতবাক চাঞ্চল্যকর ভাইরাল শেয়ার করুন অলৌকিক রহস্য চমকে দেওয়া ষড়যন্ত্র গুজব বিস্ময়কর তাজ্জব আজব"
).split()


def make_article(words: list[str], rng: random.Random, n: int = 60) -> str:
    return " ".join(rng.choice(words) for _ in range(n)) + "।"


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    rng = random.Random(0)
    rows = []
    for i in range(160):
        label = FAKE if i % 4 == 0 else AUTHENTIC
        words = FAKE_WORDS if label == FAKE else AUTH_WORDS
        rows.append({"row_id": i, "headline": f"h{i}", "text": make_article(words, rng), "label": label})
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory, synthetic_df):
    """A real (tiny) trained pipeline saved exactly like nca.train saves one."""
    d = tmp_path_factory.mktemp("model")
    spec = {"analyzer": "word", "ngram_range": [1, 1], "min_df": 1, "sublinear_tf": True}
    pipe = make_pipeline(spec, "logreg", seed=0, C=4.0).fit(synthetic_df["text"], synthetic_df["label"])
    joblib.dump(pipe, d / "model.joblib")
    meta = {
        "selected_candidate": "tfidf_word_logreg",
        "labels": {"0": "fake", "1": "authentic"},
        "bands": [{"band": "strongly_fake_like", "n_test": 10, "observed_fake_rate": 0.9}],
    }
    (d / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return d
