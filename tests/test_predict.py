import random

import numpy as np
import pytest

from nca.evaluate import band_of, band_stats, classification_metrics
from nca.labels import AUTHENTIC, FAKE
from nca.predict import MAX_CHARS, InputError, Predictor, validate_text

from .conftest import AUTH_WORDS, FAKE_WORDS, make_article


def test_model_loads_and_classes_match_labels(tiny_model_dir):
    p = Predictor.load(tiny_model_dir)
    assert sorted(p.pipeline.classes_) == [FAKE, AUTHENTIC]


def test_missing_model_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nca.train"):
        Predictor.load(tmp_path)


def test_prediction_direction_follows_label_meaning(tiny_model_dir):
    p = Predictor.load(tiny_model_dir)
    rng = random.Random(123)
    fake = p.predict(make_article(FAKE_WORDS, rng))
    auth = p.predict(make_article(AUTH_WORDS, rng))
    assert fake.label == FAKE and fake.label_name == "fake" and fake.p_fake > 0.5
    assert auth.label == AUTHENTIC and auth.label_name == "authentic" and auth.p_fake < 0.5
    assert fake.band.endswith("fake_like")
    assert fake.top_fake_terms and all(t in FAKE_WORDS for t, _ in fake.top_fake_terms)
    assert auth.top_authentic_terms


def test_band_statistics_from_metadata_are_attached(tiny_model_dir):
    p = Predictor.load(tiny_model_dir)
    r = p.predict(make_article(FAKE_WORDS, random.Random(5)))
    if r.band == "strongly_fake_like":
        assert r.band_test_fake_rate == 0.9 and r.band_test_n == 10


@pytest.mark.parametrize(
    "text, msg",
    [
        ("", "paste"),
        ("   ", "paste"),
        (None, "paste"),
        ("This is an English news article about the economy and politics today.", "Bangla"),
        ("ঢাকা", "too short"),
    ],
)
def test_invalid_input_rejected(text, msg):
    with pytest.raises(InputError, match=msg):
        validate_text(text)


def test_excessive_input_rejected():
    with pytest.raises(InputError, match="too long"):
        validate_text("ক" * (MAX_CHARS + 1))


def test_short_text_warns_but_scores(tiny_model_dir):
    p = Predictor.load(tiny_model_dir)
    r = p.predict("সরকার মন্ত্রণালয় জানিয়েছে প্রকল্প বাস্তবায়ন সভায় সিদ্ধান্ত হয়েছে")
    assert r.warnings and "Short input" in r.warnings[0]


def test_metrics_report_fake_class_explicitly():
    y = np.array([0, 0, 1, 1, 1, 1])
    pred = np.array([0, 1, 1, 1, 1, 0])
    score = np.array([0.9, 0.4, 0.1, 0.2, 0.3, 0.6])
    m = classification_metrics(y, pred, score)
    assert m["per_class"]["fake"]["recall"] == 0.5
    assert m["per_class"]["fake"]["support"] == 2
    assert m["confusion_matrix"]["matrix"] == [[1, 1], [1, 3]]
    assert 0 <= m["roc_auc"] <= 1 and "pr_auc_fake" in m


def test_bands_cover_unit_interval():
    assert band_of(0.0) == "strongly_authentic_like"
    assert band_of(0.5) == "uncertain"
    assert band_of(1.0) == "strongly_fake_like"
    stats = band_stats(np.array([0, 1, 1]), np.array([0.95, 0.1, 0.5]))
    assert sum(s["n_test"] for s in stats) == 3
