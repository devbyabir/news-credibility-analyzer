import pytest

from nca import labels
from nca.text import bangla_ratio, dedup_key, normalize_text, tokenize


def test_label_mapping_matches_dataset_definition():
    # BanFakeNews-2.0: 0 = fake, 1 = authentic. The legacy notebook had this reversed.
    assert labels.FAKE == 0
    assert labels.AUTHENTIC == 1
    assert labels.label_name(0) == "fake"
    assert labels.label_name(1) == "authentic"


def test_unknown_label_rejected():
    with pytest.raises(ValueError):
        labels.label_name(2)


def test_tokenizer_keeps_bangla_words_whole():
    # sklearn's default \b\w\w+\b pattern would split this into fragments like "বব", "লয".
    assert tokenize(normalize_text("বিশ্ববিদ্যালয়ে প্রশাসন")) == ["বিশ্ববিদ্যালয়ে", "প্রশাসন"]


def test_normalize_removes_source_typography_artifacts():
    a = normalize_text("আহত।এরপর পুলিশ 25 জন পে‌য়াজ")
    b = normalize_text("আহত । এরপর পুলিশ ২৫ জন পেয়াজ")
    assert a == b


def test_normalize_strips_urls_and_is_idempotent():
    s = normalize_text("দেখুন https://example.com/x এবং www.test.org এখানে")
    assert "http" not in s and "www" not in s
    assert normalize_text(s) == s
    assert normalize_text(None) == ""


def test_dedup_key_ignores_spacing_and_punctuation():
    assert dedup_key("ঢাকা: আজ   বৃষ্টি!") == dedup_key("ঢাকা আজ বৃষ্টি")
    assert dedup_key("ঢাকা আজ বৃষ্টি") != dedup_key("ঢাকা আজ রোদ")


def test_bangla_ratio():
    assert bangla_ratio("বাংলা") == 1.0
    assert bangla_ratio("english only") == 0.0
    assert 0 < bangla_ratio("বাংলা and english") < 1
