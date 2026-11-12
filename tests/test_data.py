import pandas as pd

from nca.data import audit_and_clean, grouped_train_test_split, near_duplicate_pairs
from nca.labels import AUTHENTIC, FAKE
from nca.text import normalize_text

BASE = (
    "সরকারি চাকরিতে প্রবেশের বয়সসীমা বাড়ানোর দাবিতে আজ জাতীয় প্রেস ক্লাবের সামনে মানববন্ধন "
    "করেছে সাধারণ শিক্ষার্থীরা তারা বলেন দ্রুত সুপারিশ বাস্তবায়ন করতে হবে"
)


def _df(rows):
    return pd.DataFrame(
        [{"row_id": i, "headline": "", "text": t, "label": lab} for i, (t, lab) in enumerate(rows)]
    )


def test_near_duplicate_detection_finds_small_edits_only():
    edited = BASE.replace("আজ", "গতকাল")
    other = "বার্সেলোনা লা লিগায় ভ্যালেন্সিয়ার সঙ্গে ড্র করে আবারও পয়েন্ট হারিয়েছে মেসির গোলে"
    pairs = near_duplicate_pairs([normalize_text(t) for t in (BASE, edited, other)], threshold=0.5)
    linked = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in linked
    assert not any(2 in p for p in linked)


def test_blank_rows_and_exact_duplicates_are_accounted_for():
    df = _df(
        [
            (BASE, AUTHENTIC),
            (BASE + " ", AUTHENTIC),
            ("   ", FAKE),
            (None, AUTHENTIC),
            ("সম্পূর্ণ আলাদা একটি খবর যেখানে অন্য বিষয়ে লেখা আছে", FAKE),
        ]
    )
    res = audit_and_clean(df)
    a = res.audit
    assert a["blank_text_rows"] == 2
    assert a["same_label_exact_duplicates_removed"] == 1
    assert a["clean_rows"] == 2
    assert a["clean_rows"] + a["excluded_rows_total"] == a["raw_rows"]
    assert set(res.excluded["reason"]) == {"blank_text", "exact_duplicate_same_label"}


def test_conflicting_labels_are_excluded_not_resolved():
    df = _df(
        [
            (BASE, AUTHENTIC),
            (BASE, FAKE),
            (BASE + " বলে জানা গেছে", AUTHENTIC),
            ("সম্পূর্ণ আলাদা একটি খবর যেখানে অন্য বিষয়ে লেখা আছে", FAKE),
        ]
    )
    res = audit_and_clean(df)
    # The exact pair conflicts, and the near-duplicate joins the same cluster: all three excluded.
    assert res.audit["label_conflict_clusters"] == 1
    assert res.audit["label_conflict_rows_excluded"] == 3
    assert res.data["row_id"].tolist() == [3]


def test_grouped_split_never_puts_near_duplicates_on_both_sides(synthetic_df):
    # Add near-duplicate copies of some articles; they must land on the same side.
    extra = synthetic_df.head(30).copy()
    extra["text"] = extra["text"].str.replace("।", " আরও তথ্য।", regex=False)
    extra["row_id"] = extra["row_id"] + 1000
    res = audit_and_clean(pd.concat([synthetic_df, extra], ignore_index=True))
    data = res.data
    tr, te = grouped_train_test_split(data, test_fraction=0.25, seed=1)
    assert not set(data.iloc[tr]["group_id"]) & set(data.iloc[te]["group_id"])
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(data)
    # both classes represented in the test split
    assert set(data.iloc[te]["label"]) == {FAKE, AUTHENTIC}


def test_split_is_deterministic(synthetic_df):
    res = audit_and_clean(synthetic_df)
    a = grouped_train_test_split(res.data, 0.2, seed=42)[1]
    b = grouped_train_test_split(res.data, 0.2, seed=42)[1]
    assert a.tolist() == b.tolist()
