# Model card: news-credibility-analyzer (v1.0.0)

## Model details
- **Type:** scikit-learn pipeline. Robust Bangla text normalisation → TF-IDF over word unigrams
  and bigrams (`min_df=2`, `max_df=0.9`, ≤300k features, sublinear TF) → logistic regression
  (`C=16`, `class_weight="balanced"`, liblinear).
- **Output:** `P(fake)`, taken from the probability column of class `0`. The decision rule is
  `P(fake) ≥ 0.5 → fake-like`. The demo shows five bands instead of a hard label.
- **Labels:** `0 = fake`, `1 = authentic` (BanFakeNews-2.0 convention).
- **Artifact:** `artifacts/model/model.joblib` (7.0 MB) and `metadata.json`, produced by
  `python -m nca.train --config configs/robust.json`. Requires scikit-learn 1.9.1.
- **Author:** Abir Ahmmed. Code license: MIT.

## Intended use
- Demonstrating and teaching text classification, leakage-free evaluation and dataset-artifact
  analysis on a low-resource language.
- Exploring which wording patterns separate the two collections in BanFakeNews-2.0.

## Out of scope
- Deciding whether any real news story is true or false.
- Content moderation, ranking or takedown decisions, or judging individuals or outlets.
- Text that is not Bangla, or short text (headlines, single posts). The demo warns on short input
  and rejects non-Bangla input.

## Training data
BanFakeNews-2.0: Hrithik Majumdar Shibu, Mendeley Data V1 (2024), doi:10.17632/kjh887ct4j.1,
CC BY 4.0. Downloaded from the Hugging Face mirror `hrshihab/BanFakeNews-2.0` at revision
`97a97418…`, SHA-256 verified. Of 61,581 rows, the audit used 52,911 after removing
1 blank row, 6,825 rows in label-conflicting duplicate clusters and 1,844 same-label duplicates.
See [docs/DATA.md](docs/DATA.md).

## Evaluation
- **Split:** a grouped, stratified held-out test set (10,583 articles, 1,606 fake). Exact and
  near-duplicates (word-5-gram shingle similarity ≥ 0.5) share a group, so no story appears on
  both sides. The split is created before any fitting and saved in `reports/robust/split_manifest.csv.gz`.
- **Selection:** 5-fold grouped CV on the 42,328 training articles only. Candidates: majority
  baseline, Complement NB, logistic regression, linear SVM and char-n-gram logistic regression.
  The rule, fixed before testing, picks the highest CV macro F1 among models that output
  probabilities.
- **Test results (single evaluation):**

| Metric | Value |
|---|---:|
| Macro F1 | 0.956 |
| Accuracy | 0.978 |
| Balanced accuracy | 0.942 |
| Fake: precision / recall / F1 | 0.962 / 0.891 / 0.925 |
| Authentic: precision / recall / F1 | 0.981 / 0.994 / 0.987 |
| ROC-AUC | 0.988 |
| PR-AUC (fake) | 0.968 (prevalence 0.152) |
| Brier (fake) / ECE (10 bins) | 0.019 / 0.014 |
| Median CPU latency per article | 1.2 ms |

Majority-class baseline: macro F1 0.459, fake recall 0.

Full tables: `reports/robust/test_results.json`, `cv_summary.csv` and `cv_folds.csv`.
Most confident errors: `reports/robust/failure_cases.md`.

## Known shortcuts and limitations
- **Collection artifacts.** The two classes come from different portals and periods. Without
  masking, the strongest features were years, months, COVID-era words, byline fragments and a
  mis-encoded word (`reports/baseline/top_features.md`). The deployed preprocessing masks those.
  The test score fell only from 0.962 to 0.956, and the remaining top features still include
  portal names (for example `স্বাধীনবাংলা`, `সিটিনিউজ সেভেন`), spelling conventions and
  call-to-action words (`পড়ুন`, `দেখুন`). The model therefore still partly identifies sources and
  styles. **The high test score should not be read as real-world fake-news detection accuracy.**
- **Distribution shift.** No out-of-distribution test set exists. In an informal check, a
  hand-written, neutral 2025 economy report scored `P(fake) = 0.55` ("uncertain") with the deployed
  model.
- **Fake class.** About 11 % of fake test articles are missed (recall 0.891). The excluded
  label-conflict clusters removed 25 % of fake rows, so the fake class seen in training is narrower
  than in the raw data.
- **Calibration** looks good in-distribution (ECE 0.014), but this is not guaranteed for other text.
  The UI calls the number a "model score", not a probability of falsehood.
- **Fairness.** Errors may fall unevenly across topics, regions, outlets or political subjects.
  This was not audited.

## Ethical considerations
Labelling a real article as "fake" can harm its authors and readers. The demo shows a permanent
notice that it does not check facts, uses "fake-like / authentic-like" wording, shows uncertainty
bands, and explains which words drove each score.
