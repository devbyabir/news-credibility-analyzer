# CV / portfolio text

Every claim below can be checked against this repository (`reports/robust/test_results.json`,
`docs/DATA.md`, the tests and the commit history).

## Title

**News Credibility Analyzer: Bangla Fake-News Text Classifier**
Python · scikit-learn · Streamlit · pandas · pytest · GitHub Actions

## One-line version

Built a leakage-free Bangla news-text classifier on BanFakeNews-2.0 (61.6k articles): test macro F1
0.956 and fake-class F1 0.925, with shortcut-cue analysis and a live, explainable Streamlit demo.

## Bullet version (pick 2–4)

- Rebuilt an earlier notebook project into a tested Python package. Fixed reversed labels,
  TF-IDF/SMOTE leakage, cross-fold duplicates and a tokenizer that split Bangla words at vowel signs.
- Designed a duplicate-aware evaluation: a deterministic audit (exact and shingle-based
  near-duplicates, 3,159 label-conflict clusters excluded under a documented policy) and a
  grouped stratified held-out split. Model selection used grouped cross-validation only.
- Compared 5 models under one protocol. The selected TF-IDF + logistic regression reached test
  macro F1 0.956 (fake precision 0.962, recall 0.891) against a 0.459 majority baseline.
- Found that the first model relied on collection artifacts (years, months, bylines, COVID-era
  words). Masked them and documented the residual source bias instead of reporting the higher score.
- Shipped a public Streamlit demo with uncertainty bands, per-word explanations and input validation;
  the model is published as a checksum-verified GitHub Release. 50 pytest tests and CI on Python 3.12/3.13.

## Links
- Code: `https://github.com/devbyabir/news-credibility-analyzer`
- Live demo: `https://bangla-news-credibility-analyzer.streamlit.app`
- Model release: `https://github.com/devbyabir/news-credibility-analyzer/releases/tag/v1.0.0`

## Portfolio card

**News Credibility Analyzer**: a Bangla news-text classifier trained on BanFakeNews-2.0 with a
leakage-free, duplicate-aware evaluation (test macro F1 0.956, in-distribution). It explains which
words drove each score and states plainly that it does not verify facts.
[Live demo](https://bangla-news-credibility-analyzer.streamlit.app) · [Code](https://github.com/devbyabir/news-credibility-analyzer)

## Do not claim
- That it "detects fake news with 96–98 % accuracy" in general. The scores are in-distribution
  only (see Limitations in the README).
- That BanglaBERT or any transformer was fine-tuned or evaluated. The script exists but has not
  been run at full scale.
- That the dataset or the legacy figures were created by you.
