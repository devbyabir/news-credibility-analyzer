# CV / portfolio text

Every claim below can be checked against this repository (`reports/robust/test_results.json`,
`docs/DATA.md`, the tests and the commit history). Replace `<links>` once the repository and demo
are public.

## Title

**News Credibility Analyzer: Bangla Fake-News Text Classifier**
Python · scikit-learn · Gradio · pandas · pytest · GitHub Actions

## One-line version

Built a leakage-free Bangla news-text classifier on BanFakeNews-2.0 (61.6k articles): test macro F1
0.956 and fake-class F1 0.925, with shortcut-cue analysis and an explainable Gradio demo.

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
- Shipped Streamlit and Gradio web demos with uncertainty bands, per-word explanations and input validation,
  plus 41 pytest tests, CI and deployment packaging for Streamlit Cloud and Hugging Face Spaces.

## Links
- Code: `https://github.com/<your-github-username>/news-credibility-analyzer`
- Live demo: `https://<your-subdomain>.streamlit.app` or
  `https://huggingface.co/spaces/<your-hf-username>/news-credibility-analyzer` (only after you deploy it)

## Do not claim
- That it "detects fake news with 96–98 % accuracy" in general. The scores are in-distribution
  only (see Limitations in the README).
- That BanglaBERT or any transformer was fine-tuned or evaluated. The script exists but has not
  been run at full scale.
- That the dataset or the legacy figures were created by you.
