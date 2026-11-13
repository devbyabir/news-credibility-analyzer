---
title: News Credibility Analyzer
emoji: 📰
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
short_description: Bangla news text-style classifier (does not verify facts)
---

# News Credibility Analyzer (demo)

Paste Bangla news text and see whether it **resembles** the fake or authentic articles in the
BanFakeNews-2.0 dataset. This is a machine-learning text classifier. It does **not** check facts,
sources or dates, and its output is not a judgement about whether a story is true.

- Source code, evaluation protocol and model card: https://github.com/devbyabir/news-credibility-analyzer
- Model: TF-IDF + logistic regression (scikit-learn), trained with a leakage-free, duplicate-aware split.
- Labels: `0 = fake`, `1 = authentic`.
- Training data: BanFakeNews-2.0, Hrithik Majumdar Shibu, Mendeley Data, V1, 2024,
  doi:10.17632/kjh887ct4j.1, licensed CC BY 4.0. The model artifact in this Space is derived from that data.
- Code license: MIT.
