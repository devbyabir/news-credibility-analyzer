# Data: source, license, audit and split

## Source and citation

| | |
|---|---|
| Dataset | **BanFakeNews-2.0** |
| Original publication | Shibu, Hrithik Majumdar (2024). *BanFakeNews-2.0*. Mendeley Data, V1. doi:[10.17632/kjh887ct4j.1](https://doi.org/10.17632/kjh887ct4j.1), published 4 June 2024 |
| License (original) | **CC BY 4.0**, as shown on the Mendeley Data page. Reuse and redistribution are allowed with attribution. |
| Mirror used for download | Hugging Face [`hrshihab/BanFakeNews-2.0`](https://huggingface.co/datasets/hrshihab/BanFakeNews-2.0), file `All60Kdataset.csv`, pinned to revision `97a974186fc0ec52d3031a4a31ef8ca1f52192aa` |
| Mirror license metadata | `apache-2.0`. The mirror's card has no description, citation or label documentation. |
| File checksum | SHA-256 `d9aed69b6019843a769bbcb045affd31c98cfa02f2b0bc821f551c54ba780c0a` (304,586,219 bytes). `nca.data download` verifies it. |

**License note.** The two pages show different licenses. The Mendeley Data record is the original
publication, so this project treats **CC BY 4.0** as authoritative and attributes the dataset to
its author. The Apache-2.0 tag on the mirror was set by the uploader and is not relied on. This
repository does **not** redistribute the CSV. It is downloaded at setup time. Reports quote
short excerpts (≤160 characters) only to illustrate failure cases, with attribution. The trained
model is derived from the data, so the demo and the model card carry the CC BY 4.0 attribution.
(Checked 12 Nov 2026. Licenses can change, so re-check before redistributing.)

**Labels.** The Mendeley record describes label `1` as authentic news (~47 k articles) and label
`0` as fake news (~13 k, manually annotated). The CSV has 48,678 rows labelled `1` and 12,903
labelled `0`. Every module imports this mapping from `src/nca/labels.py`:
`0 = fake`, `1 = authentic`. The legacy notebook had it reversed.

Columns used: `Content` (article body) and `Label`. `Headline` is loaded but not used for
training, because the demo accepts article text.

## Audit (deterministic, `python -m nca.data audit`)

All numbers come from `reports/robust/data_audit.json`. The baseline run's file is identical.

| Step | Rows |
|---|---:|
| Raw rows | 61,581 |
| Blank `Content` (excluded) | 1 |
| Invalid labels | 0 |
| *Raw repeats, exact `Content` match (informational)* | *3,187 (1,640 with a label different from the first copy)* |
| *Raw repeats after stripping surrounding whitespace (informational)* | *3,593 (1,970 with a different label)* |
| *Repeats after normalisation (informational)* | *3,848 (1,987 with a different label)* |
| Rows in **label-conflict duplicate clusters** (excluded) | 6,825 (3,243 fake + 3,582 authentic, in 3,159 clusters) |
| Same-label exact duplicates collapsed (excluded) | 1,844 |
| **Clean rows used** | **52,911** (8,028 fake, 44,883 authentic) |

Row accounting is asserted in code: clean + excluded = raw. Every excluded `row_id` and its
reason is saved in `reports/*/excluded_rows.csv.gz`.

### Definitions
- **Normalisation** (`nca.text.normalize_text`): Unicode NFC; zero-width characters, soft hyphens,
  URLs and e-mails removed; whitespace (including NBSP) collapsed; ASCII digits mapped to Bangla
  digits; sentence punctuation spaced out. The duplicate key also drops every non-letter/digit.
- **Near duplicates**: cosine similarity of binary word-5-gram shingle sets,
  `|A∩B| / sqrt(|A|·|B|)`, computed exactly with sparse matrix products (no sampling).
  Inspecting sampled pairs showed:
  - ≥ 0.8: the same article with small edits.
  - 0.5–0.8: the same story with heavily shared wording, often from different portals.
  - < 0.5: mostly distinct articles.

### Policy for conflicting labels
Exact duplicates plus near duplicates at similarity ≥ 0.8 form **duplicate clusters**. If one
cluster carries both labels, the same text was annotated as both fake and authentic. Nothing in
the data says which annotation is right, so **every row of that cluster is excluded** from training
and evaluation. This is reported and never resolved silently. The alternatives were rejected:
majority vote is meaningless for 1-vs-1 conflicts, and "keep the first" is arbitrary. The cost is
large: 25 % of the fake rows are removed. That cost is itself a finding about the dataset (fake
and authentic portions overlap heavily) and is listed under limitations.

### Split (leakage protection)
- **Split groups**: rows connected by similarity ≥ 0.5 (transitively) share a `group_id`.
  The 52,911 clean rows form 44,396 groups. 5,541 groups have more than one row, and the
  largest has 31 rows.
- **Held-out test set**: one fold of `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`
  gives 10,583 test rows (1,606 fake) and 42,328 training rows (6,422 fake). It is created
  **before** any vectorizer or model is fit, and no group appears on both sides (asserted in code).
- **Model selection**: 5-fold `StratifiedGroupKFold` on the training rows only. Each fold fits
  its own TF-IDF vocabulary.
- The exact split is saved in `reports/*/split_manifest.csv.gz` (row_id, group_id, label, split,
  text hash). The baseline and robust runs produce identical manifests.

## Shortcut cues (why there is a "robust" configuration)

The first leakage-free model (`reports/baseline/`) scored macro F1 ≈ 0.96. Its most influential
features (`reports/baseline/top_features.md`) were not about credibility. They were markers of
*when* and *where* an article was collected. Share of clean articles containing each cue:

| Cue (after normalisation) | % of fake | % of authentic |
|---|---:|---:|
| Year 2018 (`২০১৮`) | 1.6 | 20.8 |
| A year 2020–2029 (`২০২x`) | 3.3 | 1.6 |
| COVID-era words (`করোনা`, `কোভিড`) | 4.3 | 0.0 |
| `বিডি…` source/byline tokens | 0.7 | 12.7 |
| Slash tokens such as reporter initials (`x/y`) | 6.7 | 39.0 |
| Mis-encoded `মতিনিধি` (for প্রতিনিধি) | 3.9 | 0.0 |
| Parentheses | 30.0 | 56.0 |
| Exclamation mark | 21.4 | 6.6 |
| Month `সেপ্টেম্বর` / `অক্টোবর` (training split only) | 2.4 / 2.8 | 18.2 / 16.9 |
| Weekday names, e.g. `বুধবার` (training split only) | 4.4 | 15.0 |

The authentic articles appear to come mostly from an earlier period than many fake ones, and they carry
portal-specific bylines. A model that keys on these cues would label a genuine 2025 article
"fake-like" just because it mentions a 2020s year. `nca.text.normalize_robust` therefore removes
digits, slash tokens, brackets, COVID-era words, `বিডি…` tokens, the mis-encoded word and
Gregorian month names. The month names were added in a second iteration: an informal check with
a hand-written 2025 article showed `জানুয়ারি` pushing the score towards fake. The intermediate
run without month masking (not committed) scored test macro F1 0.958. For transparency: the test
split had already been scored for that intermediate run. The change was motivated by
training-split statistics and the hand-written example, not by test metrics, and it lowered the
test score slightly (0.958 → 0.956). Weekday names are kept,
because citing the day of an event is ordinary reporting style rather than a collection artifact. The
deployed model uses this preprocessing. Exclamation marks are kept on purpose: sensational
punctuation is a legitimate style signal, not a collection artifact.

This masking is a mitigation, not a cure. Topic, portal style and period still differ between the
classes in ways a bag-of-n-grams model can exploit. See the model card's limitations.

## Typography artifacts neutralised by standard normalisation

In the raw CSV, 58.3 % of fake vs 30.1 % of authentic articles have a danda directly followed by a
letter (`।এ`). 22.5 % vs 43.8 % contain NBSP characters, and 3.4 % vs 0.8 % use ASCII digits.
These are scraping and typesetting differences between sources. Standard normalisation makes them
identical across classes.
