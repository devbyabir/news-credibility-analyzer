# Legacy notebook (superseded — do not cite its metrics)

This folder keeps the original Google Colab notebook and the four result images it
produced, unchanged, as a historical record of the project's first version.

| File | What it is |
|---|---|
| `Bangla_Fake_News_Detection.ipynb` | Original Colab notebook (TF-IDF + SMOTE + RF / LightGBM / XGBoost / CatBoost / Naive Bayes). Outputs are kept as they were saved. |
| `figures/accuracy_comparison.jpg` | Accuracy bar chart exported from the notebook (was `1.jpg`). |
| `figures/wordcloud.jpg` | Word cloud exported from the notebook (was `2.jpg`). Rendered with the third-party Kalpurush font, downloaded at run time. |
| `figures/catboost_confusion_matrix.jpg` | CatBoost confusion matrix (was `3.jpg`). |
| `figures/catboost_roc.jpg` | CatBoost ROC curve (was `4.jpg`). |

The images were saved alongside the notebook. They appear to be exports of its plots
(the numbers match the notebook outputs), but nothing in the files records who made
them, so treat their provenance as "derived from the legacy notebook".

## Why the old results are invalid

The ~90 % accuracy / 0.96 AUC reported in the notebook and in these images are
**not a trustworthy benchmark**:

1. **Labels are reversed.** BanFakeNews-2.0 uses `0 = fake`, `1 = authentic`.
   The notebook labels `1` as "Fake" in every chart, in the confusion matrix and in
   `predict_news()`. Every "Real"/"Fake" caption in these figures is swapped.
2. **Leakage from SMOTE and TF-IDF.** The TF-IDF vocabulary was fit on the full
   dataset and SMOTE generated synthetic samples *before* K-fold splitting, so
   synthetic neighbours of test articles were in the training folds. The confusion
   matrix sums to 48,678 per class — the oversampled data, not real articles.
3. **Duplicates across folds.** The dataset contains thousands of repeated articles
   (some with conflicting labels). They were neither removed nor kept on one side
   of each split, so identical text was in train and test.
4. **Imbalance hidden by accuracy.** Only accuracy and fake-class-agnostic plots
   were reported for a 79 % / 21 % class split.

The corrected, leakage-free pipeline lives in `src/nca/` and its verified results are
in the root `README.md` and `reports/`.
