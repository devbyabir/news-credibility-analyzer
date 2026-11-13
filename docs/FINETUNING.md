# Optional: fine-tuning BanglaBERT

`src/nca/finetune.py` fine-tunes [`csebuetnlp/banglabert`](https://huggingface.co/csebuetnlp/banglabert)
(ELECTRA-base, ~110 M parameters) as a second candidate model. It uses the **same data audit and
the same held-out test split** as the scikit-learn pipeline (same config and seed). It carves a
grouped validation split out of the training portion to select checkpoints, and scores the test
split only once, at the end.

## Status: not yet run at full scale

| | Status |
|---|---|
| Entry point runs end to end (download model → train → select checkpoint → test → save → latency) | ✅ Verified on CPU with a tiny debug run (64 training articles, 4 optimiser steps, robust preprocessing, `transformers` 5.22.2, `torch` 2.15.0+cpu). Output: `reports/finetune_smoke.json`. The metrics in that file are meaningless by design. |
| Full fine-tuning on ~38 k articles | ❌ Not run. On this project's development machine (CPU only) a debug step took ~3.5 s at batch 8 × 128 tokens, which puts the full 3-epoch run at roughly a day. It needs a GPU. |

No BanglaBERT benchmark numbers are claimed anywhere in this repository.

## How to run it on a free GPU (Google Colab or Kaggle)

1. Open a new Colab notebook and select **Runtime → Change runtime type → T4 GPU**.
2. Run:
   ```bash
   !git clone https://github.com/devbyabir/news-credibility-analyzer.git
   %cd news-credibility-analyzer
   !pip install -q -r requirements.txt -e ".[finetune]"
   !python -m nca.data download
   !python -m nca.finetune --output-dir artifacts/banglabert --results reports/finetune_banglabert.json
   ```
   Defaults: 3 epochs, batch size 16, learning rate 2e-5, max 256 tokens, class-weighted loss and
   fp16 on GPU. Expect very roughly 1–2 hours on a T4. This is an estimate, not a measurement.
3. Download `reports/finetune_banglabert.json` and commit it. Keep the model weights (~440 MB)
   out of Git.

## How to compare with the deployed model

`reports/finetune_banglabert.json` contains the same metric block as `reports/robust/test_results.json`.
Compare macro F1 and fake-class precision/recall on the identical test split, CPU latency per article
and artifact size. Measured on CPU here (debug run): **~87 ms per article** and **443.5 MB** for BanglaBERT.
The deployed TF-IDF model is listed in the README.

Caveats:
- By default the script uses the same robust preprocessing as the deployed model
  (`--preprocessor robust`), which masks dates, bylines and COVID-era words. Use
  `--preprocessor standard` to compare against the unmasked baseline instead.
- A transformer would make hosting heavier: about 0.5 GB of weights, more RAM and roughly 87 ms vs
  1.2 ms per article on CPU. Streamlit Community Cloud allows up to about 2.7 GB of memory per app,
  so it may still fit, but this is untested. A CPU Gradio Space on Hugging Face needs a paid plan
  (see docs/DEPLOYMENT.md §3). Switch only if it is clearly better on the fake class.
