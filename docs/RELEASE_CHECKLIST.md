# Release checklist

There are three separate milestones. Tick them in order; finishing one does not mean the next is done.

| Milestone | Meaning | Section |
|---|---|---|
| **A. Code on GitHub** | The repository is pushed and CI passes. | A |
| **B. Model release published** | GitHub Release `v1.0.0` holds both model files, and their checksums match. | B |
| **C. Live demo deployed** | A public Streamlit URL works, and the links in the README, CV and portfolio point to it. | C |

## 0. Before anything is published (local)

- [ ] `git status` is clean, and `git log --format='%an <%ae>' | sort -u` shows only your identity.
- [ ] No dataset or model binaries are tracked:
      `git ls-files | grep -Ei '\.(csv|joblib|pkl|pt|bin|safetensors)$' | grep -v '^reports/'` prints nothing.
      The small CV result CSVs in `reports/` are intentional.
- [ ] No tracked file is larger than 5 MB: `git ls-files -z | xargs -0 du -k | sort -n | tail -5`.
      The largest is the legacy notebook (~2.6 MB).
- [ ] No secrets: `git grep -nIE "(hf_[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16})"`
      prints nothing.
- [ ] **Your decision:** `LICENSE` says "MIT, Copyright (c) 2026 Abir Ahmmed" for the code only.
      Confirm the holder, year and license, or change them before pushing.
- [ ] Quality gates pass: `ruff check .`, `ruff format --check .` and `pytest` (no dataset needed).
- [ ] Demos start locally:
      - `streamlit run deploy/streamlit/streamlit_app.py`
      - `python app.py`
      - In both, the two examples return a result, and empty, English-only and over-long input
        each show a clear message.
- [ ] Optional full reproduction (~25 min): `python -m nca.data download && python -m nca.train`,
      then compare the new `reports/robust/test_results.json` with the committed one
      (`git diff --stat reports/robust`).
- [ ] README and MODEL_CARD metrics match `reports/robust/test_results.json`:
      test macro F1 0.956, fake F1 0.925.

## A. Code on GitHub

- [ ] Replace the username placeholder in the publishing files, after a dry run:
      `python scripts/set_github_username.py YOUR_NAME`, then the same command with `--write`.
      This updates `README.md` (CI badge and clone URL), `deploy/streamlit/model_source.json`
      (`base_url`), `deploy/huggingface/README.md`, `docs/CV.md` and `docs/FINETUNING.md`.
      Review with `git diff` and commit.
- [ ] Create an **empty, public** GitHub repository named `news-credibility-analyzer`, with no
      README, license or `.gitignore`. It must be public for the Streamlit model download.
- [ ] `git remote add origin https://github.com/YOUR_NAME/news-credibility-analyzer.git`
- [ ] `git push -u origin main`
- [ ] The GitHub Actions CI run (ruff + pytest on Python 3.12 and 3.13) passes.
- [ ] Optional: add repository topics `bangla`, `nlp`, `fake-news-detection`, `text-classification`,
      `scikit-learn`, `streamlit`.

## B. Model release published

- [ ] `sha256sum artifacts/model/model.joblib artifacts/model/metadata.json` matches
      `deploy/streamlit/model_source.json`:
      - `model.joblib` → `88148c0b…f976c`
      - `metadata.json` → `e9dd516e…2dfc3`
- [ ] GitHub → *Releases* → *Draft a new release* → new tag **`v1.0.0`** on `main`.
- [ ] Attach **both** assets with their original names: `model.joblib` and `metadata.json`.
- [ ] *Publish release*. It must not be a draft, because draft assets cannot be downloaded publicly.
- [ ] In a private browser window, both URLs download without a login:
      - `https://github.com/YOUR_NAME/news-credibility-analyzer/releases/download/v1.0.0/model.joblib`
      - `https://github.com/YOUR_NAME/news-credibility-analyzer/releases/download/v1.0.0/metadata.json`
- [ ] Optional check that the uploaded files are the committed ones: download them and compare
      `sha256sum` with the two values above.

## C. Live demo deployed (Streamlit Community Cloud)

- [ ] <https://share.streamlit.io> → create an app from `YOUR_NAME/news-credibility-analyzer`:
      branch `main`, main file path `deploy/streamlit/streamlit_app.py`, and *Advanced settings* →
      Python **3.12** (docs/DEPLOYMENT.md §2).
- [ ] The build log shows `deploy/streamlit/requirements.txt` being installed, and the app starts.
- [ ] Smoke test of the **public URL** in a private browser window:
  - [ ] Both illustrative examples return a result (the sensational one should be "fake-like", the
        neutral one "authentic-like").
  - [ ] Empty input shows "Please paste some Bangla news text."
  - [ ] English-only text shows "This model only supports Bangla text…"
  - [ ] The "How it works" panel shows macro F1 0.956.
  - [ ] There is no "The model could not be loaded" message.
- [ ] Update links:
  - [ ] Replace the README "Live demo" line with the URL.
  - [ ] Add the URL to `docs/CV.md` "Links".
  - [ ] Add the URL to your CV and portfolio.
  - [ ] Commit and push.
- [ ] Optional, paid: deploy the Gradio version to Hugging Face Spaces (docs/DEPLOYMENT.md §3).
      A CPU Gradio Space needs a PRO/Team/Enterprise plan.

## Claims to keep accurate everywhere

- The tool estimates whether text *resembles* the fake or authentic articles in BanFakeNews-2.0.
  It does **not** verify news or replace fact-checking.
- The 0.956 macro F1 is in-distribution. Do not present it as real-world fake-news accuracy.
- BanglaBERT fine-tuning was **not** run at full scale. Make no transformer claims.
