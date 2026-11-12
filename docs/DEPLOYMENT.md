# Deployment

There are two equivalent web demos, both using the same 7 MB scikit-learn model on CPU:

| Demo | Entry point | Free hosting |
|---|---|---|
| Streamlit | `deploy/streamlit/streamlit_app.py` | **Streamlit Community Cloud** (section 2) |
| Gradio | `app.py` | Hugging Face Spaces (section 3) |

No paid service, GPU or API key is required. Both hosting options need your own account,
so you do the final publishing steps yourself.

## 1. Local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
python -m nca.data download     # ~305 MB, checksum-verified
python -m nca.train             # robust config -> artifacts/model/ and reports/robust/
python app.py                   # Gradio:    http://127.0.0.1:7860
streamlit run deploy/streamlit/streamlit_app.py   # Streamlit: http://localhost:8501
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `NCA_MODEL_DIR` | `artifacts/model` | Folder containing `model.joblib` and `metadata.json` |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Use `0.0.0.0` inside a container |
| `GRADIO_SERVER_PORT` | `7860` | Gradio port |
| `NCA_MODEL_BASE_URL` | from `deploy/streamlit/model_source.json` | Streamlit only: where to download the model when `artifacts/model` is absent |

## 2. Streamlit Community Cloud (needs your GitHub + Streamlit accounts)

Streamlit Cloud deploys straight from your GitHub repository. The model file is kept out of Git,
so the app downloads it once from a **GitHub Release** of your repository. It then checks the
SHA-256 against `deploy/streamlit/model_source.json` before loading it. This check matters:
loading a model file can run code, so an unverified download is never loaded.

1. **Push the repository to GitHub** (see `docs/RELEASE_CHECKLIST.md`). It can be public or
   private, but the release asset must be downloadable without a login, so a **public** repo is
   simplest.
2. **Point the app at your release.** In `deploy/streamlit/model_source.json`, replace
   `<your-github-username>` with your GitHub username. Commit and push.
3. **Create the release.** On GitHub, open the repository → *Releases* → *Draft a new release* →
   tag `v1.0.0` → attach these two files from your computer:
   - `artifacts/model/model.joblib`
   - `artifacts/model/metadata.json`

   Then click *Publish release*. The download URLs will be
   `https://github.com/<you>/news-credibility-analyzer/releases/download/v1.0.0/model.joblib`
   (and the same for `metadata.json`).
4. **Deploy.** Sign in at https://share.streamlit.io with GitHub → *Create app* →
   *Deploy a public app from GitHub* and fill in:
   - Repository: `<you>/news-credibility-analyzer`, branch `main`
   - Main file path: `deploy/streamlit/streamlit_app.py`
   - App URL: choose a subdomain, e.g. `news-credibility-analyzer`
   - *Advanced settings* → Python version **3.12**

   Click *Deploy*. Streamlit installs `deploy/streamlit/requirements.txt` (the requirements file
   next to the entry point) and starts the app. The first visit downloads the model (~7 MB).
5. **Test** the public URL in a private browser window: run both examples, an empty input and
   an English text. Then put the URL in the README "Live demo" line and on your portfolio/CV.

If the app shows *"The model could not be loaded"*, the message says why: the URL placeholder was
not replaced, the release is missing, the repo is private, or the checksum does not match.

**After retraining:** run `python scripts/model_checksums.py`, commit the updated
`model_source.json`, and upload the new files to a new release tag (update `base_url` to match).
A model trained with a different scikit-learn version must not be deployed with the pinned
requirements unchanged.

**Alternative without a release:** set the app secret `NCA_MODEL_BASE_URL` (Streamlit Cloud →
*App settings* → *Secrets*: `NCA_MODEL_BASE_URL = "https://..."`) to any HTTPS folder that serves
the two files. The checksums in `model_source.json` still apply.

## 3. Hugging Face Space (needs your account)

These steps need your own Hugging Face login and they publish the demo, so you run them yourself.

1. Create a free account at https://huggingface.co and an access token with **write** scope
   (Settings → Access Tokens). Never commit the token.
2. Build the Space folder from your trained model:
   ```bash
   python scripts/prepare_space.py
   ```
   This creates `build/hf_space/` with `app.py`, the runtime part of `src/nca`,
   pinned `requirements.txt`, the Space card (`README.md`) and the model files.
3. Edit `build/hf_space/README.md` and replace `<your-github-username>` with your GitHub username.
4. Log in, create the Space (free `cpu-basic` hardware) and upload
   (commands verified against `huggingface_hub` 2.4's `hf` CLI):
   ```bash
   hf auth login
   hf repos create <your-hf-username>/news-credibility-analyzer --repo-type space --space-sdk gradio --public
   hf upload <your-hf-username>/news-credibility-analyzer build/hf_space . --repo-type space
   ```
5. Wait for the build (2–4 minutes). The Space will be available at
   `https://huggingface.co/spaces/<your-hf-username>/news-credibility-analyzer`.
6. Put that URL in the root `README.md` (the "Live demo" line) and on your portfolio.

The model file is derived from BanFakeNews-2.0 (CC BY 4.0), so the Space card keeps the
dataset attribution. Keep it there.

### Updating the Space
Retrain, run `scripts/prepare_space.py` again, and repeat the `hf upload` command.
Keep `scikit-learn` identical in `requirements.txt` and `deploy/huggingface/requirements.txt`.
The script refuses to build if the model was trained with a different version.

## 4. Any other host (Render, Railway, a VM, Docker)

The Space folder is a plain Python app. On any host:

```bash
pip install -r requirements.txt
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=$PORT python app.py
```
