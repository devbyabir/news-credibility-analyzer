# Deployment

There are two web front-ends for the same 7.0 MB scikit-learn model. Both run on CPU:

| Demo | Entry point | Recommended host | Cost (verified 13 Nov 2026, re-check before deploying) |
|---|---|---|---|
| Streamlit | `deploy/streamlit/streamlit_app.py` | **Streamlit Community Cloud** (primary route, §2) | Free for public apps |
| Gradio | `app.py` | Hugging Face Spaces (optional, §3) | Needs a **paid** Hugging Face plan for a CPU Gradio Space (see §3) |

**Live deployment of v1.0.0:** <https://bangla-news-credibility-analyzer.streamlit.app>

No GPU or API key is needed. Every hosting option needs your own accounts, and publishing is
done by you, not by this repository's tooling.

## 1. Local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
python -m nca.data download     # ~305 MB, checksum-verified
python -m nca.train             # robust config -> artifacts/model/ and reports/robust/
streamlit run deploy/streamlit/streamlit_app.py   # Streamlit: http://localhost:8501
python app.py                   # Gradio:    http://127.0.0.1:7860
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `NCA_MODEL_DIR` | `artifacts/model` | Folder containing `model.joblib` and `metadata.json` |
| `NCA_MODEL_BASE_URL` | `base_url` in `deploy/streamlit/model_source.json` | Streamlit only: HTTPS folder to download the model from when `NCA_MODEL_DIR` has no model |
| `NCA_MODEL_CACHE` | `.model_cache/` | Streamlit only: where downloaded, verified model files are kept |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Gradio only: use `0.0.0.0` inside a container |
| `GRADIO_SERVER_PORT` | `7860` | Gradio only: port |

## 2. Streamlit Community Cloud (primary route)

**Terms.** These points come from Streamlit's official pages, checked on 13 Nov 2026:
- Community Cloud is described as "totally free" for community apps.
- It hosts public apps only.
- Deploying needs *admin* permission on the GitHub repository.
- Apps with no traffic for 12 hours go to sleep and wake on the next visit.
- Each app gets up to about 2.7 GB of memory, far more than this app needs.

Sources:
- <https://streamlit.io/cloud>
- <https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account>
- <https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app>

**How this repository fits Community Cloud** (also checked against the official docs):
- **Main file path.** `deploy/streamlit/streamlit_app.py`. Entrypoint files "can be anywhere in your
  repository", and the working directory is always the repository root
  ([file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)).
  The app builds all of its paths from its own location, so it works either way.
- **Dependencies.** Community Cloud searches the entrypoint's directory first, then the repository
  root, and uses the first dependency file it finds
  ([app dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)).
  It therefore installs `deploy/streamlit/requirements.txt` (5 pinned runtime packages), not the
  root `requirements.txt` or `pyproject.toml`.
- **Theme.** `.streamlit/config.toml` is at the repository root, as the docs require when the
  entrypoint is in a subdirectory.
- **Python.** The default is 3.12 ([deploy guide](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)).
  Select **3.12** under *Advanced settings* anyway, because all pinned packages were checked for 3.12.
- **Model.** The model file is not in Git. On first start the app downloads `model.joblib` and
  `metadata.json` from a GitHub Release into `.model_cache/` and verifies both SHA-256 values
  against `deploy/streamlit/model_source.json` before loading. Loading a pickle can execute code,
  so an unverified file is never loaded. Release assets of a **private** repository need a login
  to download, so the repository must be **public** for this route.

### Steps (you do these: they use your accounts)

1. **Push the code** to a public GitHub repository named `news-credibility-analyzer`.
   See `docs/RELEASE_CHECKLIST.md`, part A.
2. **Set the download URL.** In `deploy/streamlit/model_source.json`, replace
   `<your-github-username>` in `base_url` with your GitHub username. Commit and push.
3. **Publish the model release.** On GitHub, open *Releases* → *Draft a new release* →
   *Choose a tag* → type `v1.0.0` → *Create new tag*.
   - Attach exactly these two files, keeping their names: `artifacts/model/model.joblib` and
     `artifacts/model/metadata.json`.
   - Then *Publish release*.
   - The app will download from
     `https://github.com/<your-github-username>/news-credibility-analyzer/releases/download/v1.0.0/model.joblib`
     (and `…/metadata.json`).
4. **Deploy.** Sign in at <https://share.streamlit.io> with GitHub and create a new app from your
   repository:
   - Repository: `<your-github-username>/news-credibility-analyzer`
   - Branch: `main`
   - Main file path: `deploy/streamlit/streamlit_app.py`
   - App URL: pick a free subdomain. This project uses `bangla-news-credibility-analyzer`;
     `news-credibility-analyzer` was already taken.
   - *Advanced settings* → Python **3.12**

   Deploy, then wait for the build to finish (the log is shown while it runs).
5. **Smoke-test the public URL** in a private browser window. Check that:
   - both illustrative examples return a result;
   - empty input shows "Please paste some Bangla news text.";
   - English-only text shows "This model only supports Bangla text…";
   - the "How it works" panel shows test macro F1 0.956.
6. **Publish the link.** Put the URL in the README "Live demo" line, in `docs/CV.md` and on your
   portfolio.

If the app shows *"The model could not be loaded"*, the message gives the reason. The usual causes:
- the username placeholder is still in `model_source.json`;
- the release or one of the two assets is missing;
- the repository is private;
- a checksum mismatch, because a different file was uploaded.

**After retraining:**
1. Run `python scripts/model_checksums.py`.
2. Commit the updated `model_source.json`.
3. Publish the new files under a new tag, and update `base_url` to that tag.

Keep `scikit-learn` identical in all requirements files: the model is a scikit-learn 1.9.1 pickle.

**Alternative download location:** add the app secret `NCA_MODEL_BASE_URL = "https://…"`
(*App settings* → *Secrets*). It can point to any HTTPS folder that serves the two files. The
committed checksums still apply.

## 3. Hugging Face Spaces (optional, needs a paid plan for CPU)

This route is ready, but read the cost first. Hugging Face's
[Spaces overview](https://huggingface.co/docs/hub/spaces-overview) (checked 13 Nov 2026) says
"Gradio and Docker Spaces run on compute and require a paid plan to create: PRO for personal
accounts, Team or Enterprise for organizations." The default CPU Basic hardware has no hourly
charge, but a free account cannot create a Gradio Space on it. The [pricing page](https://huggingface.co/pricing)
lists PRO at $9/month.

There is one free exception: personal accounts with a verified e-mail that are older than 30 days
can host up to 2 Gradio Spaces on **ZeroGPU** ([ZeroGPU docs](https://huggingface.co/docs/hub/spaces-zerogpu)).
ZeroGPU is designed for PyTorch GPU workloads. This CPU-only scikit-learn app has **not** been
tested on ZeroGPU, so that option is unverified. Static Spaces are free but cannot run Python.

If you have a plan that allows a CPU Gradio Space:

1. Create a Hugging Face access token with **write** scope (*Settings → Access Tokens*).
   Never commit it.
2. Build the Space folder:
   ```bash
   python scripts/prepare_space.py
   ```
   This creates `build/hf_space/` with `app.py`, the runtime part of `src/nca`, pinned
   `requirements.txt`, the Space card and the model files.
3. In `build/hf_space/README.md`, replace `<your-github-username>`.
4. Log in, create the Space and upload. These commands were checked against the `hf` CLI in
   `huggingface_hub` 2.4. Creation fails on a free account unless your plan allows it:
   ```bash
   hf auth login
   hf repos create <your-hf-username>/news-credibility-analyzer --repo-type space --space-sdk gradio --public
   hf upload <your-hf-username>/news-credibility-analyzer build/hf_space . --repo-type space
   ```
5. After the build, the Space is at
   `https://huggingface.co/spaces/<your-hf-username>/news-credibility-analyzer`. Spaces on free
   hardware sleep when unused.

The model file is derived from BanFakeNews-2.0 (CC BY 4.0), so the Space card keeps the dataset
attribution. To update the Space, retrain, run `scripts/prepare_space.py` again and repeat
`hf upload`. The script refuses to build if the model's scikit-learn version differs from
`deploy/huggingface/requirements.txt`.

## 4. Any other host (Render, Railway, a VM, Docker)

Both apps are plain Python. With the model in `artifacts/model/` (or `NCA_MODEL_DIR`):

```bash
pip install -r requirements.txt
streamlit run deploy/streamlit/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
# or: GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=$PORT python app.py
```

Check the host's own pricing; none of these was tested for this project.
