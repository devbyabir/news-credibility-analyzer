# Deployment

The demo is a single Gradio app (`app.py`) plus a ~10 MB scikit-learn model. It runs on
CPU, so the free **Hugging Face Spaces** "CPU basic" tier is enough. No paid service,
GPU or API key is required.

## 1. Local

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-deps
python -m nca.data download     # ~305 MB, checksum-verified
python -m nca.train             # writes artifacts/model/ and reports/
python app.py                   # http://127.0.0.1:7860
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `NCA_MODEL_DIR` | `artifacts/model` | Folder containing `model.joblib` and `metadata.json` |
| `GRADIO_SERVER_NAME` | `127.0.0.1` | Use `0.0.0.0` inside a container |
| `GRADIO_SERVER_PORT` | `7860` | Port |

## 2. Hugging Face Space (needs your account)

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

## 3. Any other host (Render, Railway, a VM, Docker)

The Space folder is a plain Python app. On any host:

```bash
pip install -r requirements.txt
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=$PORT python app.py
```
