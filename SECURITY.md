# Security policy

## Supported version

| Version | Supported |
|---|---|
| 1.0.x (current `main`, release `v1.0.0`) | Yes |

## Reporting a vulnerability

Please do **not** open a public issue for a security problem. Use GitHub's private
[vulnerability reporting](https://github.com/devbyabir/news-credibility-analyzer/security/advisories/new)
for this repository. Include the affected file or URL, the steps to reproduce it, and the impact.
You should get a reply within 7 days.

## Design notes

- **Model loading.** `joblib.load` can execute code, so hosted demos load a model only after its
  SHA-256 matches the value committed in `deploy/streamlit/model_source.json`
  (`src/nca/remote.py`). Downloads must use HTTPS and are capped at 100 MB.
- **Untrusted input.** User text is length-checked and every value shown in the demo's HTML is
  escaped. Nothing a visitor types is stored or logged by the app.
- **Secrets.** The project needs no API keys. An optional `NCA_MODEL_BASE_URL` may be set as a
  Streamlit secret; never commit credentials.
- **Dependencies.** Versions are pinned and checked with `pip-audit`. Dependabot security alerts are
  enabled. scikit-learn must stay at the version the model was trained with (1.9.1).
