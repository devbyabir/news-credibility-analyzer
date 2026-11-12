# Release checklist

Work through this list before pushing to GitHub or updating the live demo.

## Repository hygiene
- [ ] `git status` is clean and `git log` shows only your configured author identity.
- [ ] No dataset, model or cache files are tracked:
      `git ls-files | grep -Ei '\.(csv|joblib|pkl|pt|bin|safetensors)$' | grep -v '^reports/'` prints nothing
      (the small CV result CSVs in `reports/` are intentional).
- [ ] No tracked file is larger than 5 MB:
      `git ls-files -z | xargs -0 du -k | sort -n | tail -5`.
- [ ] No secrets: `git grep -nIiE "(hf_[A-Za-z0-9]{20,}|api[_-]?key|secret|password|token\s*=)"`
      shows only documentation text.
- [ ] `LICENSE` holder and year are correct, and you have decided the code license (see README).

## Quality gates
- [ ] `ruff check . && ruff format --check .`
- [ ] `pytest` (no dataset needed)
- [ ] The full pipeline reproduces from scratch:
      `python -m nca.data download && python -m nca.train`. Compare the new
      `reports/test_results.json` with the committed one.
- [ ] `python app.py` and `streamlit run deploy/streamlit/streamlit_app.py` start. Both illustrative examples return a result.
      Empty, English-only and over-long input each show a clear message.

## Documentation
- [ ] README metrics match `reports/test_results.json` exactly.
- [ ] README "Live demo" line points to the deployed Space, or still says "not yet deployed".
- [ ] `<your-github-username>` placeholders are replaced in the README, the Space card and `docs/CV.md`.
- [ ] Dataset citation (Mendeley DOI, CC BY 4.0) is present in README, MODEL_CARD and the Space card.

## Publish (you do these; they need your accounts)
- [ ] Create an empty GitHub repository `news-credibility-analyzer`. Do not add a README, license or `.gitignore` there.
- [ ] `git remote add origin https://github.com/<you>/news-credibility-analyzer.git && git push -u origin main`
- [ ] Check that the GitHub Actions CI run passes.
- [ ] Streamlit: replace the username in `deploy/streamlit/model_source.json`, publish release `v1.0.0`
      with `model.joblib` + `metadata.json`, and deploy `deploy/streamlit/streamlit_app.py` (docs/DEPLOYMENT.md §2).
- [ ] Or/also deploy the Hugging Face Space (docs/DEPLOYMENT.md §3).
- [ ] Test the public URL in a private browser window.
- [ ] Add repository topics: `bangla`, `nlp`, `fake-news-detection`, `text-classification`, `scikit-learn`, `gradio`.
