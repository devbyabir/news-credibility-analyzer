"""Headless run of the Streamlit demo with streamlit.testing (no browser, no network)."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).parents[1] / "deploy" / "streamlit" / "streamlit_app.py")


def _markdown(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_streamlit_app_analyzes_example(monkeypatch, tiny_model_dir):
    monkeypatch.setenv("NCA_MODEL_DIR", str(tiny_model_dir))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.button(key="ex1").click().run()  # load the sensational example
    next(b for b in at.button if b.label == "Analyze").click().run()
    assert not at.exception
    assert "Model assessment" in _markdown(at)


def test_streamlit_app_rejects_empty_input(monkeypatch, tiny_model_dir):
    monkeypatch.setenv("NCA_MODEL_DIR", str(tiny_model_dir))
    at = AppTest.from_file(APP, default_timeout=60).run()
    next(b for b in at.button if b.label == "Analyze").click().run()
    assert any("paste" in e.value for e in at.error)


def test_streamlit_app_reports_missing_model(monkeypatch, tmp_path):
    # No local model, an empty download cache and a download URL the app must refuse (not https).
    # This stays offline and deterministic now that model_source.json points at a real release.
    monkeypatch.setenv("NCA_MODEL_DIR", str(tmp_path / "no_model"))
    monkeypatch.setenv("NCA_MODEL_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("NCA_MODEL_BASE_URL", "http://insecure.invalid/models")
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert any("could not be loaded" in e.value for e in at.error)


def test_streamlit_app_rejects_excessive_input_explicitly(monkeypatch, tiny_model_dir):
    from nca.predict import MAX_CHARS

    monkeypatch.setenv("NCA_MODEL_DIR", str(tiny_model_dir))
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.text_area(key="text").input("ক" * (MAX_CHARS + 5000)).run()
    next(b for b in at.button if b.label == "Analyze").click().run()
    assert not at.exception
    assert any("too long" in e.value for e in at.error)
    assert "Model assessment" not in _markdown(at)
