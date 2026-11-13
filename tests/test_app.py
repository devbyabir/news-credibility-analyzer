"""Smoke tests for the Gradio app's request handler (no server needed)."""

import importlib

import pytest


@pytest.fixture()
def app_module(monkeypatch, tiny_model_dir):
    monkeypatch.setenv("NCA_MODEL_DIR", str(tiny_model_dir))
    import app

    app = importlib.reload(app)
    app._PREDICTOR, app._LOAD_ERROR = None, None
    return app


def test_app_builds_and_analyzes(app_module):
    assert app_module.demo is not None
    html = app_module.analyze(app_module.EXAMPLES[1])
    assert "Model assessment" in html and "P(fake)" in html


def test_app_reports_input_errors(app_module):
    assert "paste" in app_module.analyze("")
    assert "Bangla" in app_module.analyze("Only English text here, nothing in Bangla script at all.")


def test_app_survives_missing_model(monkeypatch, tmp_path):
    monkeypatch.setenv("NCA_MODEL_DIR", str(tmp_path))
    import app

    app = importlib.reload(app)
    app._PREDICTOR, app._LOAD_ERROR = None, None
    out = app.analyze("ঢাকায় আজ বৃষ্টি হয়েছে এবং সড়কে যানজট দেখা দিয়েছে বলে জানা গেছে")
    assert "could not be loaded" in out


def test_app_escapes_html(app_module):
    out = app_module.analyze("<script>alert(1)</script> " + app_module.EXAMPLES[0])
    assert "<script>" not in out


def test_app_does_not_start_gradio_telemetry_threads(app_module):
    # Gradio's version check and analytics run in non-daemon threads that make network calls;
    # on restricted networks they stalled test runs and process exit. The demo disables them.
    import threading

    assert app_module.demo.analytics_enabled is False
    names = {t.name for t in threading.enumerate()}
    assert not any("version_check" in n or "analytics" in n for n in names)
