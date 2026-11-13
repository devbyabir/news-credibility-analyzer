"""Streamlit web demo for the News Credibility Analyzer.

Run locally (from the repository root):
    streamlit run deploy/streamlit/streamlit_app.py

Streamlit Community Cloud: set "Main file path" to deploy/streamlit/streamlit_app.py.
The app uses artifacts/model/ when it exists (local runs). Otherwise it downloads
the model files from the GitHub Release in deploy/streamlit/model_source.json
(or $NCA_MODEL_BASE_URL) and verifies their SHA-256 before loading them.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

import streamlit as st

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nca.content import EXAMPLES, NOTICE_HTML, SUBTITLE, TITLE, about_markdown  # noqa: E402
from nca.predict import MAX_CHARS, InputError, Prediction, Predictor  # noqa: E402
from nca.remote import ensure_model, load_source  # noqa: E402

st.set_page_config(page_title=TITLE, page_icon="📰", layout="wide")

CSS = """
<style>
.block-container {max-width: 1150px; padding-top: 2.2rem;}
h1 {font-size: clamp(1.7rem, 5.5vw, 2.6rem) !important;}
.nca-sub {opacity: .75; margin-top: -0.6rem; margin-bottom: 1rem;}
.nca-notice {border-left: 4px solid #d97706; background: rgba(217,119,6,.10); padding: 10px 14px;
             border-radius: 6px; font-size: .93rem; line-height: 1.5; margin-bottom: 1.2rem;}
.nca-card {border: 1px solid rgba(127,127,127,.28); border-radius: 12px; padding: 18px 20px;}
.nca-band {font-size: 1.3rem; font-weight: 650; margin: .1rem 0 .3rem;}
.nca-muted {opacity: .75; font-size: .9rem;}
.nca-meter {position: relative; height: 14px; border-radius: 7px; margin: 14px 0 4px;
  background: linear-gradient(90deg,#15803d 0%,#65a30d 25%,#a3a3a3 50%,#ea580c 75%,#b91c1c 100%);}
.nca-pin {position: absolute; top: -5px; width: 4px; height: 24px; background: currentColor;
          border-radius: 2px; transform: translateX(-2px);}
.nca-scale {display: flex; justify-content: space-between; font-size: .78rem; opacity: .7;}
.nca-chips span {display: inline-block; margin: 4px 5px 0 0; padding: 2px 10px; border-radius: 999px;
                 font-size: .88rem; border: 1px solid rgba(127,127,127,.35);}
.nca-chips.fake span {background: rgba(185,28,28,.10);}
.nca-chips.auth span {background: rgba(21,128,61,.10);}
.nca-warn {background: rgba(234,179,8,.15); border-radius: 6px; padding: 8px 12px; margin-top: 12px;
           font-size: .9rem;}
</style>
"""


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(local_dir: str, base_url: str) -> Predictor:
    """Load the local model if present, else download + verify it. Cached per location."""
    local = Path(local_dir)
    if (local / "model.joblib").exists():
        return Predictor.load(local)
    source = load_source(HERE / "model_source.json")
    model_dir = ensure_model(ROOT / ".model_cache", base_url or source["base_url"], source["sha256"])
    return Predictor.load(model_dir)


def _secret(key: str) -> str | None:
    try:
        return st.secrets.get(key)
    except Exception:  # no secrets file configured
        return None


def _chips(terms, cls: str) -> str:
    if not terms:
        return '<span class="nca-muted">none</span>'
    inner = "".join(f"<span>{html.escape(t)}</span>" for t, _ in terms)
    return f'<div class="nca-chips {cls}">{inner}</div>'


def result_html(r: Prediction) -> str:
    pct = r.p_fake * 100
    band_stat = ""
    if r.band_test_fake_rate is not None and r.band_test_n:
        band_stat = (
            f"On the held-out test set, <b>{r.band_test_fake_rate * 100:.0f}%</b> of the "
            f"{r.band_test_n:,} articles that fell in this band were labelled fake. That figure "
            "describes text like the training data; it may not hold for articles from other "
            "periods, portals or topics."
        )
    warnings = "".join(f'<div class="nca-warn">⚠ {html.escape(w)}</div>' for w in r.warnings)
    return f"""
<div class="nca-card">
  <div class="nca-muted">Model assessment</div>
  <div class="nca-band">{html.escape(r.band_text)}</div>
  <div class="nca-meter"><div class="nca-pin" style="left:{pct:.1f}%"></div></div>
  <div class="nca-scale"><span>authentic-like</span><span>uncertain</span><span>fake-like</span></div>
  <p class="nca-muted" style="margin-top:10px">Model score P(fake) = <b>{r.p_fake:.2f}</b>. This is a
  raw model score, not a calibrated probability that the story is false. {band_stat}</p>
  {warnings}
  <hr style="opacity:.2;margin:14px 0">
  <div class="nca-muted">Words and phrases that pushed the score towards <b>fake-like</b></div>
  {_chips(r.top_fake_terms, "fake")}
  <div class="nca-muted" style="margin-top:10px">…and towards <b>authentic-like</b></div>
  {_chips(r.top_authentic_terms, "auth")}
</div>"""


def _set_text(value: str) -> None:
    st.session_state.text = value
    st.session_state.pop("result", None)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(f"📰 {TITLE}")
    st.markdown(f'<div class="nca-sub">{SUBTITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="nca-notice">{NOTICE_HTML}</div>', unsafe_allow_html=True)

    try:
        predictor = load_predictor(
            os.environ.get("NCA_MODEL_DIR") or str(ROOT / "artifacts" / "model"),
            os.environ.get("NCA_MODEL_BASE_URL") or _secret("NCA_MODEL_BASE_URL") or "",
        )
        load_error = None
    except Exception as exc:  # shown in the UI instead of crashing the app
        predictor, load_error = None, f"{type(exc).__name__}: {exc}"

    st.session_state.setdefault("text", "")
    left, right = st.columns([6, 5], gap="large")
    with left:
        # No max_chars: the widget would silently cut long pastes. Length is checked by
        # nca.predict.validate_text, which shows an explicit "too long" message instead.
        st.text_area(
            "Bangla news text",
            key="text",
            height=300,
            help=f"Paste a full article (up to {MAX_CHARS:,} characters).",
            placeholder="এখানে বাংলা সংবাদের পূর্ণ লেখা পেস্ট করুন…",
        )
        b1, b2 = st.columns(2)
        analyze = b1.button("Analyze", type="primary", use_container_width=True)
        b2.button("Clear", use_container_width=True, on_click=_set_text, args=("",))
        st.caption("Illustrative examples (written for this demo, not real news)")
        for i, ex in enumerate(EXAMPLES):
            st.button(ex[:60] + "…", key=f"ex{i}", on_click=_set_text, args=(ex,), use_container_width=True)

    with right:
        if load_error:
            st.error(f"The model could not be loaded, so no analysis is possible. ({load_error})")
        elif analyze:
            try:
                st.session_state.result = predictor.predict(st.session_state.text)
            except InputError as exc:
                st.session_state.pop("result", None)
                st.error(str(exc))
        if "result" in st.session_state and not load_error:
            st.markdown(result_html(st.session_state.result), unsafe_allow_html=True)
        elif not analyze and not load_error:
            st.markdown(
                '<div class="nca-card nca-muted">The assessment will appear here.</div>',
                unsafe_allow_html=True,
            )

    with st.expander("How it works, results and limitations"):
        st.markdown(about_markdown(predictor.metadata) if predictor else "Model metadata unavailable.")


main()
