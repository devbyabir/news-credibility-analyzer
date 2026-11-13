"""Gradio web demo for the News Credibility Analyzer.

Run locally:      python app.py            (http://127.0.0.1:7860)
Hugging Face:     this file is the Space entry point (see deploy/huggingface/).
Model location:   artifacts/model/ or the directory in $NCA_MODEL_DIR.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if (ROOT / "src").is_dir():  # allow running from a plain checkout without `pip install -e .`
    sys.path.insert(0, str(ROOT / "src"))

import gradio as gr  # noqa: E402

from nca.content import EXAMPLES, SUBTITLE, TITLE, about_markdown  # noqa: E402
from nca.content import NOTICE_HTML as NOTICE  # noqa: E402
from nca.predict import MAX_CHARS, InputError, Predictor  # noqa: E402

CSS = """
.gradio-container {max-width: 1100px !important; margin: auto;}
#hdr h1 {margin-bottom: 0.1rem;}
#hdr p {margin-top: 0; opacity: .8;}
.notice {border-left: 4px solid #d97706; background: rgba(217,119,6,.08); padding: 10px 14px;
         border-radius: 6px; font-size: .92rem; line-height: 1.45;}
.card {border: 1px solid rgba(127,127,127,.25); border-radius: 12px; padding: 18px 20px;}
.band {font-size: 1.25rem; font-weight: 650; margin: 0 0 .25rem;}
.muted {opacity: .75; font-size: .9rem;}
.meter {position: relative; height: 14px; border-radius: 7px; margin: 14px 0 4px;
        background: linear-gradient(90deg, #15803d 0%, #65a30d 25%, #a3a3a3 50%, #ea580c 75%, #b91c1c 100%);}
.meter .pin {position: absolute; top: -5px; width: 4px; height: 24px; background: currentColor;
             border-radius: 2px; transform: translateX(-2px);}
.scale {display: flex; justify-content: space-between; font-size: .78rem; opacity: .7;}
.chips span {display: inline-block; margin: 3px 4px 0 0; padding: 2px 9px; border-radius: 999px;
             font-size: .85rem; border: 1px solid rgba(127,127,127,.35);}
.chips.fake span {background: rgba(185,28,28,.09);}
.chips.auth span {background: rgba(21,128,61,.09);}
.warn {background: rgba(234,179,8,.13); border-radius: 6px; padding: 8px 12px; margin-top: 12px; font-size: .9rem;}
.err {border-left: 4px solid #b91c1c; background: rgba(185,28,28,.08); padding: 12px 14px; border-radius: 6px;}
"""

_PREDICTOR: Predictor | None = None
_LOAD_ERROR: str | None = None


def get_predictor() -> Predictor | None:
    global _PREDICTOR, _LOAD_ERROR
    if _PREDICTOR is None and _LOAD_ERROR is None:
        try:
            _PREDICTOR = Predictor.load(os.environ.get("NCA_MODEL_DIR"))
        except Exception as exc:  # surfaced in the UI instead of crashing the app
            _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    return _PREDICTOR


def _error(msg: str) -> str:
    return f'<div class="err">{html.escape(msg)}</div>'


def _chips(terms, cls: str) -> str:
    if not terms:
        return '<span class="muted">none</span>'
    inner = "".join(f"<span>{html.escape(t)}</span>" for t, _ in terms)
    return f'<div class="chips {cls}">{inner}</div>'


def analyze(text: str) -> str:
    predictor = get_predictor()
    if predictor is None:
        return _error(f"The model could not be loaded, so no analysis is possible. ({_LOAD_ERROR})")
    try:
        r = predictor.predict(text)
    except InputError as exc:
        return _error(str(exc))
    except Exception as exc:  # pragma: no cover - defensive
        return _error(f"Unexpected error while analysing the text: {type(exc).__name__}")

    pct = r.p_fake * 100
    if r.band_test_fake_rate is not None and r.band_test_n:
        band_stat = (
            f"On the held-out test set, <b>{r.band_test_fake_rate * 100:.0f}%</b> of the "
            f"{r.band_test_n:,} articles that fell in this band were labelled fake. That figure "
            "describes text like the training data; it may not hold for articles from other "
            "periods, portals or topics."
        )
    else:
        band_stat = ""
    warnings = "".join(f'<div class="warn">⚠ {html.escape(w)}</div>' for w in r.warnings)
    return f"""
<div class="card">
  <div class="muted">Model assessment</div>
  <p class="band">{html.escape(r.band_text)}</p>
  <div class="meter"><div class="pin" style="left:{pct:.1f}%"></div></div>
  <div class="scale"><span>authentic-like</span><span>uncertain</span><span>fake-like</span></div>
  <p class="muted" style="margin-top:10px">Model score P(fake) = <b>{r.p_fake:.2f}</b>. This is a
  raw model score, not a calibrated probability that the story is false. {band_stat}</p>
  {warnings}
  <hr style="opacity:.2;margin:14px 0">
  <div class="muted">Words and phrases that pushed the score towards <b>fake-like</b></div>
  {_chips(r.top_fake_terms, "fake")}
  <div class="muted" style="margin-top:10px">…and towards <b>authentic-like</b></div>
  {_chips(r.top_authentic_terms, "auth")}
</div>"""


def _about_md() -> str:
    p = get_predictor()
    return about_markdown(p.metadata) if p is not None else "Model metadata unavailable."


def build_demo() -> gr.Blocks:
    # analytics_enabled=False: no telemetry, and no background version-check / analytics threads
    # (they block startup and process exit on restricted networks, e.g. in CI or sandboxes).
    with gr.Blocks(title=TITLE, analytics_enabled=False) as demo:
        gr.HTML(f'<div id="hdr"><h1>📰 {TITLE}</h1><p>{SUBTITLE}</p></div>')
        gr.HTML(f'<div class="notice">{NOTICE}</div>')
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                inp = gr.Textbox(
                    label="Bangla news text",
                    placeholder="এখানে বাংলা সংবাদের পূর্ণ লেখা পেস্ট করুন…",
                    lines=12,
                    max_lines=24,
                    max_length=MAX_CHARS,
                )
                with gr.Row():
                    clear = gr.Button("Clear", variant="secondary")
                    btn = gr.Button("Analyze", variant="primary")
                gr.Examples(
                    EXAMPLES, inputs=inp, label="Illustrative examples (written for this demo, not real news)"
                )
            with gr.Column(scale=5):
                out = gr.HTML('<div class="card muted">The assessment will appear here.</div>')
        with gr.Accordion("How it works, results and limitations", open=False):
            gr.Markdown(_about_md())
        btn.click(analyze, inputs=inp, outputs=out, api_name="analyze")
        inp.submit(analyze, inputs=inp, outputs=out)
        clear.click(
            lambda: ("", '<div class="card muted">The assessment will appear here.</div>'), outputs=[inp, out]
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(
            primary_hue="indigo",
            font=[gr.themes.GoogleFont("Noto Sans Bengali"), "system-ui", "sans-serif"],
        ),
        css=CSS,
    )
