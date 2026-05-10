"""
app.py — Smart Lecture Assistant (Gradio UI)
=============================================

Thin Gradio front-end. All ML/pipeline logic lives in `pipeline.py`.

Run:
    python app.py
"""

import gradio as gr

# ---------------------------------------------------------------------------
# Workaround for gradio_client schema bug (4.44.x):
#   When a callback's return type contains `dict`, gradio emits a JSON schema
#   with `additionalProperties: True`. The schema parser then does
#   `"const" in <bool>` and crashes with `TypeError: argument of type 'bool'
#   is not iterable`. Patch `get_type` to return "Any" for non-dict inputs.
# ---------------------------------------------------------------------------
import gradio_client.utils as _gcu

_orig_get_type = _gcu.get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)
_gcu.get_type = _safe_get_type

_orig_json_schema_to_python_type = _gcu._json_schema_to_python_type
def _safe_json_schema_to_python_type(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_json_schema_to_python_type(schema, defs)
_gcu._json_schema_to_python_type = _safe_json_schema_to_python_type

from pipeline import (
    run_whisper, build_index,
    generate_cheat_sheet, generate_takeaways, drill_down,
    download_audio_from_url, warm_up,
)


# =============================================================================
# Styling — bilingual UI: English chrome (LTR), Arabic content boxes (RTL).
# Applying RTL only to the elements that hold Arabic output (transcript, cheat
# sheet, takeaways, drill-down) keeps button/label text reading naturally
# left-to-right while Arabic still renders correctly.
# =============================================================================
APP_CSS = """
/* The actual Arabic content surfaces */
.arabic-text textarea,
.arabic-text .prose,
.arabic-text .markdown-body {
    direction: rtl !important;
    text-align: right !important;
    font-size: 1.02rem;
    line-height: 1.7;
}
.arabic-radio label > span:last-child {
    direction: rtl !important;
    text-align: right !important;
    unicode-bidi: plaintext;
    display: inline-block;
}

/* Hero header */
.hero {
    padding: 1.4rem 1.6rem;
    border-radius: 14px;
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    color: #fff;
    margin-bottom: 0.8rem;
}
.hero h1 { margin: 0 0 0.35rem 0; font-weight: 700; }
.hero p  { margin: 0; opacity: 0.92; }
.hero code {
    background: rgba(255,255,255,0.18);
    padding: 1px 6px;
    border-radius: 4px;
    font-size: 0.85em;
}

.section-header {
    font-weight: 600;
    margin: 0.2rem 0 0.4rem 0;
    border-bottom: 1px solid rgba(127,127,127,0.25);
    padding-bottom: 0.3rem;
}
"""


# =============================================================================
# Event handlers
# =============================================================================
def analyze_lecture(audio_path, url):
    """Top-level pipeline: (audio file OR YouTube URL) → ASR → index → cheat
       sheet → takeaways. Stuffs (chunks, index) into gr.State so drill-down
       can use them later without recomputing anything.

       If a URL is provided it takes precedence; otherwise the uploaded file
       is used. This way the user can switch between sources without clearing
       the other input."""
    if url and url.strip():
        try:
            audio_path = download_audio_from_url(url.strip())
        except Exception as e:
            empty_radio = gr.Radio(choices=[], value=None, interactive=True)
            return (f"⚠️ Failed to download video — تعذر تحميل الفيديو: {e}",
                    empty_radio, "", None, {}, "")

    if not audio_path:
        empty_radio = gr.Radio(choices=[], value=None, interactive=True)
        return ("⚠️ Please upload an audio file or paste a YouTube URL.",
                empty_radio, "", None, {}, "")

    transcript = run_whisper(audio_path)
    if not transcript:
        empty_radio = gr.Radio(choices=[], value=None, interactive=True)
        return ("⚠️ Speech recognition failed — check audio quality.",
                empty_radio, "", None, {}, "")

    chunks, index = build_index(transcript)
    cheat_md, pdf = generate_cheat_sheet(chunks, index)
    takeaways     = generate_takeaways(transcript)

    radio = gr.Radio(
        choices=takeaways,
        value=None,
        label="Key Takeaways — click one to jump to its source context",
        interactive=True,
        elem_classes=["arabic-radio"],
    )

    state_payload = {"chunks": chunks, "index": index}
    return cheat_md, radio, transcript, pdf, state_payload, ""


def on_takeaway_click(selected, state):
    """The 'magic trick' — instant drill-down on radio click."""
    if not state or not selected:
        return ""
    return drill_down(selected, state.get("chunks"), state.get("index"))


# =============================================================================
# UI layout
# =============================================================================
with gr.Blocks(title="Smart Lecture Assistant — Arabic",
               theme=gr.themes.Soft(),
               css=APP_CSS) as demo:

    gr.HTML(
        """
        <div class="hero">
            <h1>🎓 Smart Lecture Assistant — Arabic</h1>
            <p>
                Upload an Arabic lecture and get a transcript, a structured study
                guide, and one-click drill-down to the exact source segment.
            </p>
            <p style="margin-top:0.4rem; font-size:0.9em;">
                Pipeline:
                <code>Whisper (ASR)</code> →
                <code>CAMeL-BERT + FAISS (Semantic Search)</code> →
                <code>AraBART (Summarization)</code>
            </p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            audio_in = gr.Audio(sources=["upload", "microphone"],
                                type="filepath",
                                label="🎙️ Lecture audio")
            url_in = gr.Textbox(
                label="📹 …or paste a YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                lines=1,
            )
        with gr.Column(scale=1):
            analyze_btn = gr.Button("🔍 Analyze Lecture",
                                    variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📘 Study Guide", elem_classes=["section-header"])
            cheat_out = gr.Markdown(
                value="*The structured study guide will appear here after analysis.*",
                elem_classes=["arabic-text"],
            )
            pdf_out   = gr.File(label="⬇️ Download as PDF",
                                interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("### ✨ Key Takeaways", elem_classes=["section-header"])
            takeaways_radio = gr.Radio(
                choices=[], value=None,
                label="Click a takeaway to jump to its source context",
                interactive=True,
                elem_classes=["arabic-radio"],
            )
            drill_out = gr.Textbox(
                label="📝 Source context from the lecture",
                lines=8, interactive=False,
                placeholder="The original passage tied to the selected takeaway will appear here.",
                elem_classes=["arabic-text"],
            )

    with gr.Accordion("📜 Full Transcript", open=False):
        transcript_box = gr.Textbox(
            lines=12, interactive=False, show_label=False,
            elem_classes=["arabic-text"],
        )

    # Don't pass an empty dict as the State default — gradio_client's schema
    # introspection chokes on `additionalProperties: True` ("'bool' is not
    # iterable" in get_type). Initial value `None` produces a clean schema and
    # the consumer (`on_takeaway_click`) already guards against falsy state.
    state_payload = gr.State()

    analyze_btn.click(
        fn=analyze_lecture,
        inputs=[audio_in, url_in],
        outputs=[cheat_out, takeaways_radio, transcript_box,
                 pdf_out, state_payload, drill_out],
    )

    takeaways_radio.change(
        fn=on_takeaway_click,
        inputs=[takeaways_radio, state_payload],
        outputs=[drill_out],
    )


if __name__ == "__main__":
    # Front-load model loading so the first user click isn't slow.
    warm_up()
    # On HF Spaces, bind to 0.0.0.0 so the container's port is reachable.
    # Locally this also works — the browser hits 127.0.0.1 transparently.
    demo.launch(server_name="0.0.0.0", server_port=7860,
                share=False, show_error=True)
