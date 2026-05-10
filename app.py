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
# Custom Gradio theme — "Manuscript": deep midnight indigo + warm amber accent,
# inspired by classical Arabic manuscript color palettes (ink + gold leaf).
# =============================================================================
MANUSCRIPT_THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.amber,        # buttons, accents -> warm gold
    secondary_hue=gr.themes.colors.indigo,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "Consolas"],
).set(
    body_background_fill="#0b1020",
    body_background_fill_dark="#0b1020",
    body_text_color="#e7ecf3",
    body_text_color_dark="#e7ecf3",
    background_fill_primary="rgba(255,255,255,0.04)",
    background_fill_primary_dark="rgba(255,255,255,0.04)",
    background_fill_secondary="rgba(255,255,255,0.02)",
    background_fill_secondary_dark="rgba(255,255,255,0.02)",
    border_color_primary="rgba(255,255,255,0.08)",
    border_color_primary_dark="rgba(255,255,255,0.08)",
    block_background_fill="rgba(255,255,255,0.035)",
    block_background_fill_dark="rgba(255,255,255,0.035)",
    block_border_color="rgba(255,255,255,0.07)",
    block_border_color_dark="rgba(255,255,255,0.07)",
    block_border_width="1px",
    block_radius="14px",
    block_shadow="0 4px 24px -8px rgba(0,0,0,0.45)",
    block_label_background_fill="transparent",
    block_label_text_color="#cbd5e1",
    block_title_text_color="#f5e8c8",
    input_background_fill="rgba(255,255,255,0.045)",
    input_background_fill_dark="rgba(255,255,255,0.045)",
    input_border_color="rgba(255,255,255,0.08)",
    input_border_color_dark="rgba(255,255,255,0.08)",
    input_border_color_focus="#f59e0b",
    button_primary_background_fill="linear-gradient(135deg,#f59e0b 0%,#ef4444 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg,#fbbf24 0%,#f87171 100%)",
    button_primary_text_color="#1c1207",
    button_primary_border_color="transparent",
    button_secondary_background_fill="rgba(255,255,255,0.06)",
    button_secondary_background_fill_hover="rgba(255,255,255,0.10)",
    button_secondary_text_color="#e7ecf3",
)


# =============================================================================
# Styling — bilingual UI: English chrome (LTR), Arabic content boxes (RTL).
# Applying RTL only to the elements that hold Arabic output (transcript, cheat
# sheet, takeaways, drill-down) keeps button/label text reading naturally
# left-to-right while Arabic still renders correctly.
# =============================================================================
APP_CSS = """
/* === Layered background: subtle radial glows + a faint grid === */
.gradio-container {
    background:
        radial-gradient(ellipse 80% 50% at 15% 0%,
            rgba(245,158,11,0.10), transparent 60%),
        radial-gradient(ellipse 70% 50% at 100% 100%,
            rgba(99,102,241,0.14), transparent 55%),
        linear-gradient(180deg, #0b1020 0%, #0a0e1a 100%) !important;
    min-height: 100vh;
}
.gradio-container::before {
    content: "";
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
    background-size: 28px 28px;
    pointer-events: none;
    z-index: 0;
    mask-image: radial-gradient(ellipse at center, #000 30%, transparent 75%);
}
.gradio-container > * { position: relative; z-index: 1; }

/* === Hero header — "manuscript folio" feel === */
.hero {
    padding: 1.8rem 1.8rem 1.6rem;
    border-radius: 18px;
    background:
        radial-gradient(circle at 88% 0%, rgba(245,158,11,0.22), transparent 60%),
        linear-gradient(135deg, #1e1b4b 0%, #312e81 55%, #1e1b4b 100%);
    color: #f5e8c8;
    margin-bottom: 1.1rem;
    border: 1px solid rgba(245,158,11,0.18);
    box-shadow:
        0 12px 40px -12px rgba(0,0,0,0.6),
        inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
}
.hero::before {
    /* Subtle ornamental corner accent */
    content: "";
    position: absolute;
    top: -40px; right: -40px;
    width: 180px; height: 180px;
    background:
        conic-gradient(from 45deg,
            rgba(245,158,11,0.18), transparent 35%,
            rgba(245,158,11,0.10) 70%, transparent);
    filter: blur(8px);
}
.hero h1 {
    margin: 0 0 0.4rem 0;
    font-weight: 700;
    font-size: 1.85rem;
    letter-spacing: -0.01em;
    background: linear-gradient(90deg, #fef3c7 0%, #fcd34d 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.hero p {
    margin: 0;
    color: #cbd5e1;
    font-size: 0.98rem;
    line-height: 1.55;
    max-width: 720px;
}
.hero .pipeline {
    margin-top: 0.7rem;
    font-size: 0.82rem;
    color: #94a3b8;
    letter-spacing: 0.02em;
}
.hero code {
    background: rgba(245,158,11,0.12);
    color: #fcd34d;
    padding: 2px 8px;
    border-radius: 5px;
    font-size: 0.85em;
    border: 1px solid rgba(245,158,11,0.18);
    font-family: "JetBrains Mono", ui-monospace, Consolas, monospace;
}
.hero .badges { margin-top: 0.9rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
.hero .badge {
    display: inline-flex; align-items: center;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    color: #e7ecf3;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    letter-spacing: 0.02em;
}
.hero .badge b { color: #fcd34d; margin-left: 6px; }

/* === Section headers === */
.section-header {
    font-weight: 600;
    font-size: 1.05rem;
    color: #f5e8c8;
    margin: 0.3rem 0 0.5rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(245,158,11,0.22);
    letter-spacing: 0.01em;
}

/* === Arabic content surfaces (auto-direction per paragraph) ===
   `unicode-bidi: plaintext` makes each paragraph pick its direction from its
   first strong character — Arabic flows RTL, English placeholder flows LTR.
   `text-align: start` then aligns to whichever side that direction indicates. */
.arabic-text textarea,
.arabic-text .prose,
.arabic-text .markdown-body {
    unicode-bidi: plaintext !important;
    text-align: start !important;
    font-family: "Amiri", "Noto Naskh Arabic", "Segoe UI", serif;
    font-size: 1.05rem;
    line-height: 1.85;
    color: #e7ecf3;
}
.arabic-radio label > span:last-child {
    unicode-bidi: plaintext !important;
    text-align: start !important;
    display: inline-block;
    font-family: "Amiri", "Noto Naskh Arabic", serif;
    font-size: 1.0rem;
    line-height: 1.6;
}
.arabic-radio label {
    padding: 8px 12px !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    background: rgba(255,255,255,0.025) !important;
    margin: 4px 0 !important;
    transition: all 160ms ease;
}
.arabic-radio label:hover {
    background: rgba(245,158,11,0.08) !important;
    border-color: rgba(245,158,11,0.30) !important;
}

/* === Primary button polish === */
button.lg.primary {
    font-weight: 600 !important;
    letter-spacing: 0.02em;
    box-shadow: 0 6px 20px -6px rgba(245,158,11,0.55);
    transition: transform 120ms ease, box-shadow 180ms ease;
}
button.lg.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 10px 26px -6px rgba(245,158,11,0.65);
}
button.lg.primary:active { transform: translateY(0); }

/* === Footer === */
.footer {
    margin-top: 1.4rem;
    padding: 0.9rem 0;
    border-top: 1px solid rgba(255,255,255,0.08);
    color: #94a3b8;
    font-size: 0.82rem;
    text-align: center;
}
.footer a {
    color: #fcd34d;
    text-decoration: none;
    border-bottom: 1px dotted rgba(252,211,77,0.4);
}
.footer a:hover { border-bottom-style: solid; }
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
               theme=MANUSCRIPT_THEME,
               css=APP_CSS) as demo:

    gr.HTML(
        """
        <div class="hero">
            <h1>Smart Lecture Assistant <span style="opacity:0.55;font-weight:400">— Arabic</span></h1>
            <p>
                Upload an Arabic lecture and get a transcript, a structured study
                guide, and one-click drill-down to the exact source segment.
            </p>
            <div class="pipeline">
                <code>Whisper</code> &nbsp;ASR &nbsp;→&nbsp;
                <code>CAMeL-BERT + FAISS</code> &nbsp;Semantic Search &nbsp;→&nbsp;
                <code>AraBART</code> &nbsp;Summarization
            </div>
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
            gr.Markdown("### Key Takeaways", elem_classes=["section-header"])
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
