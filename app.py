"""
app.py — Smart Lecture Assistant (Gradio UI)
=============================================

Thin Gradio front-end. All ML/pipeline logic lives in `pipeline.py`.

Run:
    python app.py
"""

import gradio as gr

from pipeline import (
    run_whisper, build_index,
    generate_cheat_sheet, generate_takeaways, drill_down,
    download_audio_from_url, warm_up,
)


# =============================================================================
# Global RTL styling — every Arabic surface should read right-to-left
# =============================================================================
RTL_CSS = """
.gradio-container, .gradio-container * {
    direction: rtl;
    text-align: right;
}
textarea, input[type="text"] {
    direction: rtl !important;
    text-align: right !important;
}
.markdown-body, .prose {
    direction: rtl;
    text-align: right;
}
button { direction: rtl; }
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
            return (f"⚠️ تعذر تحميل الفيديو من الرابط: {e}",
                    empty_radio, "", None, {}, "")

    if not audio_path:
        empty_radio = gr.Radio(choices=[], value=None, interactive=True)
        return ("⚠️ الرجاء تحميل ملف صوتي أو إدخال رابط فيديو.",
                empty_radio, "", None, {}, "")

    transcript = run_whisper(audio_path)
    if not transcript:
        empty_radio = gr.Radio(choices=[], value=None, interactive=True)
        return ("⚠️ تعذر التعرف على الصوت. تأكد من جودة الملف.",
                empty_radio, "", None, {}, "")

    chunks, index = build_index(transcript)
    cheat_md, pdf = generate_cheat_sheet(chunks, index)
    takeaways     = generate_takeaways(transcript)

    radio = gr.Radio(
        choices=takeaways,
        value=None,
        label="اضغط على فكرة لعرض السياق الأصلي من المحاضرة",
        interactive=True,
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
with gr.Blocks(title="Smart Lecture Assistant",
               theme=gr.themes.Soft(),
               css=RTL_CSS) as demo:

    gr.Markdown(
        """
        # 🎓 المساعد الذكي للمحاضرات

        **حوّل صوت المحاضرة إلى دليل مراجعة كامل + اكتشف السياق الأصلي خلف كل فكرة.**

        *المراحل: التعرف على الصوت → الفهرسة الدلالية → التلخيص → إعادة الترتيب*
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            audio_in = gr.Audio(sources=["upload", "microphone"],
                                type="filepath",
                                label="🎙️ ملف صوت المحاضرة")
            url_in = gr.Textbox(
                label="📹 أو ألصق رابط فيديو يوتيوب",
                placeholder="https://www.youtube.com/watch?v=...",
                lines=1,
            )
        with gr.Column(scale=1):
            analyze_btn = gr.Button("🔍 حلل المحاضرة",
                                    variant="primary", size="lg")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 📘 دليل المراجعة")
            cheat_out = gr.Markdown(value="*سيظهر دليل المراجعة هنا بعد التحليل.*")
            pdf_out   = gr.File(label="⬇️ تحميل بصيغة PDF",
                                interactive=False)

        with gr.Column(scale=1):
            gr.Markdown("## ✨ أهم الأفكار")
            takeaways_radio = gr.Radio(
                choices=[], value=None,
                label="اضغط على فكرة لعرض السياق الأصلي",
                interactive=True,
            )
            drill_out = gr.Textbox(
                label="📝 السياق الأصلي من المحاضرة",
                lines=8, interactive=False,
                placeholder="ستظهر هنا الفقرة الأصلية المرتبطة بالفكرة المختارة.",
            )

    with gr.Accordion("📜 النص الكامل للمحاضرة", open=False):
        transcript_box = gr.Textbox(lines=12, interactive=False, show_label=False)

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
