---
title: Smart Lecture Assistant (Arabic)
emoji: 🎙️
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.1
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Arabic ASR → summary → semantic search over the transcript.
---

# Smart Lecture Assistant — Arabic

End-to-end Arabic audio understanding pipeline:

1. **ASR** — fine-tuned [Whisper-small-arabic](https://huggingface.co/Omar10lfc/whisper-small-arabic) (WER 20.61 on Common Voice ar).
2. **Summarization** — fine-tuned [AraBART-XLSum-arabic](https://huggingface.co/Omar10lfc/arabart-xlsum-arabic) (ROUGE-L 29.56 on XL-Sum ar).
3. **Semantic search** — CAMeL-BERT MSA embeddings + FAISS over the transcript chunks, with a multilingual cross-encoder reranker.

Upload an audio file (or paste a YouTube URL) → get a transcript, a structured cheat sheet, key takeaways, and click any takeaway to jump to the exact transcript segment.
