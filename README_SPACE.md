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

[![License: CC-BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Evaluated Claims Verified](https://img.shields.io/badge/Claims_Register-18_Verified_Claims-blue.svg)](Results/final/claims_register.json)

Upload an Arabic lecture and get back a transcript, a structured study guide, key takeaways, and one-click drill-down with timestamps to the exact transcript segment behind each idea.

> [!IMPORTANT]
> **Non-Commercial Research Demonstration:** This Space serves as an academic and portfolio research demo. Summarization components use AraBART fine-tuned on the **XL-Sum** dataset (licensed under **CC-BY-NC-SA 4.0**), restricted strictly to non-commercial evaluation. All metrics are cross-referenced to cryptographic claim records in the repository.

Three evaluated pipeline components composed into one deployable system:

| Stage | Evaluated Component | Benchmark | Baseline | Ours | Claim ID |
|---|---|---|---:|---:|---|
| **Speech-to-Text (Corpus)** | Whisper-Large-v3 CT2 INT8 | Common Voice 25.0 ar (10,506 clips) | — | **14.12% WER** | [`PASS2_CORPUS_WER`] |
| **Speech-to-Text (Long-Form)** | Whisper-Large-v3-Turbo CT2 INT8 | XL-Sum Neural Audio (50 articles) | — | **14.05% WER** | [`XLSUM_ASR_TURBO`] |
| **Summarization (Spoken)** | Spoken Turbo → AraBART | XL-Sum ar (50 articles, ROUGE-L ↑) | 33.94 Ref | **24.00 R-L** | [`SUMM_SPOKEN_TURBO`] |
| **Summarization (Oracle)** | Clean Text Oracle → AraBART | XL-Sum ar (50 articles, ROUGE-L ↑) | — | **33.94 R-L** | [`SUMM_CLEAN_ORACLE_RL`] |
| **Semantic Search (50 QA)** | CAMeL-BERT + mMARCO Cross-Encoder | ARCD 50 QA Pairs (P@1 ↑) | 0.5000 Bi | **0.6400 P@1** | [`RET_CLEAN_ORACLE_RERANK_P1`] |
| **Semantic Search (Full ARCD)**| CAMeL-BERT + mMARCO Cross-Encoder | ARCD Full (693 Questions, P@1 ↑) | — | **0.3694 P@1** | [`RET_FULL_ARCD_693_P1`] |

**Source code:** [GitHub repo](https://github.com/Omar10lfc/Arabic-Audio-Understanding-Retrieval-System)

---

## How to use the demo

1. **Upload audio** (MP3 / WAV / M4A) or paste a YouTube link.\*
2. Hit **Analyze** — the app runs ASR → summarization → indexing in one pass. First request loads ~1.5 GB of model weights from the Hub.
3. Read the **transcript**, **study guide**, and **key takeaways** tabs.
4. Click any takeaway to **drill down** to the matching transcript segment with its timestamp.
5. Download the study guide as a **PDF** (uses the bundled Amiri font for Arabic-correct rendering).

\* YouTube ingest is disabled on this hosted Space — YouTube blocks unauthenticated yt-dlp requests from datacenter IPs. Run the app locally (instructions in the GitHub repo) to keep the full feature set.

---

## What's inside

All four models load lazily on first use; the pure helpers (Arabic normalization, chunking, Jaccard dedup) have no model dependencies.

- **ASR** — `Whisper-Large-v3-Turbo` quantized to CTranslate2 INT8 (`faster-whisper`), delivering 14.05% WER on long-form articles [`XLSUM_ASR_TURBO`] and 14.12% on the Common Voice full pool [`PASS2_CORPUS_WER`].
- **Summarization** — `moussaKam/AraBART` fine-tuned on the full Arabic XL-Sum train split. Retains 24.00 ROUGE-L on spoken audio [`SUMM_SPOKEN_TURBO`] vs 33.94 clean oracle ceiling [`SUMM_CLEAN_ORACLE_RL`].
- **Semantic search** — `CAMeL-Lab/bert-base-arabic-camelbert-msa` mean-pooled + L2-normalized (`max_length=128`) indexed via `IndexFlatIP`, reranked by `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. Achieves 0.6400 P@1 on 50 QA pairs [`RET_CLEAN_ORACLE_RERANK_P1`] and 0.3694 P@1 on the full 693-question ARCD corpus [`RET_FULL_ARCD_693_P1`].

---

## Limitations

- **Modern Standard Arabic only.** Dialectal Arabic (Maghrebi, Khaleeji, Egyptian) is under-represented in Common Voice and degrades both ASR and summarization quality.
- **Code-switching with English** (Transformer, GPU, BERT…) gets transliterated phonetically into Arabic script — this is a Common Voice training-distribution artifact, not a deployment bug.
- **Summarization can hallucinate "article" framing** ("في هذا المقال…") because AraBART was fine-tuned on XL-Sum, which is BBC news articles, not lectures.
- **Long-form audio is chunked at 30 s** by Whisper's standard windowing — no diarization or speaker separation.
- **Free-tier CPU Spaces are slow.** A 5-minute clip takes a few minutes end-to-end. ZeroGPU or a paid GPU tier brings this down to seconds.

---

## License

Code: **Apache 2.0**. Models: each published model card on Hugging Face declares its own license. Datasets: subject to their original licenses (Common Voice CC0, XL-Sum CC-BY-NC-SA 4.0, ARCD CC-BY-SA 4.0). Bundled Amiri font: SIL Open Font License 1.1.
