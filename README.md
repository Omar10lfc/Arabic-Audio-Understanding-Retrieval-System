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

End-to-end Arabic audio understanding pipeline. Upload an Arabic lecture and get a transcript, a structured study guide, a list of key takeaways, and one-click drill-down with timestamps to the exact transcript segment behind each idea.

Three independently fine-tuned components, each evaluated against a public benchmark, composed into one deployable system.

| Stage | Model | Benchmark | Baseline | Ours | Lift |
|---|---|---|---:|---:|---:|
| Speech-to-Text | [Whisper-small Arabic (fine-tuned)](https://huggingface.co/Omar10lfc/whisper-small-arabic) | Common Voice ar (WER ↓) | 42.69 | **20.61** | **−51.7%** |
| Summarization  | [AraBART-XLSum Arabic (fine-tuned)](https://huggingface.co/Omar10lfc/arabart-xlsum-arabic) | XL-Sum ar (ROUGE-L ↑) | 13.48 | **29.56** | **+16.08** |
| Semantic Search | CAMeL-BERT MSA + FAISS + cross-encoder rerank | ARCD (P@1 ↑) | 0.64 | **0.86** | **+34%** |

**Live demo:** [Smart Lecture Assistant on HF Spaces](https://huggingface.co/spaces/Omar10lfc/smart-lecture-assistant-arabic)

---

## Pipeline

```
                   ┌──────────────────────────────────────────────────┐
   audio (.wav,    │                                                  │
   .mp3, YouTube)  ┼─► Whisper-small-arabic   ─► transcript           │
                   │       (ASR, fine-tuned)         (with timestamps)│
                   │                                                  │
                   │                ├─► chunk @ 50 words ─► CAMeL-BERT│
                   │                │     embedder + FAISS index      │
                   │                │                                 │
                   │                ├─► AraBART-XLSum ─► study guide  │
                   │                │      (5 sections)   (PDF + MD)  │
                   │                │                                 │
                   │                └─► AraBART per-segment ─► 3-5    │
                   │                        takeaways                 │
                   │                                                  │
                   │   click takeaway ─► cross-encoder rerank top-10  │
                   │                  ─► exact transcript segment     │
                   │                       (with mm:ss timestamp)     │
                   └──────────────────────────────────────────────────┘
```

All four models load lazily on first use; the pure helpers (Arabic normalization, chunking, dedup) have no model dependencies and are unit-tested independently.

---

## Quick start

### Prerequisites

- Python 3.10–3.12
- ffmpeg on `PATH` (required by `librosa` and `yt-dlp`)
- ~2.5 GB of disk for the cached HF model weights on first run

### Setup

```bash
git clone https://github.com/Omar10lfc/Arabic-Audio-Understanding-Retrieval-System.git
cd Arabic-Audio-Understanding-Retrieval-System

python -m venv .venv
# Linux / macOS
source .venv/bin/activate
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Run the Gradio app

```bash
python app.py
```

The app launches at <http://127.0.0.1:7860>. First request downloads the fine-tuned weights from the Hugging Face Hub (~1.5 GB, cached afterward).

### Run the FastAPI backend

```bash
uvicorn api:api --host 0.0.0.0 --port 8000
```

OpenAPI docs at <http://127.0.0.1:8000/docs>. Endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET`  | `/health` | Liveness + model-load status |
| `POST` | `/transcribe` | Multipart audio → transcript only |
| `POST` | `/analyze` | Multipart audio → transcript + study guide + takeaways + `session_id` |
| `POST` | `/drill-down` | `{session_id, takeaway}` → matching transcript chunk |
| `GET`  | `/cheat-sheet/{sid}.pdf` | Download the PDF study guide from a session |

### Run the tests

```bash
pytest -q
```

49 tests across [tests/test_helpers.py](tests/test_helpers.py) (Arabic normalization, chunking, Jaccard dedup) and [tests/test_api.py](tests/test_api.py) (FastAPI surface, monkey-patched to avoid loading models). All run in <1 s.

### Generate a test audio file

```bash
pip install edge-tts
python generate_test_audio.py
```

Produces `arabic_lecture_sample.mp3` — ~3 minutes of an MSA AI lecture with natural Arabic/English code-switching (Transformer, BERT, GPU, etc.) for stress-testing the ASR.

---

## Project structure

```
arabic_audio_system/
├── app.py                         # Gradio UI (bilingual, custom Manuscript theme)
├── api.py                         # FastAPI REST backend (same pipeline)
├── pipeline.py                    # Shared backend: model loading + helpers
├── generate_test_audio.py         # edge-tts helper: ~3 min Arabic lecture sample
├── requirements.txt               # Python deps (HF Space-compatible pins)
├── packages.txt                   # apt deps for HF Spaces (ffmpeg)
│
├── nlp-fine-tune-edit-1.ipynb     # Whisper fine-tuning (stage 1, lr 1e-5)
├── nlp-fine-tune-edit-2.ipynb     # Whisper fine-tuning (stage 2, lr 5e-6)
├── summarization.ipynb            # AraBART + mT5-XLSum train + eval
├── embedding-eval.ipynb           # FAISS + cross-encoder ablation
│
├── push_whisper_to_hub.py         # One-shot: push Whisper folder to HF Hub
├── push_arabart_to_hub.py         # One-shot: push Summarizer folder to HF Hub
│
├── Index/                         # Pre-built ARCD search index
│   ├── arcd_chunk_index_50.faiss  #   FAISS index (50-word chunks)
│   ├── chunk_embeddings_50.npy    #   raw embeddings
│   └── chunks_50.json             #   chunk text + metadata
│
├── fonts/
│   └── Amiri-Regular.ttf          # SIL-OFL-1.1 Arabic font for the PDF export
│
├── tests/
│   ├── test_helpers.py            # Pure-helper unit tests (no models)
│   ├── test_api.py                # FastAPI tests (monkey-patched pipeline)
│   └── conftest.py
│
├── REPORT_experiments_results.md  # Full experimental writeup
├── results-Whisper-finetuned.json # Whisper headline numbers
├── summarization_results.csv      # AraBART + mT5 ROUGE/BLEU numbers
└── chunking_reranking_results.csv # Chunk-size / rerank ablation numbers
```

Excluded from git (see [.gitignore](.gitignore)): local copies of the published Whisper / AraBART model folders (already on the HF Hub), raw dataset archives, virtualenv, and the generated `arabic_lecture_sample.mp3`.

---

## Datasets

| Component | Dataset | Source | Splits used |
|---|---|---|---|
| ASR | Mozilla Common Voice (Arabic) | [Common Voice](https://commonvoice.mozilla.org/ar/datasets) | 25,000 train / 300 test, seed 42 |
| Summarization | XL-Sum v2.0 (Arabic split) | [csebuetnlp/xl-sum](https://github.com/csebuetnlp/xl-sum) | 37,454 train / 4,689 val / 4,688 test (after cleaning) |
| Semantic Search | ARCD — Arabic Reading Comprehension Dataset | included in [Index/](Index/) | 231 passages, 200 query–context pairs |

XL-Sum is licensed CC-BY-NC-SA 4.0; Common Voice is CC0; ARCD is CC-BY-SA 4.0. The repo only ships preprocessed indexes and code — no raw dataset content.

---

## Models

### 1. [Omar10lfc/whisper-small-arabic](https://huggingface.co/Omar10lfc/whisper-small-arabic)

`openai/whisper-small` fine-tuned on Common Voice Arabic. Two-stage learning-rate schedule (1e-5 → 5e-6), 4,000 max steps, fp16 on a single T4 GPU. Published checkpoint = step 3,000 (lowest val WER).

| Metric | Baseline | Fine-tuned | Δ |
|---|---:|---:|---:|
| WER ↓ | 42.69 | **20.61** | −22.08 abs / −51.7% rel |

### 2. [Omar10lfc/arabart-xlsum-arabic](https://huggingface.co/Omar10lfc/arabart-xlsum-arabic)

`moussaKam/AraBART` fine-tuned on the full Arabic XL-Sum train split. 3 epochs, cosine schedule, label smoothing 0.1, learning rate 2e-5, effective batch size 16, fp16.

| Model | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BLEU ↑ |
|---|---:|---:|---:|---:|
| AraBART (no fine-tuning) | 20.25 | 5.08 | 13.48 | 1.83 |
| mT5-XLSum (zero-shot) | 34.82 | 14.77 | 29.17 | 7.43 |
| **AraBART (fine-tuned)** | **34.99** | **15.77** | **29.56** | **8.26** |

Evaluated with the **official XL-Sum scorer** (`csebuetnlp/xl-sum`'s `multilingual_rouge_scoring` with `lang='arabic'` Snowball stemmer) — same package used by Hasan et al. 2021. Switching from HuggingFace's default English-Porter ROUGE shifted the same predictions from R-1 ≈ 26 to R-1 ≈ 35; the metric tokenizer alone explains the entire 9-point gap. Always report with the paper's scorer if you want comparable numbers.

### Embedding + reranking (no fine-tuning)

| Role | Model | Why |
|---|---|---|
| Embedder | `CAMeL-Lab/bert-base-arabic-camelbert-msa` | Strong MSA-trained Arabic BERT; mean-pooled + L2-normalized to match `IndexFlatIP` (cosine via inner product). |
| Reranker | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Multilingual cross-encoder; reads (query, chunk) jointly — adds +20–24 P@1 over bi-encoder alone. |

Chunk-size / rerank ablation on ARCD (top-10 retrieval, 200 queries):

| Configuration | P@1 | P@3 | P@5 |
|---|---:|---:|---:|
| 50 words, FAISS only | 0.64 | 0.80 | 0.88 |
| **50 words, + rerank** | **0.86** | **0.90** | **0.90** |
| 100 words, FAISS only | 0.58 | 0.76 | 0.88 |
| 100 words, + rerank | 0.84 | 0.90 | 0.90 |
| 200 words, FAISS only | 0.60 | 0.78 | 0.86 |
| 200 words, + rerank | 0.82 | 0.92 | 0.92 |

50-word chunks + cross-encoder rerank were selected for production. Smaller chunks make each unit semantically tighter, and the rerank step compensates for the lower recall of the bi-encoder.

---

## Reproducing the experiments

The training notebooks were run on Kaggle T4 sessions and assume the dataset archives sit next to the notebook.

| Notebook | Purpose | Approx runtime on T4 |
|---|---|---|
| [nlp-fine-tune-edit-1.ipynb](nlp-fine-tune-edit-1.ipynb) | Whisper stage 1 (LR 1e-5, 0–2000 steps) | ~3 h |
| [nlp-fine-tune-edit-2.ipynb](nlp-fine-tune-edit-2.ipynb) | Whisper stage 2 (LR 5e-6, 2000–4000 steps) | ~3 h |
| [summarization.ipynb](summarization.ipynb) | AraBART fine-tune + mT5 cont-FT + ROUGE/BLEU eval | ~5 h |
| [embedding-eval.ipynb](embedding-eval.ipynb) | Build FAISS index + chunk-size/rerank ablation | ~30 min |

The full experimental writeup (methodology, decoding parameters, ablation analysis) lives in [REPORT_experiments_results.md](REPORT_experiments_results.md).

---

## Pushing your own fine-tunes to the Hub

Two convenience scripts mirror the structure of the published model cards:

```bash
# After `huggingface-cli login` with a write token:

python push_whisper_to_hub.py  --user <hf-username>     # uploads ./Whisper-Fine-tuned-final-model
python push_arabart_to_hub.py  --user <hf-username>     # uploads ./Summarizer
```

Both scripts validate required files, skip `training_args.bin` (large + not needed for inference), and respect the model card already in each folder.

---

## Deploying to Hugging Face Spaces

The repo is ready to deploy as a Gradio Space:

1. Create a new Space (Gradio SDK, CPU basic is enough for a demo).
2. Add the repo as a remote: `git remote add space https://huggingface.co/spaces/<user>/<space>`.
3. Push: `git push space main`.

The YAML frontmatter at the top of this README, [requirements.txt](requirements.txt), and [packages.txt](packages.txt) are already configured for Spaces (Python 3.11, Gradio 4.44.1, ffmpeg).

**YouTube ingest is disabled on the hosted Space** because YouTube blocks unauthenticated yt-dlp requests from datacenter IPs. The local `python app.py` path keeps the full feature set.

---

## Limitations

- **Modern Standard Arabic only.** Dialectal Arabic (Maghrebi, Khaleeji, Egyptian) is under-represented in Common Voice and degrades both ASR and summarization quality.
- **Code-switching with English** (Transformer, GPU, BERT…) gets transliterated phonetically into Arabic script — this is a Common Voice training-distribution artifact, not a deployment bug.
- **Summarization can hallucinate "article" framing** ("في هذا المقال…") because AraBART was fine-tuned on XL-Sum, which is BBC news articles, not lectures.
- **Long-form audio is chunked at 30 s** by Whisper's standard windowing — no diarization or speaker separation.
- **Free-tier CPU Spaces are slow.** A 5-minute clip takes a few minutes end-to-end. ZeroGPU or a paid GPU tier brings this to seconds.

---

## Citation

If you use this work, please cite the components it builds on:

```bibtex
@misc{whisper-small-arabic,
  title  = {whisper-small-arabic: Fine-tuned Whisper for Arabic on Mozilla Common Voice},
  author = {{Omar10lfc}},
  year   = {2026},
  howpublished = {Hugging Face},
}

@misc{arabart-xlsum-arabic,
  title  = {arabart-xlsum-arabic: Fine-tuned AraBART for Arabic abstractive summarization on XL-Sum},
  author = {{Omar10lfc}},
  year   = {2026},
  howpublished = {Hugging Face},
}

@article{radford2022whisper,
  title  = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author = {Radford, Alec and Kim, Jong Wook and Xu, Tao and others},
  journal= {arXiv preprint arXiv:2212.04356},
  year   = {2022}
}

@inproceedings{kamal-eddine-etal-2022-arabart,
  title     = {{A}ra{BART}: a Pretrained {A}rabic Sequence-to-Sequence Model for Abstractive Summarization},
  author    = {Kamal Eddine, Moussa and Tomeh, Nadi and Habash, Nizar and Le Roux, Joseph and Vazirgiannis, Michalis},
  booktitle = {Proceedings of the Seventh Arabic Natural Language Processing Workshop (WANLP)},
  year      = {2022}
}

@inproceedings{hasan-etal-2021-xl,
  title     = {{XL}-Sum: Large-Scale Multilingual Abstractive Summarization for 44 Languages},
  author    = {Hasan, Tahmid and Bhattacharjee, Abhik and Islam, Md. Saiful and Mubasshir, Kazi and Li, Yuan-Fang and Kang, Yong-Bin and Rahman, M. Sohel and Shahriyar, Rifat},
  booktitle = {Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021},
  year      = {2021}
}
```

---

## License

Code: **Apache 2.0**.
Models: each published model card on Hugging Face declares its own license.
Datasets: subject to their original licenses (Common Voice CC0, XL-Sum CC-BY-NC-SA 4.0, ARCD CC-BY-SA 4.0). The repo does not ship raw dataset content.
Bundled Amiri font: SIL Open Font License 1.1.
