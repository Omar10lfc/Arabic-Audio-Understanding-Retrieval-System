# Experiments and Evaluation Results

## 4. Experiments

### 4.1 Setup

The system was implemented as three independently trained components, each evaluated against an established public benchmark, and then composed into a single end-to-end pipeline (Whisper → AraBART → CAMeL-BERT/FAISS).

**Datasets**

| Component | Dataset | Train / Val / Test |
|---|---|---|
| ASR | Mozilla Common Voice Arabic | 25,000 train / 300 test |
| Summarization | XL-Sum v2.0 (Arabic split) | 37,454 / 4,689 / 4,688 |
| Semantic Search | ARCD (Arabic Reading Comprehension) | 231 passages, 200 query–context pairs |

**Hardware**: All training was run on a single NVIDIA Tesla T4 (16 GB VRAM) on Kaggle. Inference is supported on both CPU and GPU.

**Evaluation methodology**: ASR is scored with Word Error Rate (WER) on the held-out test split. Summarization uses the **official `multilingual_rouge_scoring` package from the XL-Sum authors** (NLTK Arabic Snowball stemmer, `lang="arabic"`) plus SacreBLEU with `tokenize="intl"`, ensuring numbers are directly comparable to Hasan et al. (2021). Search uses Precision@K with K ∈ {1, 3, 5} on top-10 FAISS retrieval.

### 4.2 Task 1 — Speech-to-Text (Arabic Whisper)

`openai/whisper-small` was fine-tuned for 4,000 steps on the Common Voice Arabic train split with a peak learning rate of 1e-5 and warm-up. The same checkpoint and decoding configuration were used for both the baseline (zero-shot) and fine-tuned evaluations to ensure WER differences are attributable solely to fine-tuning.

### 4.3 Task 2 — Text Summarization (AraBART vs. mT5-XLSum)

Three configurations were evaluated on the **full** XL-Sum Arabic test set (n = 4,688):

1. **AraBART (no fine-tuning)** — `moussaKam/AraBART` used as-is, to establish the floor.
2. **mT5-XLSum (zero-shot)** — `csebuetnlp/mT5_multilingual_XLSum`, already trained on XL-Sum-Arabic, used for inference only. Acts as a strong, paper-comparable reference.
3. **AraBART (fine-tuned)** — `moussaKam/AraBART` fine-tuned on the full 37,454-pair Arabic train split for 3 epochs with cosine LR schedule, label smoothing 0.1, learning rate 2e-5, effective batch size 16.

All three configurations used the XL-Sum paper's generation parameters at inference time — `num_beams=4`, `length_penalty=0.6`, `no_repeat_ngram_size=2`, `min_length=10`, `max_length=84`, `padding="max_length"` — copied verbatim from the model card. This ensures the comparison isolates training quality, not decoding strategy.

### 4.4 Task 3 — Semantic Search (FAISS + Cross-Encoder)

Passages were embedded with `CAMeL-Lab/bert-base-arabic-camelbert-msa` (mean-pooling over the last hidden state, L2-normalized) and indexed with `faiss.IndexFlatIP` (inner product on L2-normalized vectors equals cosine similarity). Three chunk-size configurations (50, 100, 200 words) were evaluated, both with and without a cross-encoder reranking stage using `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` over the top 10 FAISS candidates.

---

## 5. Evaluation Results

### 5.1 Speech-to-Text (Task 1)

| Configuration | WER ↓ | Δ (abs) | Δ (rel) |
|---|---:|---:|---:|
| Whisper-small (baseline)  | 42.69 | – | – |
| Whisper-small (fine-tuned) | **20.61** | **−22.08** | **−51.7%** |

Fine-tuning halves the error rate. The fine-tuned model becomes the ASR front-end of the deployed pipeline.

### 5.2 Summarization (Task 2)

| Model | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BLEU ↑ |
|---|---:|---:|---:|---:|
| AraBART (no fine-tuning)            | 20.25 |  5.08 | 13.48 | 1.83 |
| mT5-XLSum (zero-shot)               | 34.82 | 14.77 | 29.17 | 7.43 |
| **AraBART (fine-tuned, ours)**      | **34.99** | **15.77** | **29.56** | **8.26** |

n = 4,688 (full Arabic test set). All metrics × 100.

Three observations:

1. **Paper reproduction.** Our zero-shot mT5-XLSum row (R-1 = 34.82) is within 0.1 of the published number for the same checkpoint (Hasan et al. 2021, Table 4: R-1 = 34.91). This confirms the loader, tokenization, and metric pipeline are correctly aligned with the paper.
2. **Fine-tuning effect (R-L 13.48 → 29.56).** Three epochs on the full Arabic train split lift AraBART by **+16.08 absolute ROUGE-L**, comparable in magnitude to the Whisper improvement on the ASR side.
3. **Monolingual beats multilingual.** Fine-tuned AraBART outperforms `mT5_multilingual_XLSum` on every metric (ROUGE-1: +0.17, ROUGE-2: +1.00, ROUGE-L: +0.39, BLEU: +0.83) despite being a smaller monolingual model. The multilingual model still has the advantage of seeing English code-switched tokens cleanly, so it is retained as a fallback option in the deployed system for transcripts with heavy English mixing.

### 5.3 Semantic Search (Task 3)

| Configuration | P@1 ↑ | P@3 ↑ | P@5 ↑ |
|---|---:|---:|---:|
| Small (50 words) — FAISS only       | 0.64 | 0.80 | 0.88 |
| Small (50 words) — **+ Re-Rank**    | **0.86** | **0.90** | **0.90** |
| Medium (100 words) — FAISS only     | 0.58 | 0.76 | 0.88 |
| Medium (100 words) — + Re-Rank      | 0.84 | 0.90 | 0.90 |
| Large (200 words) — FAISS only      | 0.60 | 0.78 | 0.86 |
| Large (200 words) — + Re-Rank       | 0.82 | 0.92 | 0.92 |

Two findings:

1. **Cross-encoder reranking adds +20 to +24 P@1** across every chunk size. The reranker reads the full (query, candidate) pair jointly, which captures fine-grained relevance that the bi-encoder's single-vector cosine similarity cannot.
2. **50-word chunking wins at P@1** (the metric most relevant for question-answering). Smaller chunks make each unit semantically tighter, so the embedder produces more discriminative representations. Larger chunks gain a small advantage at P@3/P@5 by simply containing more text, but lose precision. The 50-word configuration was therefore selected for the deployed system.

### 5.4 End-to-End Pipeline (combined)

The final integrated system performs three sequential transformations on a single Arabic audio input. Each stage's improvement story is summarized below:

| Stage | Metric | Untrained / Baseline | Trained System | Lift |
|---|---|---:|---:|---:|
| Speech-to-Text | WER ↓             | 42.69 | **20.61** | −22.08 abs / −52% rel |
| Summarization  | ROUGE-L ↑         | 13.48 | **29.56** | +16.08 abs |
| Semantic Search | P@1 ↑            | 0.64 (FAISS only) | **0.86** (+ rerank) | +0.22 abs / +34% rel |

Each component shows a clear, quantified before/after improvement, and each result is competitive with published baselines on its respective benchmark.

### 5.5 Demo Integration

The three components are exposed through two equivalent interfaces:

- A **Gradio web app** (`app.py`) built around the *Smart Lecture Assistant* concept: audio upload → automated cheat-sheet (PDF + Markdown) + 3–5 clickable takeaways → drill-down to the exact transcript chunk via cross-encoder rerank.
- A **FastAPI REST backend** (`api.py`) exposing `/transcribe`, `/analyze`, `/drill-down`, and `/cheat-sheet/{sid}.pdf` for programmatic clients.

Both are thin layers over a shared `pipeline.py` module that performs all model loading lazily; the test suite (`pytest`, 44 tests) verifies the helpers and the API surface without requiring any model weights to be present.

---

## 6. Conclusion

The system meets every deliverable in §10 of the project specification:

- **Source code**: three notebooks (`embedding-eval.ipynb`, `summarization.ipynb`, the Whisper fine-tuning script) plus the deployed Python package (`pipeline.py` / `app.py` / `api.py`).
- **Dataset description**: §4.1 above.
- **Architecture diagram**: matches §8 of the specification, instantiated in `pipeline.audio_to_insights()`.
- **Experiments and evaluation results**: §5.1–§5.3.
- **Demo interface**: the Gradio app described in §5.5, with a FastAPI alternative for non-browser clients.

The most important methodological finding is that **evaluation infrastructure matters as much as model quality**: switching from the default HuggingFace ROUGE implementation to the XL-Sum-paper's `multilingual_rouge_scoring` package (Arabic Snowball stemmer) shifted the same predictions from R-1 ≈ 26 to R-1 ≈ 35 — a 9-point gap explained entirely by the metric tokenizer. Reporting the paper's official scorer is what makes our numbers directly comparable to prior work.
