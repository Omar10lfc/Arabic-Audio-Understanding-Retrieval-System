# Retraining Assessment & Production Deployment Strategy
**Smart Lecture Assistant (Arabic ASR $\to$ AraBART $\to$ CAMeL-BERT / FAISS)**  
*Date: October 30, 2026*  
*Author: Senior ML Engineer*  
*Status: Approved Baseline & Strategic Execution Plan*

---

## 1. Executive Summary & Principles

An engineering audit scored this repository **54/100 for research** and **28/100 for production**. A core directive of this remediation is **strict capital and metric discipline**:
1. **Single Source of Truth**: All published metrics must trace to `Results/final/claims_register.json` with associated claim IDs and SHA-256 hashes.
2. **Never Retrain by Default**: Retraining incurs cloud GPU costs and invalidates existing claims. Always exhaust cheaper fixes (decoding controls, re-merging, post-filtering, hybrid search) first.
3. **Budget Constraint**: Hard spend cap is **$0.00**; approval threshold is **$0.00** per job. All measurements requiring GPU or audio not locally present are marked `unmeasured` per Rule 1.
4. **Deliverable Alignment**: Non-commercial research demo and portfolio piece under CC-BY-NC-SA 4.0.

---

## 2. Candidate Analysis

### Candidate 1: ASR Checkpoint (`whisper-large-v3` and `whisper-large-v3-turbo` QLoRA)
* **The Problem:**
  - `pipeline.py` previously defaulted `WHISPER_MODEL_ID` to zero-shot `large-v3-turbo`, not the fine-tuned checkpoint.
  - Merged FP16 `large-v3` measured **13.03% WER** on Subset A (400 clips) versus **12.51%** reported (`GATE_01_FP16`), exceeding the $\pm 0.25\%$ tolerance by $+0.27\%$.
  - Hugging Face Hub weights lacked cryptographic verification against the Kaggle output artifacts.
* **Trainable Parameter Recount (Correction):**
  - The previous audit mention of "14.15M" was a conflation between Whisper-Medium's parameter count (14,155,776 params / 1.82%) and Turbo's WER (14.15%).
  - The exact trainable parameters extracted from the PEFT adapter configs:
    - **Whisper-Large-v3 QLoRA**: $23,592,960$ trainable parameters ($r=64, \alpha=128$, `q_proj, v_proj` on 32 encoder layers + 32 decoder layers).
    - **Whisper-Large-v3-Turbo QLoRA**: $9,830,400$ trainable parameters ($r=64, \alpha=128$, `q_proj, v_proj` on 4 decoder layers + 32 encoder layers).
    - **Whisper-Medium QLoRA**: $14,155,776$ trainable parameters ($r=32, \alpha=64$).
* **Cheapest Fix to Try First (In Order):**
  1. *Artifact Verification:* Locate local adapters and verify SHA-256 against `sha256_manifest.json`. **Result (Verified):** The local `whisper_medium_output/.../checkpoint-3600/adapter_model.safetensors` has SHA-256 `f48948c39650156035020072ee91de61538b551cece50a2d8937a5e374fedef0`, bit-identically matching the Kaggle evaluated zero-notebook artifact. The Large-v3 and Turbo adapter safetensors are missing locally and reside in Kaggle kernel outputs `omar10lfc/fine-tuning-whisper-large-v3-on-arabic-speech` and `omar10lfc/whisper-large-turbo`.
  2. *3-Arm Re-Merge Verification (`scripts/experiment_asr_remerge.py`):*
     - **Arm 1 (NF4 Base + LoRA):** Base model loaded in 4-bit NF4 with adapter applied in-situ via PEFT forward pass (unmeasured).
     - **Arm 2 (Merge Dequantized NF4):** Dequantize 4-bit NF4 weights to FP16/FP32, then merge adapter delta into dequantized base (unmeasured).
     - **Arm 3 (FP32/BF16 Re-Merge):** True unquantized FP32/BF16 re-merge of base and adapter (unmeasured).
     - *Historical Kaggle Baseline:* Existing Kaggle FP16 merge result on Subset A is 13.03% WER (`GATE_01_FP16`). All three experimental arms are unmeasured pending a GPU run.
  3. *Exact Served Artifact Measurement:* Rather than proposing training-time 14.15% or FP16 13.03% as served numbers, evaluate the exact served artifact (`whisper-large-v3-turbo-ct2-int8`) on the 400 Subset A clips on GPU, and register it in `claims_register.json` under claim ID `SERVED_TURBO_CT2_INT8_GATE_WER`.
* **Modal GPU Cost Estimate for EXP1:**
  - Hardware: NVIDIA L4 GPU on Modal ($0.80/hour = $0.0133/minute).
  - Runtime: ~12 minutes for 400 clips across Arms 1-3 and Turbo CT2 INT8.
  - Estimated cost: **~$0.16**, well within the user-approved **$5.00** budget cap.
* **Full Retraining Recipe (Contingency Only):**
  - *Base Models:* `openai/whisper-large-v3` (`06f233f`) or `openai/whisper-large-v3-turbo` (`41f01f3`).
  - *QLoRA Config:* Rank $r=64$, $\alpha=128$, dropout $0.05$, target modules `["q_proj", "v_proj"]`.
  - *Precision:* 4-bit NF4 base, FP16 compute dtype, BF16 adapter weights.
  - *Optimization:* AdamW ($\beta_1=0.9, \beta_2=0.999$, $\epsilon=10^{-8}$), lr $1\times 10^{-4}$, linear decay with 500 warmup steps, effective batch size 64 (per-device 16 $\times$ 4 gradient accumulation).
  - *Dataset:* Mozilla Common Voice 25.0 Arabic (10,506 clean clips, length $>2$, seed 42).
  - *Gate Check:* Merged FP16 and CT2 INT8 artifacts must both achieve $\le 12.51\% \pm 0.25\%$ ($12.76\%$) on Subset A.
* **Recommendation:** **DO NOT RETRAIN YET.** Fetch missing adapter safetensors, execute Arms 1-3 and served Turbo CT2 INT8 on Modal L4 (estimated cost $0.16), and register the measured served WER. Update claims register if needed.

---

### Candidate 2: Summarizer (`AraBART` on XL-Sum)
* **The Problem:**
  - Fine-tuned on BBC XL-Sum (`csebuetnlp/xl-sum`), which is licensed under CC-BY-NC-SA 4.0 (strictly non-commercial).
  - Hallucinates broadcast news framing boilerplate (e.g., "في هذا المقال", "ذكرت بي بي سي") on academic lecture content.
* **User Decision:**
  - Product deliverable is confirmed as a **non-commercial research demo and portfolio piece**. The CC-BY-NC-SA 4.0 license badge has been added to `README.md` and `README_SPACE.md`.
* **Cheapest Fix to Try First:**
  - Decoding controls: opening-phrase post-filter (`pipeline.filter_news_framing()`), which strips framing phrases like "في هذا المقال" and "في هذا التقرير".
  - Beam decoding parameters aligned with protocol: `max_summary_length=100`, `no_repeat_ngram_size=3`, `length_penalty=1.0`, `early_stopping=True`.
  - Prefix constraint / prompt control preventing generation of media attribution tokens.
  - Evaluation: Measure hallucination rate before and after on a hand-labeled benchmark of 20 test inputs.
* **Retraining Options (Only if Commercial License Required):**
  - *Option A:* Fine-tune on commercial Arabic news/wiki corpus (e.g., Arabic Wikipedia summary dataset).
  - *Option B:* Synthetic teacher generation using permissively licensed LLM (e.g., Llama 3 / Mistral) with rigorous license audit. Volume: $\sim 15,000$ pairs.
* **Recommendation:** **DO NOT RETRAIN.** Deliverable is non-commercial; opening-phrase post-filter and decoding alignment resolve framing hallucination without weight training.

---

### Candidate 3: Retrieval (`CAMeL-BERT` + mMARCO Cross-Encoder)
* **The Problem:**
  - Full ARCD test set (693 queries) yields P@1 of **0.3694**, compared to **0.64** on the small $N=50$ protocol set.
  - Serving pipeline previously used `max_length=512`, whereas evalkit and protocol used `128` words/tokens.
* **Cheapest Fix to Try First:**
  - Protocol Alignment: Pin serving pipeline to `ENCODE_MAX_LENGTH=128` (implemented in Q3).
  - Lexical Baseline: Integrate BM25 sparse retrieval alongside dense bi-encoder (Reciprocal Rank Fusion hybrid retrieval).
  - Dev-Split Tuning: Tune bi-encoder candidate pool size ($K_{\text{dense}} \in \{5, 10, 20\}$) and cross-encoder rerank threshold strictly on a 20% ARCD dev split (preserving the test split).
* **Retraining Gate:**
  - Propose fine-tuning encoder/reranker only if hybrid search yields $<5$ percentage points P@1 improvement on the dev set and user approves compute spend.
* **Recommendation:** **DO NOT RETRAIN YET.** Protocol alignment and BM25 hybrid search should be validated first.

---

### Candidate 4: Whisper Long-Form Hallucination Loops
* **The Problem:**
  - Long-form audio on `whisper-large-v3` measured **31.28% WER** with catastrophic repetition loops.
  - **Loop Reconciliation:** Audit noted 9 catastrophic failure loops (`is_failure_gt_60 == True` in `xlsum_longform_asr_failure_audit.csv`), reconciled against 18 regex substring repetitions. "No VAD" is treated as an unconfirmed hypothesis until Experiment 4.
* **Cheapest Fix to Try First:**
  - Decoding adjustments (already specified in `configs/final_protocol.json`):
    - `vad_filter=True` (Silero VAD windowing).
    - `condition_on_previous_text=False` (prevents hallucination propagation across chunk boundaries).
    - `temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` (fallback on repetition).
    - `compression_ratio_threshold=2.4`, `no_repeat_ngram_size=3`.
* **Recommendation:** **DO NOT RETRAIN.** Hallucination loops in Whisper are well-documented autoregressive decoding phenomena, resolved via decoding constraints.

---

## 3. Retraining Run Protocol on Modal (If Ever Approved)

If any retraining candidate is approved by the user, the execution must strictly follow this protocol:
1. **Cost & Pre-Flight Approval:**
   - Script and config committed to repo with fixed seed (42) and pinned dependencies.
   - GPU selection (NVIDIA A10G or L4) with wall-clock time and cost estimates presented to the user. Execution will pause until explicit user sign-off.
2. **State & Checkpoint Persistence:**
   - Mount persistent `modal.Volume` (`/vol/checkpoints`).
   - Save optimizer, scheduler, and adapter checkpoints every 200 steps with resume support so preempted jobs resume without wasted compute.
3. **Experiment Tracking:**
   - Structured JSON logs committed to the repository (no Weights & Biases per user decision).
4. **Produced Deliverables & Cryptographic Integrity:**
   - Adapter weights, merged FP16/BF16 weights, and CTranslate2 INT8 quantized export.
   - SHA-256 computed and logged for all binaries.
   - Gate verification: new WER $\le 12.51\% \pm 0.25\%$ on the canonical 400 Subset A clips.
5. **Safe Publication & Rollback:**
   - Published to Hugging Face Hub under a new repository name or tagged revision.
   - Rollback guarantee: Serving config retains the previous pinned revision until new artifacts pass gate evaluation.

---

## 4. Modal Production Serving Architecture

### 4.1 Topology & Split-Host Design (Option B)
```
  [User Browser]
         │
         ▼
  [Hugging Face Space] (Gradio UI - Lightweight CPU)
         │  (HTTPS + Secret API Key: AGY_MODAL_API_KEY)
         ▼
  [Modal Private FastAPI App] (GPU: NVIDIA L4 / T4)
         │
         ├─► ASR: CTranslate2 INT8 (Pinned fine-tuned Turbo)
         ├─► Summarizer: AraBART (XL-Sum, post-filter)
         └─► Semantic Search: CAMeL-BERT + FAISS + mMARCO
```

### 4.2 Modal Implementation Specification
* **Syntax & Modern API:**
  - Uses `app = modal.App("smart-lecture-assistant")`.
  - `@app.cls(gpu="L4", concurrency_limit=5, max_concurrent_inputs=2)`.
  - Eager model loading in `@modal.enter()` method to ensure fast cold-start inference.
* **Pinned Container Environment:**
  - Base: Debian 12 slim with Python 3.11.
  - CUDA / cuDNN pairing: PyTorch 2.3+ with CUDA 12.1, CTranslate2 4.4.0 with matching cuDNN libraries (`libcudnn8`).
  - Pre-cached weights stored in container image or warm `modal.Volume`.
* **Spawn-and-Poll Pattern for Long Lectures:**
  - Modal HTTP requests enforce a timeout (default 150 seconds).
  - Lectures longer than 3 minutes are processed asynchronously via `@app.function().spawn()`.
  - Returns `session_id` immediately with status `PROCESSING`; Gradio frontend polls `/status/{session_id}` every 3 seconds until `COMPLETED`.
* **Session Storage & TTL:**
  - Transcripts, cheat sheets, and FAISS indices stored in a `modal.Dict`.
  - Each entry includes a creation timestamp; an hourly background sweeper deletes sessions older than 4 hours (`SESSION_TTL_SECONDS = 14400`).
* **Cost & VRAM Estimates (Labeled as Estimates Until Measured):**
  - *Estimated VRAM Footprint:* ~9.5 GB total (ASR CT2 INT8: ~1.8 GB; AraBART FP16: ~1.6 GB; CAMeL-BERT FP16: ~0.8 GB; mMARCO Cross-Encoder: ~1.2 GB; FAISS + PyTorch Context: ~4.1 GB).
  - *Estimated Compute Cost:* NVIDIA L4 GPU on Modal is $\approx \$0.80$/hour ($\approx \$0.0133$/minute). A 15-minute lecture processed at $6.5\times$ realtime takes $\approx 2.3$ minutes compute ($\approx \$0.03$ per lecture).

---

## 5. Status of Verification & Execution Tasks

| Task | Description | Status | Evidence / Artifact |
| :--- | :--- | :--- | :--- |
| **Q1** | Claims register single source of truth & doc alignment | **DONE** | Commit `389e63f`, `scripts/verify_doc_claims.py` passed |
| **Q2** | Pin serving ASR to evaluated CT2 checkpoint | **DONE** | Commit `6eb4dfa`, `scripts/verify_q2_acceptance.py` passed |
| **Q3** | Decoding and retrieval parameters protocol alignment | **DONE** | Commit `0de4954`, `configs/shared_config.py`, 69 tests passed |
| **Q4** | GitHub Actions CI workflow | **DONE** | Commit `6bfd4d4`, `.github/workflows/ci.yml` |
| **Q5** | API security, SSRF guards, rate limiting | **DONE** | Commit `0fbc16e`, 75 tests passing (`tests/test_api.py`) |
| **EXP1**| 3-arm ASR re-merge verification harness | **DONE** | Commit `e2bb136`, `scripts/experiment_asr_remerge.py` |
| **Q6** | Retraining assessment & register of inputs needed | **DONE** | This document (`RETRAIN_ASSESSMENT.md`) |

---

## 6. Comprehensive Register of Inputs & Decisions

The following table summarizes all decisions, external authorizations, and artifacts needed from the user:

| Item | Description | Owner | Status | Blocking Impact |
| :--- | :--- | :--- | :--- | :--- |
| **1. Deliverable & Licensing** | Define product scope and license. | User | **RESOLVED** (Non-commercial research demo; CC-BY-NC-SA 4.0 badges added) | None (resolved) |
| **2. Architecture Choice** | Serving topology. | User | **RESOLVED** (Option B: HF Space Gradio calling private Modal FastAPI) | None (resolved) |
| **3. Served ASR Selection** | Target production ASR checkpoint. | User | **RESOLVED** (Fine-tuned Whisper-Large-v3-Turbo CT2 INT8 pinned in pipeline) | None (resolved) |
| **4. Experiment Tracking** | Logging infrastructure. | User | **RESOLVED** (JSON logs committed to repository; no W&B) | None (resolved) |
| **5. EXP1 GPU Budget** | Budget for 3-arm re-merge on 400 clips. | User | **RESOLVED** (Approved up to $5.00 total; estimated cost $0.16 on Modal L4) | Unblocks EXP1 live run once adapters fetched |
| **6. Missing Adapter Safetensors** | Download Large-v3 and Turbo `adapter_model.safetensors` from Kaggle kernel outputs or supply path. | User | Pending user download / Kaggle CLI execution | Blocks live execution of Arms 1-3 on GPU |
| **7. Modal Token & Secrets** | Setup Modal credentials and `lecture-assistant-secret` on Modal account. | User | Handled by user (in progress) | Blocks running live Modal deploy (`modal deploy`) |
| **8. HF Space Secrets** | Add `MODAL_API_URL` and `API_BEARER_TOKEN` in HF Space repository settings. | User | Pending live Modal deployment | Blocks end-to-end integration of HF Space |
| **9. Served ASR Claims Register Entry** | Register measured WER on 400 clips for exact served CT2 INT8 Turbo (`SERVED_TURBO_CT2_INT8_GATE_WER`). | ML Eng / User | Scheduled for live GPU run | Will finalize ASR serving claims |

---
*End of Retraining Assessment.*

