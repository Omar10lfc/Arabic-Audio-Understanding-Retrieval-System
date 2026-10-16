# Smoke & Full Runbook: Arabic Audio Evaluation Protocol

This runbook specifies the required execution order, Kaggle environment configuration, input datasets, manual dataset creation steps, measured runtimes (with citations to source result files), and exact pass/fail criteria across all benchmark stages.

---

## 1. Global Setup & Requirements

- **Platform:** Kaggle Notebooks (Ubuntu 20.04 LTS container)
- **Primary Source Archive:** `evalkit-src` (`/kaggle/input/evalkit-src/evalkit-src.zip` or `/kaggle/input/evalkitsrc/evalkit-src.zip`)
- **Expected `evalkit-src.zip` SHA-256:** `66d16ba721a0050820996c5998b3be9412057faa978fd98d331015f6065819ad`
- **Output Hygiene:**
  - `RUN_MODE="smoke"`: Writes to `Results/smoke/`
  - `RUN_MODE="full"`: Writes to `Results/final/`
  - Environment logged to `Results/env/<notebook_stem>_<mode>_env.txt`

---

## 2. Manual Kaggle Dataset-Creation Steps

Before executing the offline evaluation pipeline, create and mount the required private Kaggle datasets following these exact manual steps:

### Dataset 1: `omar10lfc/evalkit-src`
1. On your local terminal, package the evaluation suite:
   ```bash
   python scripts/package_evalkit_src.py
   ```
2. In the Kaggle web UI, navigate to **Datasets** $\rightarrow$ **New Dataset**.
3. Drag and drop `evalkit-src.zip`.
4. Set Title: `evalkit-src` (Slug: `omar10lfc/evalkit-src`).
5. Click **Create**.
6. Verify that Cell 1 in each notebook confirms SHA-256 matches `66d16ba721a0050820996c5998b3be9412057faa978fd98d331015f6065819ad`.

### Dataset 2: `omar10lfc/final-merged-models`
1. Create a new Kaggle notebook and import `Notebooks/final_00_merge_models.ipynb`.
2. Configure settings: Accelerator: **GPU T4 x2**, Internet: **ON**, Persistence: **Files Only**.
3. Mount the three adapter checkpoints:
   - `omar10lfc/fine-tuning-whisper-large-v3-on-arabic-speech`
   - `omar10lfc/whisper-large-v3-turbo-arabic-qlora`
   - `omar10lfc/whisper-medium-arabic-stage2`
4. Run all cells to completion (~15 minutes).
5. Cell 3 saves base weights, adapters, merged FP16, and CT2 INT8 for all 4 models, plus the 3 downstream models into `/kaggle/working/final-merged-models/`.
6. In Kaggle's right-hand panel under **Output**, click **...** next to `final-merged-models` $\rightarrow$ **New Dataset from Output**.
7. Title: `final-merged-models` (Slug: `omar10lfc/final-merged-models`).
8. Click **Create**. Mount this dataset in N1a, N1b, N1c, N2, and N3.

### Dataset 3: `omar10lfc/tts-asr-propagation-results`
1. Run N2 (`Notebooks/final_02_tts_asr.ipynb`) with Accelerator: **GPU T4 x1**, Internet: **OFF**.
2. Mount inputs: `evalkit-src`, `arabic-xl-sum`, `arabic-reading-comprehension-dataset-arcd`, and `final-merged-models`.
3. Cell 4 transcribes the 50 XL-Sum articles, 50 ARCD passages, and 50 ARCD questions across all 4 architectures.
4. From the notebook outputs (`tts_asr_propagation_results.json` and `spoken_arcd_transcripts.json`), create dataset `omar10lfc/tts-asr-propagation-results`.
5. Mount `tts-asr-propagation-results` into N3 (`final_03_summarization_retrieval.ipynb`).

---

## 3. Notebook Execution Matrix & Measured Runtimes

| # | Notebook | Kaggle Accelerator | Internet | Persistence | Measured Runtime (Full) | Measured Throughput / Citation |
|---|---|---|---|---|---|---|
| **N0** | `Notebooks/final_00_merge_models.ipynb` | GPU T4 x2 | **ON** | Files Only | ~15 mins *(estimate)* | Multi-model download, merge & CT2 export |
| **N1a** | `Notebooks/final_01a_cv_gate_pass1.ipynb` | GPU T4 x1 | **OFF** | None | **2.73 hours** | Gate: 400 clips @ 5.0x (6 min); Pass 1: 10,506 clips @ 5.0x (2.63 h) *(PyTorch FP16 Greedy; uses preinstalled transformers/peft)* |
| **N1b** | `Notebooks/final_01b_cv_pass2_beam5.ipynb` | GPU T4 x1 | **ON** *(pip install)* | None | **1.00 hour** | Pass 2: 10,506 clips @ 18.0x (0.73 h); Pass 2b: 1,400 clips @ 6.5x (0.27 h) *(CT2 INT8; loads models & data offline)* |
| **N1c** | `Notebooks/final_01c_cv_wer_matrix.ipynb` | GPU T4 x1 | **ON** *(pip install)* | None | **0.55 hours** | Small (12.4x), Med (8.8x), Turbo (8.5x), Large (18.0x) [`whisper_large_benchmark_results.json`] |
| **N2** | `Notebooks/final_02_tts_asr.ipynb` | GPU T4 x1 | **ON** *(pip install)* | None | **0.70 hours** | 4 models $\times$ 13,051.7 s audio @ $\ge 12.0\times$ continuous throughput *(resumable per item)* |
| **N3** | `Notebooks/final_03_summarization_retrieval.ipynb` | GPU T4 x1 | **OFF** | None | **0.35 hours** | AraBART (batch size 8: 18 min) + Retrieval Sets i, ii, iii (3 min) [`spoken_retrieval_results.json`] |

*Note on Measured vs. Estimated figures:*
- **Measured numbers** (cited directly from saved results):
  - Whisper-Small throughput: **12.4x** (`Results/legacy/whisper_large_benchmark_results.json`)
  - Whisper-Medium CT2 throughput: **8.8x** (`Results/legacy/quantization_benchmark_results.json`)
  - Whisper-Large-v3-Turbo CT2 throughput: **8.5x** (`Results/legacy/whisper_large_benchmark_results.json`)
  - Whisper-Large-v3 PyTorch throughput: **5.0x** (`Results/legacy/cv_wer_gap_isolation_results.json`)
  - Whisper-Large-v3 CT2 INT8 throughput: **18.0x** (`Results/legacy/cv_wer_gap_isolation_results.json`)
  - Whisper-Large-v3 Training WER: **12.51%** (`Results/results-Whisper-large-v3-qlora.json`)
  - Whisper-Large-v3-Turbo Training WER: **14.15%** (`Results/results-Whisper-large-v3-turbo-qlora.json`)
  - Whisper-Medium Stage 2 WER: **18.16%** (`Results/results-Whisper-medium-stage2.json`)
  - Whisper-Small Fine-Tuned WER: **20.61%** (`Results/results-Whisper-finetuned.json`)
- All other figures (e.g. N0 Hugging Face download duration) are marked **estimate**.

---

## 4. Pass/Fail Criteria & 200-Clip Timing Probe

### Timing Probe Protocol (N1a, N1b, N1c, N2)
- Cell 2 loads the first 200 clips/items and executes a timed probe.
- It computes the empirical duration per item and projects total notebook runtime:
  $$\text{Projected Total} = \frac{T_{200}}{200} \times N_{\text{total}}$$
- If Projected Total exceeds the allocated Kaggle session threshold (N1a > 3.5 hours; N1b > 1.5 hours; N2 > 1.0 hour), the cell halts execution with `AssertionError("Runtime budget exceeded")`.

### Retrieval Sets (i), (ii), (iii) Pass/Fail Criteria (N3)
N3 evaluates spoken document semantic retrieval across 4 conditions:
1. **Oracle Clean Baseline:** Clean Text Query $\rightarrow$ Clean Text Passages.
   - P@1 must equal $0.5000$ on Bi-Encoder (25/50 hits) on the canonical 121-chunk ARCD corpus.
2. **Set (i) [Text Query $\rightarrow$ Spoken Passages]:**
   - Transcribed by each of the 4 models (Large-v3, Turbo, Medium, Small).
   - **PASS:** Valid retrieval metrics ($P@1 \in [0.30, 0.90]$, $MRR \in [0.40, 0.95]$); Cross-Encoder P@1 $\ge$ Bi-Encoder P@1 for all models.
3. **Set (ii) [Spoken Query $\rightarrow$ Clean Passages]:**
   - 50 questions synthesized and transcribed by Whisper.
   - **PASS:** Query transcription non-empty for all 50 questions; Quality retention $\ge 70\%$ relative to Oracle.
4. **Set (iii) [Spoken Query $\rightarrow$ Spoken Passages]:**
   - Full acoustic distortion on both queries and passages.
   - **PASS:** Compound degradation bounded; retrieval metrics monotonically consistent or within expected acoustic noise bounds.
