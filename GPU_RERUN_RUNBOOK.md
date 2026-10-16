# GPU RERUN RUNBOOK (READY-TO-RUN SPECIFICATION)

> **CRITICAL PROTOCOL NOTICE:**  
> This document specifies the exact configurations, input datasets, probe-based compute runtimes, and output file expectations for executing GPU evaluations.  
> **DO NOT EXECUTE GPU WORK LOCALLY.** This runbook is prepared for the external reviewer or authorized cluster operator.

---

## 1. Summary of Rerun Runs (R1 to R5)

| Run ID | Scope & Objective | Hardware Target | Audio / Item Volume | Measured Probe RTF | Expected Wall Time | Primary Output Artifacts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **R1** | Zero-shot Base Model Benchmark (Prompt Diagnostic) | Tesla T4 (16 GB) | 400 clips (Subset A) x 4 models | 0.1471 (FP16) | **12.8 minutes** | `zero_shot_diagnostic.json`, `zero_shot_hypotheses.txt` |
| **R2** | Pass 4 Architecture Matrix (1,000 clips, seed 42) | Tesla T4 (16 GB) | 1,000 clips x 3 models | 0.1745 (CT2 INT8) | **27.3 minutes** | `cv_pass4_matrix.json`, `pass4_clip_ids.json`, `cv_pass4_*_clips.csv` |
| **R3** | Pass 2 & Beam-5 with Per-Clip Itemized Logging | Tesla T4 (16 GB) | 1,400 clips (Pass 2 + Pass 3) | 0.1745 (P2) / 0.5584 (P3) | **74.2 minutes (1.24 hrs)** | `cv_pass2_clips.csv`, `cv_pass3_beam5_clips.csv` |
| **R4** | Downstream Summarization & Retrieval with Oracle Controls | Tesla T4 (16 GB) | 50 articles / 50 queries | 0.964 s/art, 0.652 s/query | **8.6 minutes** | `downstream_retrieval_summarization.json`, per-article ROUGE |
| **R5** | 4-bit NF4 Unmerged Training Replica Gate Check | Tesla T4 (16 GB) | 400 clips (Subset A) | 0.2600 (NF4 bitsandbytes) | **7.5 minutes** | `cv_gate_subset_a_clips.csv` |

**Total Estimated GPU Compute Across All Runs**: **~130.4 minutes (~2.17 hours)** on a single NVIDIA Tesla T4 GPU.

---

## 2. Detailed Run Specifications

### Run R1: Zero-Shot Baseline Anomaly Resolution
- **Notebook**: `Notebooks/final_01c_cv_wer_matrix.ipynb` (Cell 3 Diagnostic)
- **Purpose**: Verify that vanilla zero-shot Whisper models achieve expected baseline WERs (18.35% for large-v3, 27.36% for turbo, 30.95% for medium) when prompted with `<|ar|>` `<|transcribe|>`, whereas unprompted execution explains the ~100% preflight WER due to default English language hallucination.
- **Required Input Datasets**:
  1. `common-voice-scripted-speech-25-0-arabic` (`test.tsv`, `clips/`)
  2. `final-merged-models` (or Hugging Face Hub access for base checkpoints)
- **Decoding Configuration**:
  - `language`: `"ar"`
  - `task`: `"transcribe"`
  - `num_beams`: `1` (Greedy)
  - `temperature`: `0.0`
- **Output Files to Collect**:
  - `Results/final/zero_shot_diagnostic.json`
  - Printed 20-clip hypothesis comparison log

---

### Run R2: Pass 4 Benchmark Matrix (1,000 Random Clips, Seed 42)
- **Notebook**: `Notebooks/final_01c_cv_wer_matrix.ipynb` (Cells 2 and 4)
- **Purpose**: Multi-architecture scaling comparison on an identical 1,000-clip partition across `Whisper-Large-v3`, `Whisper-Large-v3-Turbo`, and `Whisper-Medium`.
- **Required Input Datasets**:
  1. `common-voice-scripted-speech-25-0-arabic`
  2. `final-merged-models` (`ct2_int8/`)
  3. `evalkit-src.zip`
- **Partition Rule**: `df_valid.sample(n=1000, random_state=42)`
- **Output Files to Collect**:
  - `Results/final/pass4_clip_ids.json` (exact list of 1,000 clip paths)
  - `Results/final/cv_pass4_matrix.json`
  - `Results/final/cv_pass4_matrix_summary.csv`
  - `Results/final/cv_pass4_{model}_clips.csv` (per-clip S/D/I/H/N)

---

### Run R3: Pass 2 CT2 Greedy & Pass 3 Beam-5 Per-Clip Logging
- **Notebook**: `Notebooks/final_01b_cv_pass2_beam5.ipynb`
- **Purpose**: Generate granular per-clip CSVs for CT2 INT8 inference to enable independent external audit recomputation of edit distances.
- **Required Input Datasets**:
  1. `common-voice-scripted-speech-25-0-arabic`
  2. `final-merged-models` (`ct2_int8/whisper-large-v3-ct2-int8`)
  3. `evalkit-src.zip`
- **Output Files to Collect**:
  - `Results/final/cv_pass2_clips.csv` (10,506 rows)
  - `Results/final/cv_pass3_beam5_clips.csv` (1,400 rows)
  - `Results/final/cv_pass2_ct2_int8.json`
  - `Results/final/cv_beam5_vad_comparison.json`

---

### Run R4: Downstream Evaluation with Oracle Controls & True NDCG@3
- **Notebook**: `Notebooks/final_03_summarization_retrieval.ipynb`
- **Purpose**: Complete end-to-end evaluation including:
  1. Pure Bi-Encoder (`CAMeL-BERT`) vs. Bi-Encoder + Cross-Encoder (`mMARCO`) retrieval
  2. True NDCG@3 (`DCG@3 / IDCG@3`) per query
  3. Full query question, golden answer, and retrieved passage IDs
  4. Oracle reference summaries and normalized reference controls
  5. Per-article ROUGE scores (`rouge1`, `rouge2`, `rougeL`)
  6. Exact paired McNemar tests and Holm-Bonferroni correction
- **Required Input Datasets**:
  1. `tts-asr-propagation-results` (`tts_asr_propagation_results.json`, `spoken_arcd_transcripts.json`)
  2. `final-merged-models` (`downstream/`)
  3. BBC Arabic XL-Sum test set
  4. ARCD (`train.csv`)
  5. `evalkit-src.zip`
- **Output Files to Collect**:
  - `Results/final/downstream_retrieval_summarization.json`
  - `Results/final/downstream_summarization_summary.csv`
  - `Results/final/spoken_retrieval_results.json`
  - `Results/final/spoken_retrieval_summary.csv`

---

### Run R5: 4-bit NF4 Unmerged Training Replica Gate
- **Notebook**: `Notebooks/final_01a_cv_gate_pass1.ipynb` (Cell 5)
- **Purpose**: Verify that the exact training setup (4-bit NF4 base model + unmerged LoRA adapter) replicates the published 12.51% WER within a 0.25 pp tolerance window (`[12.26%, 12.76%]`).
- **Required Input Datasets**:
  1. `common-voice-scripted-speech-25-0-arabic`
  2. `final-merged-models` (`base/whisper-large-v3`, `adapters/whisper-large-v3-adapter`)
  3. `evalkit-src.zip`
- **Output Files to Collect**:
  - `Results/final/cv_gate_subset_a_clips.csv` (400 rows)

---

## 3. Post-Run Collection & Verification Procedure

Upon receiving the generated JSON and CSV files from the GPU runs:
1. Copy all output files into `Results/final/`.
2. Run automated validation:
   ```bash
   python scripts/verify_rerun.py
   ```
3. Run provenance check:
   ```bash
   python scripts/verify_provenance.py
   ```
4. Regenerate review packet:
   ```bash
   python scripts/review_generate_report.py
   python scripts/review_bundle_builder.py
   ```
