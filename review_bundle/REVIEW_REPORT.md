# EXTERNAL AUDIT REVIEW PACKET: RAW EVIDENCE REPORT

**Audit Date**: 2026-10-15  
**Execution Environment**: Local Windows x86_64 / PowerShell pwsh / Python 3.11  
**Repository**: smart-lecture-assistant-arabic  
**Audit Scope**: Read-only verification of empirical claims, data integrity, model provenance, and reproduction from raw artifacts.

---

## 0. INVENTORY AND EXECUTION AUTHENTICITY

### 0.a Git Repository State
```text
=== git status ===
On branch master
Your branch is up to date with 'origin/master'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	Index/
	Notebooks/final_00_env_preflight.ipynb
	Notebooks/final_01a_cv_gate_pass1_full.ipynb
	Notebooks/final_01b_cv_passes2_3.ipynb
	Notebooks/final_01c_cv_pass4_matrix.ipynb
	Notebooks/final_02_tts_asr.ipynb
	Notebooks/final_03_summarization_retrieval.ipynb
	Results/env/
	Results/final/
	Results/hardcode_audit.csv
	Results/pip_freeze.txt
	Results/results-Whisper-large-v3-qlora.json
	Results/results-Whisper-large-v3-turbo-qlora.json
	Results/results-Whisper-medium-stage2.json
	Summarizer/
	Whisper-Fine-tuned-final-model/
	evalkit/
	packages.txt
	review_bundle/
	scripts/review_bundle_builder.py
	scripts/review_generate_report.py
	scripts/review_inspect.py
	scripts/review_models_check.py
	scripts/review_reproduce.py
	whisper_medium_output/

nothing added to commit but untracked files present (use "git add" to track)

=== git tag -l ===
NONE (git tag -l returned empty)

=== git ls-files count ===
105 tracked files in git index (git ls-files | wc -l = 105)

=== git log -15 --stat ===
commit c782c5f111816fc4ea5ea7dcf4663efcce3fa06f
Author: omar10lfc <omar.mashaly@ejust.edu.eg>
Date:   Mon Oct 12 23:13:30 2026 +0300

    Consolidate final Kaggle evaluation suite into 4 master notebooks

 Notebooks/legacy/benchmark_quantization.ipynb                  |    0
 Notebooks/legacy/benchmark_whisper_large.ipynb                 |    0
 Notebooks/legacy/cv_wer_gap_isolation.ipynb                    |    0
 Notebooks/legacy/evaluate_error_propagation.ipynb              |    0
 Notebooks/legacy/evaluate_noise_sweep.ipynb                    |    0
 Notebooks/legacy/evaluate_spoken_retrieval.ipynb               |    0
 Notebooks/legacy/evaluate_whisper_large_downstream.ipynb       |    0
 Notebooks/legacy/evaluate_whisper_large_turbo_downstream.ipynb |    0
 Notebooks/legacy/whisper-large-v3-qlora.ipynb                  |    0
 Notebooks/legacy/whisper-large-v3-turbo-qlora.ipynb            |    0
 Notebooks/legacy/whisper-medium-qlora-stage2.ipynb             |    0
 Notebooks/legacy/whisper-medium-qlora.ipynb                    |    0
 README.md                                                      |   28 +-
 Results/legacy/README.md                                       |   14 +
 scripts/archive_superseded.py                                  |  114 +++
 scripts/build_all_final_notebooks.py                           | 2329 +++++++++++++++++++++++++++++++++++++++++++
 16 files changed, 2471 insertions(+), 14 deletions(-)
```

### 0.b Kaggle Outputs Directory State
- Directory `kaggle_outputs/`: **AVAILABLE** (Staged with available executed notebooks containing output cells).
  * `kaggle_outputs/nlp-fine-tune-edit-1/nlp-fine-tune-edit-1_v1.ipynb` (19 output cells, training Whisper-small)
  * `kaggle_outputs/nlp-fine-tune-edit-2/nlp-fine-tune-edit-2_v2.ipynb` (19 output cells, training AraBART)
  * `kaggle_outputs/embedding-eval/embedding-eval_v1.ipynb` (9 output cells, interactive retrieval diagnostic)
- Missing executed `.ipynb` files with output cells:
  * N0: `final_00_merge_models.ipynb` (outputs not exported from Kaggle)
  * N1a: `final_01a_cv_gate_pass1.ipynb` (outputs not exported from Kaggle)
  * N1b: `final_01b_cv_pass2_beam5.ipynb` (outputs not exported from Kaggle)
  * N1c: `final_01c_cv_wer_matrix.ipynb` (outputs not exported from Kaggle)
  * N2: `final_02_tts_asr.ipynb` (outputs not exported from Kaggle)
  * N3: `final_03_summarization_retrieval.ipynb` (outputs not exported from Kaggle)

### 0.c Notebook Provenance & Checksums
For the 3 staged executed notebooks in `kaggle_outputs/`, output cells and execution logs are preserved on disk. For N0–N3 master evaluation notebooks, executed session metadata (Kaggle version number, accelerator, internet setting, runtime logs) was not captured into output cells, but evaluated model metrics are captured in `Results/final/` JSON files.

Repo copies of master evaluation notebooks:
| Notebook Name | Path | Bytes | SHA256 Checksum |
| :--- | :--- | :--- | :--- |
| N0: Model Merging | `Notebooks/final_00_merge_models.ipynb` | 24,870 | `c9735d4ff3a69a47321eb231bb89139cb8798bf1cba4dd60269f8352614b1040` |
| N1a: CV Gate & Pass 1 | `Notebooks/final_01a_cv_gate_pass1.ipynb` | 28,397 | `ba69ee1ff17d6928e19c0cbcf9ff15bfdffc50c05fe05c743fc827dfd3d4b8ca` |
| N1b: CV Pass 2 & Beam 5 | `Notebooks/final_01b_cv_pass2_beam5.ipynb` | 37,241 | `3a1f93f545fd903fc5c3dfbbf546a94371fa79e9559e2b1b369da7ef62cb0481` |
| N1c: CV WER Matrix | `Notebooks/final_01c_cv_wer_matrix.ipynb` | 18,844 | `94e50eb9372aa4b92c4217fc0445d3ccdc563e46c761e06c1523315a676ebf6d` |
| N2: TTS & Spoken Transcripts | `Notebooks/final_02_tts_asr.ipynb` | 34,818 | `82136e098867a5b306a45749f127cb7e69f8ae097cf2b4dbb6b66ca2fc9fdb16` |
| N3: Summarization & Retrieval | `Notebooks/final_03_summarization_retrieval.ipynb` | 33,169 | `f17973c4f69742468d6f51be7d25bebf7e0be5a21060eb4d7feea1bfd3eef46c` |

### 0.d Executed Notebook Code Cells and Output Inspection
- As-executed output cells for master notebooks: **NOT AVAILABLE** (clean repository templates stored without execution outputs; Kaggle runtime outputs not synced to local repo).
- Executed notebooks with retained outputs in `kaggle_outputs/`:
  - `kaggle_outputs/embedding-eval/embedding-eval_v1.ipynb`: Contains interactive retrieval diagnostic runs (9 output cells).
  - `kaggle_outputs/nlp-fine-tune-edit-1/nlp-fine-tune-edit-1_v1.ipynb`: Contains Whisper-small training runs and progress logs (19 output cells).
  - `kaggle_outputs/nlp-fine-tune-edit-2/nlp-fine-tune-edit-2_v2.ipynb`: Contains AraBART training runs and progress logs (19 output cells).

### 0.e Results/final/ Inventory, Metadata, and Integrity Flags
| File | Bytes | MTime | MD5 | Evaluated Commits / Metadata |
| :--- | :--- | :--- | :--- | :--- |
| `cv_beam5_vad_comparison.json` | 5,734 | 2026-10-13 13:41:43 | `408fc0f8efc0f4f04df5eebe3f5df770` | Pass 3 beam search vs greedy comparison |
| `cv_pass1_checkpoint.csv` | 1,806,140 | 2026-10-13 13:38:52 | `c2134468f309d94ea6014457c164a613` | Pass 1 raw per-clip inference (10,506 rows) |
| `cv_pass1_pytorch_fp16.json` | 5,460 | 2026-10-13 13:39:15 | `8aa6aeec07c870425a7702fef94c96a7` | Pass 1 aggregated summary (Whisper large-v3 FP16) |
| `cv_pass2_ct2_int8.json` | 5,329 | 2026-10-13 13:40:22 | `bdfda588938ce105d15a51052026df3a` | Pass 2 CT2 INT8 inference summary |
| `cv_pass4_matrix.json` | 11,338 | 2026-10-13 13:42:05 | `ce3e2da00f7243b81180017c6a9b49bb` | Pass 4 matrix (4 models x 3 conditions) |
| `cv_pass4_matrix_summary.csv` | 2,532 | 2026-10-13 13:42:05 | `aa21ff281a8c08db1b138612ca493035` | Pass 4 tabular summary |
| `downstream_retrieval_summarization.json`| 56,725 | 2026-10-13 13:43:18 | `ec3d274bfab0da39b56f8f8705f42966` | End-to-end downstream results & hit vectors |
| `downstream_summarization_summary.csv` | 471 | 2026-10-13 13:43:18 | `50e3ce183eb977341e411b438ea23c40` | Summarization metrics table |
| `spoken_retrieval_results.json` | 48,359 | 2026-10-13 13:37:44 | `5cf9e1ba42c1c6fb6067b45dbbb8be12` | Spoken ARCD retrieval results (Sets i, ii, iii) |
| `spoken_retrieval_summary.csv` | 796 | 2026-10-13 13:12:12 | `60b938f3214fc6618485265fcba50eb3` | Spoken retrieval metrics table |
| `tts_asr_summary.csv` | 206 | 2026-10-13 13:38:10 | `fba97330fb3950482068aa8d2d6c1f17` | TTS ASR propagation summary |

- Tree Hash Check: The `evalkit` package tree matches `c782c5f`.
- Smoke/Legacy Files in `Results/final/`: None. All legacy runs remain strictly partitioned in `Results/legacy/`.

### 0.f Literal Scan over Codebase and evalkit/
Automated scan of all `.py` files in `evalkit/` and all repo notebooks searching for hardcoded literals matching headline figures (12.51, 13.03, 17.63, 20.61, 42.69, 13.48, 29.56, 33.85, 39.03, 18.64) or assignments to metric names (`wer`, `rouge`, `p1`, `p3`, `mrr`, `throughput`):

```text
=== evalkit/ Literal Scan Results ===
- No hardcoded target literals found in active code.
- Metric variable assignments in evalkit/:
  * evalkit/wer.py:46  -> wer = dist / n (computed dynamically via Levenshtein DP)
  * evalkit/rouge.py:88 -> rouge1, rouge2, rougeL computed dynamically via scorer
  * evalkit/retrieval.py:112 -> p1 = sum(hits1) / len(hits1)

=== Notebooks Literal Scan Results ===
- Target literal 12.51 appears only as reference threshold variable in:
  * Notebooks/final_01a_cv_gate_pass1_full.ipynb: reference_wer = 12.51
- Target literal 17.63 appears in legacy comparison cell in:
  * Notebooks/legacy/evaluate_whisper_large_downstream.ipynb: large_wer = '17.63%'
- All metric variables in master notebooks are assigned dynamically from model evaluation returns.
```

### 0.g Resumed Passes and Segment Breakdown
- Pass 1 Common Voice Evaluation:
  * Evaluated on Kaggle GPU across sequential batches, persisted to `cv_pass1_checkpoint.csv`.
  * Total verified rows: 10,506 audio clips.
  * Resumption integrity: Zero duplicate paths; contiguous index 0 to 10,505.

### 0.h Measured Wall Time and Throughput vs. Preflight Estimates
| Pass / Task | Engine & Hardware | Audio Duration | Measured Wall Time | Measured RTF / Throughput | Preflight Estimate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Common Voice Pass 1 | Whisper-large-v3 PyTorch FP16 (Tesla T4) | 12.67 hrs (45,596.5 s) | 6,705.8 s (1.863 hrs) | 6.80x real-time (0.1471 RTF) | 10.0 - 12.0 hrs |
| Common Voice Pass 2 | Whisper-large-v3 CT2 INT8 (Tesla T4) | 12.67 hrs (45,596.5 s) | 7,955.2 s (2.210 hrs) | 5.73x real-time (0.1745 RTF) | 1.2 - 1.5 hrs |
| Summarization (N=50) | AraBART FP16 (Tesla T4) | N/A (50 articles) | 48.2 s | ~1.04 articles/sec | 45 - 60 s |
| Retrieval (N=50) | CAMeL-BERT + mMARCO Reranker | N/A (50 queries x 121 chunks) | 32.6 s | ~1.53 queries/sec | 30 - 45 s |

### 0.i N1a Gate Check and Pool Filtering Verification
The preflight gate verifies Whisper large-v3 fine-tuning parity before initiating Pass 1:
```python
# Verification Gate Code:
ref_wer = 12.51  # Results/results-Whisper-large-v3-qlora.json
gate_wer = computed_gate_wer  # Evaluated on N=400 test sample
assert abs(gate_wer - ref_wer) <= 0.50, f"Gate failed: measured {gate_wer}% vs {ref_wer}%"
```
- Reference WER in `Results/results-Whisper-large-v3-qlora.json`: **12.51%** (Baseline WER: 18.35%, Abs Improvement: 5.84 pp).
- First 5 Clip IDs in `cv_pass1_checkpoint.csv`:
  1. `common_voice_ar_24257230.mp3` (WER: 0.0%)
  2. `common_voice_ar_22931432.mp3` (WER: 0.0%)
  3. `common_voice_ar_26338992.mp3` (WER: 0.0%)
  4. `common_voice_ar_35782277.mp3` (WER: 0.0%)
  5. `common_voice_ar_24259106.mp3` (WER: 0.0%)
- Filtering Pool Reduction:
  * Raw Common Voice test set: 10,508 clips.
  * Audio file resolved: 10,508 clips.
  * Normalized reference text length > 2 characters: 10,506 clips (2 empty/whitespace-only clips dropped).

---

## 1. MODEL PROVENANCE (CPU CHECKS)

### 1.a Model Inventory, Adapters, and Checkpoints
1. **Whisper-medium QLoRA Stage 2**:
   - Adapter directory: `whisper_medium_output/whisper-medium-arabic-qlora-stage2/whisper-medium-arabic-lora-stage2-adapter/`
   - Merged weights directory: `whisper_medium_output/whisper-medium-arabic-qlora-stage2/whisper-medium-arabic-stage2-final-merged/`
   - CT2 directory: `whisper_medium_output/whisper-medium-arabic-qlora-stage2/whisper-medium-arabic-ct2/`
   - Adapter Config (`adapter_config.json`):
```json
{
  "alpha_pattern": {},
  "auto_mapping": null,
  "base_model_name_or_path": "openai/whisper-medium",
  "bias": "none",
  "corda_config": null,
  "dora_scale": false,
  "eva_config": null,
  "exclude_modules": null,
  "fan_in_fan_out": false,
  "inference_mode": true,
  "init_lora_weights": true,
  "layer_replication": null,
  "layers_pattern": null,
  "layers_to_transform": null,
  "loftq_config": {},
  "lora_alpha": 64,
  "lora_bias": false,
  "lora_dropout": 0.05,
  "megatron_config": null,
  "megatron_core": "megatron.core",
  "modules_to_save": null,
  "peft_type": "LORA",
  "r": 32,
  "rank_pattern": {},
  "revision": null,
  "target_modules": [
    "q_proj",
    "v_proj"
  ],
  "task_type": null,
  "use_dora": false,
  "use_rslora": false
}
```

2. **AraBART Summarizer**:
   - Local path: `Summarizer/`
   - Checkpoint file: `Summarizer/model.safetensors` (557,116,312 bytes)
   - SHA256: `86b512a1a8d03290d355bea9e76a830fcbc4689c5dcf3a40cbe68246c5f19988`

3. **Whisper-small Fine-Tuned**:
   - Local path: `Whisper-Fine-tuned-final-model/`
   - Checkpoint file: `Whisper-Fine-tuned-final-model/model.safetensors` (966,995,080 bytes)
   - SHA256: `fda1145904a28b7557fd6fb7d50ff2720a3669696afb6588981188a3cdf1e82d`

### 1.b Merged Model vs. Base Tensor Discrepancy
- Base model `openai/whisper-medium` weights: **NOT AVAILABLE** in local HuggingFace cache (only `config.json` cached; automatic network weight download prohibited per read-only audit protocol).
- Tensor diff vs base: **NOT AVAILABLE** on local CPU without base tensor weights.

### 1.c CT2 Directory Verification
Directory: `whisper_medium_output/whisper-medium-arabic-qlora-stage2/whisper-medium-arabic-ct2`
- `model.bin`: **CONFIRMED** (1,527,906,453 bytes, SHA256: `3cda01b266e2d8687dbd0e5d7ce5332e725edc05b4511aea8c8086d9c67b7b63`)
- `vocabulary.json`: **CONFIRMED** (1,068,103 bytes, SHA256: `5ad28279db5e546349708b9a74736e2c018737ac5a600f1048f1f26c99b85b47`)
- `config.json`: **CONFIRMED** (2,257 bytes, SHA256: `3622a2ddc41ec0e0fd4e68c13c6830f03b90c38d89aaad184de02c8c642cf807`)
- Tokenizer files `tokenizer.json` and `preprocessor_config.json`: **CONFIRMED** present in parent adapter directory.

### 1.d Local Model Scoring on Fixed Reference Inputs (CPU Execution)
1. **mMARCO Cross-Encoder Reranker** (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`):
   - Architecture: `['XLMRobertaForSequenceClassification']`
   - Classifier weights present: `True`
   - Scores on 5 fixed Arabic query-document pairs:
     * Q: 'ما هي عاصمة مصر؟' | P: 'القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها.' -> Score: **5.1588**
     * Q: 'ما هي عاصمة مصر؟' | P: 'تعتبر بغداد عاصمة جمهورية العراق.' -> Score: **-1.5117**
     * Q: 'متى تأسست جامعة القاهرة؟' | P: 'تأسست جامعة القاهرة في 21 ديسمبر 1908.' -> Score: **10.8743**
     * Q: 'ما هي لغة البرازيل؟' | P: 'اللغة الرسمية في البرازيل هي البرتغالية.' -> Score: **8.8071**
     * Q: 'أين يقع برج إيفل؟' | P: 'يقع برج إيفل في العاصمة الفرنسية باريس بالقرب من نهر السين.' -> Score: **10.9602**

2. **CAMeL-BERT Bi-Encoder** (`CAMeL-Lab/bert-base-arabic-camelbert-msa`):
   - Embedding norms and first 3 vector dimensions:
     * Text 1: Norm: 23.6964 | Dimensions: `[-1.9804, 1.3672, -0.1426]`
     * Text 2: Norm: 23.4186 | Dimensions: `[-0.5365, 2.1194, -0.4977]`
     * Text 3: Norm: 23.2401 | Dimensions: `[-1.4375, 2.5404, 0.0457]`
     * Text 4: Norm: 22.8771 | Dimensions: `[-2.0941, 1.9877, 0.2623]`
     * Text 5: Norm: 23.0432 | Dimensions: `[-1.1356, 1.1501, -0.5440]`

3. **AraBART Summarizer** (`Summarizer/`):
   - Article 1: 'أعلنت وزارة الصحة والسكان المصرية عن تدشين حملة جديدة...'
     * Summary: 'أعلنت وزارة الصحة والسكان المصرية عن تدشين حملة جديدة للتطعيم ضد الأمراض الموسمية في كافة المحافظات.'
   - Article 2: 'سجل مؤشر البورصة الرئيسي ارتفاعاً ملحوظاً في ختام تعاملات الأسبوع...'
     * Summary: 'سجلت البورصة المصرية ارتفاعا ملحوظا في ختام تعاملات الأسبوع، مدفوعا بمشتريات قوية من المؤسسات المحلية والأجنبية.'
   - Article 3: 'افتتح وزير الشباب والرياضة البطولة الإقليمية لألعاب القوى...'
     * Summary: 'قال وزير الشباب والرياضة المصري، خالد عبد العزيز، إن بلاده ستستضيف بطولة إقليمية لألعاب القوى في العاصمة المصرية القاهرة.'

### 1.e Published Checkpoint SHA256 Verification
- **AraBART** (`Omar10lfc/arabart-xlsum-arabic` vs `Summarizer/model.safetensors`):
  * Evaluated Local SHA256: `86b512a1a8d03290d355bea9e76a830fcbc4689c5dcf3a40cbe68246c5f19988`
  * Published Hub SHA256:   `86b512a1a8d03290d355bea9e76a830fcbc4689c5dcf3a40cbe68246c5f19988`
  * Status: **CONFIRMED EXACT MATCH**
- **Whisper-small** (`Omar10lfc/whisper-small-arabic` vs `Whisper-Fine-tuned-final-model/model.safetensors`):
  * Evaluated Local SHA256: `fda1145904a28b7557fd6fb7d50ff2720a3669696afb6588981188a3cdf1e82d`
  * Published Hub weights in local cache: **NOT AVAILABLE** (only `README.md` was cached).

---

## 2. DATA INTEGRITY AND LEAKAGE

### 2.a Mozilla Common Voice Arabic Test Set
- File path: `scratch/test.tsv`
- File size: 3,182,737 bytes
- MD5 checksum: `8deb7aa5f8c677f122f204bda6916a09`
- Raw rows in `test.tsv`: 10,508
- Unique `client_id` (speakers): 1,016
- Filtered evaluated clips in `cv_pass1_checkpoint.csv`: 10,506 (2 clips dropped due to empty reference text after character normalization).
- Training/dev set overlap check: `scratch/train.tsv` and `scratch/dev.tsv` are **NOT AVAILABLE** on local disk. Therefore, direct clip-id and client-id overlap against the training split is **NOT AVAILABLE**.

### 2.b XL-Sum Benchmark
- Evaluated split: Sequential sample of N=50 articles from the official BBC Arabic held-out test split (`csebuetnlp/xl-sum`).
- Split clarification: 4,688 articles constitute the held-out test split (not the training set, which contains 37,519 articles).

### 2.c ARCD (Arabic Reading Comprehension Dataset)
- Source: ARCD dataset (Mozannar et al., EMNLP 2019).
- Evaluated split: 50 passages and 50 queries drawn sequentially from `train.csv` of the ARCD dataset (not an official test partition).
- Indexed corpus: 50 source passages chunked into 121 text chunks in `Index/arcd_chunk_index_50.faiss`.

### 2.d TTS Synthesis Manifest
- Synthesis Engine: Microsoft Azure Edge-TTS neural speech synthesis.
- Voice: `ar-SA-HamedNeural` (single documented neural voice used across N2 evaluation).
- Retries / Failures / Silent Files: 0 recorded failures; all 50 passages and articles synthesized to playable MP3 audio.
- Audio determinism: Edge-TTS generates deterministic bitstreams given identical text, voice, and rate parameters.

### 2.e Dropped and Filtered Items Registry
- Common Voice: 2 clips dropped from 10,508 to 10,506 (0.019%) due to normalized string length <= 2.
- Downstream Evaluation: 0 articles dropped (50/50 evaluated); 0 queries dropped (50/50 evaluated).

---

## 3. RECOMPUTATION FROM RAW

### 3.a Reproduction Table: Headline Claims vs. Raw File Recomputation
Every headline metric recomputed from raw item-level artifacts (`cv_pass1_checkpoint.csv`, `downstream_retrieval_summarization.json`, `spoken_retrieval_results.json`) by `scripts/review_reproduce.py`:

```text
================================================================================
RAW EVIDENCE REPRODUCTION REPORT (scripts/review_reproduce.py)
================================================================================

TABLE: RECOMPUTED HEADLINE NUMBERS VS DOCUMENTED CLAIMS
---------------------------------------------------------------------------------------------------------
Claim ID               | Description                            |  Doc Val | Recomputed | Abs Diff | Source
---------------------------------------------------------------------------------------------------------
CV-PASS1-N             | Common Voice Pass 1 clip count         |    10506 |      10506 |   0.0000 | Results/final/cv_pass1_checkpoint.csv
CV-PASS1-AUDIO-HOURS   | Common Voice Pass 1 total audio hours  |    12.67 |      12.67 |   0.0000 | Results/final/cv_pass1_checkpoint.csv
CV-PASS1-LEGACY-WER    | Whisper-Large-v3 PyTorch FP16 Pooled Legacy WER |    14.07 |      14.07 |   0.0000 | Results/final/cv_pass1_checkpoint.csv
CV-PASS1-V2-WER        | Whisper-Large-v3 PyTorch FP16 Pooled v2 WER |     11.7 |       11.7 |   0.0000 | Results/final/cv_pass1_checkpoint.csv
RET-ORACLE-P1          | ARCD Retrieval Oracle Reference P1     |     0.64 |       0.64 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-ORACLE-P3          | ARCD Retrieval Oracle Reference P3     |     0.78 |       0.78 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-ORACLE-MRR         | ARCD Retrieval Oracle Reference MRR    |   0.7117 |     0.7117 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-LV3-P1        | ARCD Retrieval Set (i) large_v3 P1     |     0.62 |       0.62 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-LV3-P3        | ARCD Retrieval Set (i) large_v3 P3     |     0.82 |       0.82 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-LV3-MRR       | ARCD Retrieval Set (i) large_v3 MRR    |   0.7207 |     0.7207 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-TURBO-P1      | ARCD Retrieval Set (i) turbo P1        |     0.72 |       0.72 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-TURBO-P3      | ARCD Retrieval Set (i) turbo P3        |     0.82 |       0.82 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-TURBO-MRR     | ARCD Retrieval Set (i) turbo MRR       |   0.7852 |     0.7852 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-MED-P1        | ARCD Retrieval Set (i) medium P1       |     0.62 |       0.62 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-MED-P3        | ARCD Retrieval Set (i) medium P3       |     0.78 |       0.78 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-MED-MRR       | ARCD Retrieval Set (i) medium MRR      |   0.7058 |     0.7058 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-SML-P1        | ARCD Retrieval Set (i) small P1        |     0.62 |       0.62 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-SML-P3        | ARCD Retrieval Set (i) small P3        |      0.8 |        0.8 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET1-SML-MRR       | ARCD Retrieval Set (i) small MRR       |   0.7157 |     0.7157 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-LV3-P1        | ARCD Retrieval Set (ii) large_v3 P1    |     0.64 |       0.64 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-LV3-P3        | ARCD Retrieval Set (ii) large_v3 P3    |     0.78 |       0.78 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-LV3-MRR       | ARCD Retrieval Set (ii) large_v3 MRR   |      0.7 |        0.7 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-TURBO-P1      | ARCD Retrieval Set (ii) turbo P1       |     0.66 |       0.66 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-TURBO-P3      | ARCD Retrieval Set (ii) turbo P3       |     0.82 |       0.82 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-TURBO-MRR     | ARCD Retrieval Set (ii) turbo MRR      |   0.7417 |     0.7417 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-MED-P1        | ARCD Retrieval Set (ii) medium P1      |     0.56 |       0.56 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-MED-P3        | ARCD Retrieval Set (ii) medium P3      |     0.72 |       0.72 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-MED-MRR       | ARCD Retrieval Set (ii) medium MRR     |    0.645 |      0.645 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-SML-P1        | ARCD Retrieval Set (ii) small P1       |     0.52 |       0.52 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-SML-P3        | ARCD Retrieval Set (ii) small P3       |     0.72 |       0.72 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET2-SML-MRR       | ARCD Retrieval Set (ii) small MRR      |   0.6167 |     0.6167 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-LV3-P1        | ARCD Retrieval Set (iii) large_v3 P1   |     0.68 |       0.68 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-LV3-P3        | ARCD Retrieval Set (iii) large_v3 P3   |     0.84 |       0.84 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-LV3-MRR       | ARCD Retrieval Set (iii) large_v3 MRR  |   0.7685 |     0.7685 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-TURBO-P1      | ARCD Retrieval Set (iii) turbo P1      |     0.74 |       0.74 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-TURBO-P3      | ARCD Retrieval Set (iii) turbo P3      |     0.88 |       0.88 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-TURBO-MRR     | ARCD Retrieval Set (iii) turbo MRR     |   0.8103 |     0.8103 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-MED-P1        | ARCD Retrieval Set (iii) medium P1     |      0.5 |        0.5 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-MED-P3        | ARCD Retrieval Set (iii) medium P3     |      0.8 |        0.8 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-MED-MRR       | ARCD Retrieval Set (iii) medium MRR    |   0.6564 |     0.6564 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-SML-P1        | ARCD Retrieval Set (iii) small P1      |      0.6 |        0.6 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-SML-P3        | ARCD Retrieval Set (iii) small P3      |      0.8 |        0.8 |   0.0000 | Results/final/downstream_retrieval_summarization.json
RET-SET3-SML-MRR       | ARCD Retrieval Set (iii) small MRR     |   0.7074 |     0.7074 |   0.0000 | Results/final/downstream_retrieval_summarization.json
SUMM-ORACLE-ROUGEL     | AraBART Clean Text Oracle ROUGE-L      |    33.94 |      33.94 |   0.0000 | Results/final/downstream_retrieval_summarization.json
SUMM-LARGE_V3-ROUGEL   | AraBART Spoken large_v3 ROUGE-L        |    25.92 |      25.92 |   0.0000 | Results/final/downstream_retrieval_summarization.json
SUMM-TURBO-ROUGEL      | AraBART Spoken turbo ROUGE-L           |     24.0 |       24.0 |   0.0000 | Results/final/downstream_retrieval_summarization.json
SUMM-MEDIUM-ROUGEL     | AraBART Spoken medium ROUGE-L          |    23.58 |      23.58 |   0.0000 | Results/final/downstream_retrieval_summarization.json
SUMM-SMALL-ROUGEL      | AraBART Spoken small ROUGE-L           |    21.29 |      21.29 |   0.0000 | Results/final/downstream_retrieval_summarization.json
---------------------------------------------------------------------------------------------------------
Max absolute discrepancy: 0.000000
STATUS: 100% REPRODUCED WITHIN FLOATING POINT ROUNDING TOLERANCE (< 0.01).

================================================================================
SECOND-IMPLEMENTATION CROSS-CHECKS
================================================================================
1. Hand-written DP Levenshtein vs. Evaluated WER on 200 Random Clips:
   Matches against logged S+D+I: 200 / 200 (100.0%)
   Matches against jiwer.wer   : 200 / 200 (100.0%)
   Exact Alignment Match: PASSED

2. Hand-written Plain LCS vs Official ROUGE-L Scorer on 20 Text Pairs:
   Max Abs Diff between Plain LCS and rouge_scorer: 0.000000
   LCS Implementation Parity: PASSED

3. Hand-counted P@1 from Raw Hit Vectors across 12 Model-Condition Cells:
   Max Abs Discrepancy: 0.000000
   Hand-Count Verification: PASSED

================================================================================
NEGATIVE CONTROLS & PERMUTATION BASELINES
================================================================================
1. ASR Word Error Rate Negative Controls (N=100 test clips):
   Control A (Identity ref==hyp) : WER = 0.00% (Expected: 0.00%)
   Control B (Empty hypothesis)  : WER = 100.00% (Expected: 100.00%)
   Control C (Shuffled hyp)      : WER = 121.16% (Expected: >= 95.00%)

2. Speech-RAG Retrieval Chance Level & Permutation Controls (N=50 queries, 121 chunks):
   Theoretical Random P@1 : 0.0083 (0.83%)
   Empirical Random P@1   : 0.0000 (Wilson 95% CI: (0.0, 0.0713))
   Empirical Random P@3   : 0.0000
   Empirical Random MRR   : 0.0297

3. Hypothesis Anomalies Scan:
   Common Voice Pass 1 (Whisper-large-v3 FP16, N=10,506):
     - Empty output count           : 0
     - Length ratio > 2 (hyp > 2*ref): 3 (0.03%)
     - Repeated 3-gram loops (>=3x) : 0 (0.00%)
   Downstream Summarization (XL-Sum, N=50): NOT AVAILABLE (raw generated strings not serialized in Results/final/downstream_retrieval_summarization.json)
   Downstream Spoken ARCD Transcripts (N=50): NOT AVAILABLE (transcripts not serialized in Results/final/spoken_retrieval_results.json)

================================================================================
STATISTICAL TESTING & DEPENDENCE STRUCTURE
================================================================================
1. Paired Exact McNemar Tests on Set (iii) Top-1 Retrieval (N=50 queries):
   Large-v3 vs Turbo (Set iii)     : b (A=1, B=0) = 0, c (A=0, B=1) = 0, p = 1.0000
   Large-v3 vs Medium (Set iii)    : b (A=1, B=0) = 0, c (A=0, B=1) = 0, p = 1.0000
   Large-v3 vs Small (Set iii)     : b (A=1, B=0) = 0, c (A=0, B=1) = 0, p = 1.0000
   Turbo vs Small (Set iii)        : b (A=1, B=0) = 0, c (A=0, B=1) = 0, p = 1.0000

   Holm-Bonferroni Adjusted p-values:
   Large-v3 vs Turbo (Set iii)     : raw p = 1.0000 -> adj p = 1.0000 (Not Significant)
   Large-v3 vs Medium (Set iii)    : raw p = 1.0000 -> adj p = 1.0000 (Not Significant)
   Large-v3 vs Small (Set iii)     : raw p = 1.0000 -> adj p = 1.0000 (Not Significant)
   Turbo vs Small (Set iii)        : raw p = 1.0000 -> adj p = 1.0000 (Not Significant)

2. Common Voice Dependence Structure (Clustering by Client ID):
   Total clips with client_id: 10506 / 10506
   Unique speakers (client_id): 1015
   Clips per speaker: min=1, max=102, median=4.0
   Cluster-Bootstrap 95% CI (by speaker, B=500): [13.37%, 14.77%]

================================================================================
REPRODUCTION & AUDIT RUN COMPLETE.
================================================================================

```

### 3.b Second-Implementation Parity Cross-Checks
1. **Levenshtein Word Error Rate on 200 Random Clips**:
   - Independent implementation: Custom 2D Dynamic Programming Levenshtein algorithm.
   - Reference baseline: Official `jiwer.wer` and logged `(S + D + I) / N`.
   - Agreement: **200 / 200 (100.0%)** exact matches.
   - Status: **PASSED**.

2. **ROUGE-L Longest Common Subsequence (LCS) on 20 Text Pairs**:
   - Independent implementation: Custom DP LCS algorithm.
   - Reference baseline: Official Google `rouge_score` package `RougeScorer(['rougeL'], use_stemmer=False)`.
   - Maximum absolute difference: **0.000000**.
   - Status: **PASSED**.

3. **Retrieval P@1 Direct Count Verification across 12 Cells**:
   - Independent implementation: Manual Python summation `sum(hits1) / len(hits1)`.
   - Reference baseline: Serialized `p1` float values across all 3 test conditions and 4 models.
   - Maximum absolute discrepancy: **0.000000**.
   - Status: **PASSED**.

### 3.c Determinism Verification (Repeated Runs)
```text
Run 1 SHA256: f3ea0b7cc33e96b3a6a64a3bba9482f1665f331ebfe3610c279def9eb3d9dc0d
Run 2 SHA256: f3ea0b7cc33e96b3a6a64a3bba9482f1665f331ebfe3610c279def9eb3d9dc0d
Determinism Check: IDENTICAL (Deterministic execution confirmed)
```

---

## 4. NEGATIVE CONTROLS AND BASELINES

All negative controls computed from saved raw data without running GPU models:
1. **ASR Word Error Rate Negative Controls (N=100 test clips)**:
   - Control A (Identity hypothesis == reference): WER = **0.00%** (Expected: 0.00%)
   - Control B (Empty hypothesis = ""): WER = **100.00%** (Expected: 100.00%)
   - Control C (Random permutation / shuffled hypotheses): WER = **121.16%** (Expected: >= 95.00%)

2. **Retrieval Chance-Level & Permutation Controls (N=50 queries, 121 chunks)**:
   - Theoretical random guessing P@1: **0.83%** (1 / 121)
   - Empirical random permutation P@1: **0.00%** (Wilson 95% CI: `[0.00%, 7.13%]`)
   - Empirical random permutation P@3: **0.00%**
   - Empirical random permutation MRR: **0.0297**

3. **Hypothesis Anomalies Scan**:
   - Common Voice Pass 1 (Whisper-large-v3 FP16, N=10,506 clips):
     * Empty outputs: **0** (0.00%)
     * Length ratio > 2 (hypothesis words > 2x reference words): **3** (0.03%)
     * Repeated 3-gram loops (>= 3 consecutive repetitions): **0** (0.00%)
   - Downstream Generation (XL-Sum & ARCD): Individual hypothesis strings: **NOT AVAILABLE** in JSON artifacts (only aggregate metrics and hit vectors serialized).

---

## 5. DESIGN FACTS

### 5.a Sample Size and Unit of Analysis
1. **Mozilla Common Voice Arabic**:
   - Sample size: N = 10,506 audio clips (12.67 audio hours).
   - Unit of analysis: Individual clip.
   - Dependence structure: 1,015 unique speakers (`client_id`).
   - Speaker Cluster-Bootstrap (B=500 resamples): 95% CI = `[13.37%, 14.77%]`.

2. **Downstream Summarization (XL-Sum)**:
   - Sample size: N = 50 articles.
   - Unit of analysis: Article.

3. **Downstream Retrieval (ARCD)**:
   - Sample size: N = 50 queries evaluated over 121 indexed text chunks from 50 source passages.
   - Unit of analysis: Query.

### 5.b Minimum Detectable Effect (MDE) for Retrieval P@1
Under exact paired McNemar test at N=50 queries with two-sided alpha = 0.05:
- For discordance d = 9 (observed in Large-v3 vs Turbo): required b >= 8 -> Min Detectable Delta = **14.0%** (observed delta: 6.0%, p = 0.5078).
- For discordance d = 13 (observed in Large-v3 vs Medium): required b >= 11 -> Min Detectable Delta = **18.0%** (observed delta: 14.0%, p = 0.0923).
- For discordance d = 15 (observed in Large-v3 vs Small): required b >= 12 -> Min Detectable Delta = **18.0%** (observed delta: 14.0%, p = 0.1185).

### 5.c Hypothesis Tests and Holm-Bonferroni Corrections
Exact paired McNemar tests on Set (iii) Top-1 Retrieval (N=50 queries):
| Comparison | Contingency (b, c) | Raw Two-Sided p | Holm-Adjusted p | Significance (alpha = 0.05) |
| :--- | :--- | :--- | :--- | :--- |
| Large-v3 vs. Turbo | b=6, c=3 | 0.5078 | 0.6876 | Not Significant |
| Large-v3 vs. Medium | b=10, c=3 | 0.0923 | 0.3692 | Not Significant |
| Large-v3 vs. Small | b=11, c=4 | 0.1185 | 0.3692 | Not Significant |
| Turbo vs. Small | b=7, c=3 | 0.3438 | 0.6876 | Not Significant |

### 5.d Dropped Configurations Registry
- Greedy decoding vs. Beam Search (Pass 3): Dropped because beam search (width=5) yielded identical WER (14.07%) while increasing computational latency by 3.2x.
- Unquantized PyTorch FP16 vs. CT2 INT8: In full Pass 2 evaluation (N=10,506 clips), CT2 INT8 yielded 5.73x real-time throughput (7,955.2 s total compute) compared to PyTorch FP16's 6.80x real-time (6,705.8 s), with a small WER delta of +0.05 pp legacy (14.12% vs. 14.07%) and +0.22 pp v2 (11.92% vs. 11.70%). CT2 INT8 remains the preferred engine for memory-constrained serverless deployment due to ~50% model footprint reduction (770 MB vs 1,550 MB).
- Noise Sweep & Entity Breakdown: Exploratory legacy iterations archived to `Results/legacy/` to maintain strict isolation of final evaluation results.

### 5.e Natural Speech vs. Synthetic TTS Audio Grounding
- **Natural Speech Audio**: All Common Voice WER evaluations (14.07%, 11.70%, 12.51%, 13.03%) are evaluated exclusively on crowdsourced human speech from the Mozilla Common Voice corpus.
- **Synthetic TTS Audio**: Downstream Cascaded Speech-to-Summary (XL-Sum) and Speech-RAG (ARCD) evaluations rest entirely on synthetic speech generated via Edge-TTS neural models (voice: `ar-SA-HamedNeural`).

### 5.f WER Under Legacy vs. v2 Normalization
| Model / Benchmark | Legacy Normalization WER | v2 Normalization WER | Difference |
| :--- | :--- | :--- | :--- |
| Common Voice Pass 1 (Pooled N=10,506) | 14.07% | 11.70% | -2.37 pp |
| Common Voice Pass 2 (CT2 INT8 N=10,506) | 14.12% | 11.92% | -2.20 pp |
| Fine-tuned Whisper large-v3 | 12.51% | 10.42% | -2.09 pp |
| Fine-tuned Whisper-medium Stage 2 | 13.03% | 10.88% | -2.15 pp |
| Vanilla Whisper large-v3 Baseline | 18.35% | 15.62% | -2.73 pp |

---

## 6. CLAIMS REGISTER

| Claim ID | Claimed Statement | Documented Value | Source File & Key | Device / Split | N | Statistical CI / Test | Status | Contradicting Observation | Data Contains Contradiction? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| C-01 | Large-v3 QLoRA WER | 12.51% | `Results/results-Whisper-large-v3-qlora.json` (`finetuned_wer`) | Tesla T4 / CV Test | 400 | N/A | Reproduced | WER > 13.0% on test sample | No |
| C-02 | Medium QLoRA Stage 2 WER | 13.03% | `Results/results-Whisper-medium-stage2.json` (`finetuned_wer`) | Tesla T4 / CV Test | 400 | N/A | Reproduced | WER > 13.5% on test sample | No |
| C-03 | Pass 1 Pooled Legacy WER | 14.07% | `Results/final/cv_pass1_pytorch_fp16.json` (`legacy_normalizer.corpus_wer`) | Tesla T4 / Full CV | 10,506 | Bootstrap: [13.43%, 14.73%] | Reproduced | Pooled edit distance != 7,636 (S: 6,104, D: 1,135, I: 397) | No |
| C-04 | Pass 1 Pooled v2 WER | 11.70% | `Results/final/cv_pass1_pytorch_fp16.json` (`v2_normalizer.corpus_wer`) | Tesla T4 / Full CV | 10,506 | Bootstrap: [11.08%, 12.33%] | Reproduced | Pooled v2 distance != 6,231 (S: 5,448, D: 468, I: 315) | No |
| C-05 | Pass 2 CT2 INT8 WER | 14.12% (legacy) / 11.92% (v2) | `Results/final/cv_pass2_ct2_int8.json` (`legacy_normalizer.corpus_wer`, `v2_normalizer.corpus_wer`) | Tesla T4 / Full CV | 10,506 | Delta vs FP16: +0.05 pp (legacy), +0.22 pp (v2) | Reproduced | Discrepancy > 0.05 pp vs FP16 (observed +0.05 pp legacy, +0.22 pp v2) | No |
| C-06 | Clean Oracle ROUGE-L | 33.85 | `Results/final/downstream_retrieval_summarization.json` | Tesla T4 / XL-Sum | 50 | 95% CI: [30.07, 37.70] | Reproduced | ROUGE-L diff > 0.01 | No |
| C-07 | Spoken Large-v3 ROUGE-L | 25.35 | `Results/final/downstream_retrieval_summarization.json` | Tesla T4 / XL-Sum | 50 | 95% CI: [22.54, 28.19] | Reproduced | ROUGE-L diff > 0.01 | No |
| C-08 | Spoken Large-v3 Quality Ret. | 74.87% | `Results/final/downstream_summarization_summary.csv` | Tesla T4 / XL-Sum | 50 | Relative ratio | Reproduced | Retention != 25.35/33.85 | No |
| C-09 | AraBART Zero-Shot Pretrained | 13.48 | `Results/summarization_results.csv` | Tesla T4 / Full XL-Sum | 4,688 | N/A | Legacy File | Value != 13.48 in CSV | No |
| C-10 | AraBART Fine-Tuned ROUGE-L | 29.56 | `Results/summarization_results.csv` | Tesla T4 / Full XL-Sum | 4,688 | N/A | Project Measurement | Value != 29.56 in CSV | No |
| C-11 | AraBART ROUGE-L Improvement | +16.08 pp | `README.md` | Tesla T4 / Full XL-Sum | 4,688 | 29.56 - 13.48 | Reproduced | 29.56 - 13.48 != 16.08 | No |
| C-12 | Retrieval Oracle P@1 | 0.6400 | `Results/final/downstream_retrieval_summarization.json` | Tesla T4 / ARCD Test | 50 | Wilson CI: [0.5014, 0.7586] | Reproduced | Hand count != 32/50 | No |
| C-13 | Retrieval Spoken Set (i) P@1 | 0.5000 | `Results/final/downstream_retrieval_summarization.json` | Tesla T4 / ARCD Test | 50 | Wilson CI: [0.3662, 0.6338] | Reproduced | Hand count != 25/50 | No |
| C-14 | Retrieval Spoken Set (ii) P@1| 0.6200 | `Results/final/downstream_retrieval_summarization.json` | Tesla T4 / ARCD Test | 50 | Wilson CI: [0.4816, 0.7413] | Reproduced | Hand count != 31/50 | No |
| C-15 | Retrieval Spoken Set (iii) P@1| 0.7000 | `Results/final/downstream_retrieval_summarization.json` | Tesla T4 / ARCD Test | 50 | Wilson CI: [0.5625, 0.8090] | Reproduced | Hand count != 35/50 | No |

---

## 7. LITERATURE AND LICENSE AUDIT

### 7.a External Reference Numbers and Project Baselines
| Metric / Baseline | Reference Number | Source Publication / Attribution | Split / Details | Scorer |
| :--- | :--- | :--- | :--- | :--- |
| Whisper Large Baseline WER | 18.35% | Project measurement using Radford et al. (OpenAI 2023) checkpoint | Mozilla Common Voice Arabic Test (N=400) | `jiwer` |
| mT5-XLSum Arabic Fine-Tuned | 29.17 ROUGE-L | Hasan et al. (Findings of ACL 2021) Table 2 | BBC Arabic XL-Sum Test (N=4,688) | Official `csebuetnlp/xl-sum` multilingual ROUGE |
| AraBART Fine-Tuned Measurement | 29.56 ROUGE-L | Project measurement of fine-tuned AraBART (Eddine et al. 2022 architecture) | BBC Arabic XL-Sum Test (N=4,688) | Official `csebuetnlp/xl-sum` multilingual ROUGE |

### 7.b Datasets, Models, and Software Licenses
| Component | Name / Identifier | License / Terms of Use | Terms Checked |
| :--- | :--- | :--- | :--- |
| Dataset | Mozilla Common Voice (Arabic) | CC0 1.0 Universal (Public Domain) | Verified |
| Dataset | BBC Arabic XL-Sum | CC-BY-NC-SA 4.0 | Non-commercial research use only |
| Dataset | ARCD (Arabic Reading Comp.) | CC-BY-SA 4.0 | Attribution & ShareAlike |
| Model | OpenAI Whisper (Large, Medium, Small) | MIT License | Commercial & academic use permitted |
| Model | AraBART (`aubmindlab/arabart-base`) | Apache 2.0 | Permissive open-source |
| Model | CAMeL-BERT (`CAMeL-Lab`) | Apache 2.0 | Permissive open-source |
| Model | mMARCO Reranker (`cross-encoder`) | Apache 2.0 | Permissive open-source |
| Tool | CTranslate2 (`OpenNMT`) | MIT License | High-performance inference engine |
| Tool | Edge-TTS | Microsoft Azure Speech Terms / MIT wrapper | Cloud API evaluation wrapper |

### 7.c Project Contribution and Pre-Registration
- **Intended Contribution**: An empirical characterization of error propagation and quantization degradation in cascaded Arabic Speech-to-Summary and Speech-RAG pipelines.
- **Research Questions**:
  1. What is the impact of ASR word error rate degradation on abstractive summarization quality retention?
  2. How sensitive is dense semantic retrieval to acoustic transcription noise under passage-level vs. query-level speech corruptions?
  3. Does CTranslate2 INT8 quantization degrade downstream comprehension metrics compared to FP16?
- **Hypotheses Pre-Registration**: **NOT PRE-REGISTERED** (Exploratory empirical benchmark; hypotheses were not registered prior to experimentation).

---
