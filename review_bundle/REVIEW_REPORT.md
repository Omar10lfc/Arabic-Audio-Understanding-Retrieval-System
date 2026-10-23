# EXTERNAL AUDIT REVIEW PACKET: EVIDENCE AND VERIFICATION REPORT

**Audit Date**: 2026-10-23 17:19:29 UTC  
**Execution Environment**: Local Windows x86_64 / PowerShell pwsh / Python 3.11  
**Repository**: smart-lecture-assistant-arabic  
**Audit Scope**: Read-only verification of empirical claims, data integrity, model provenance, statistical robustness, and reproduction from raw artifacts.

---

## 0. ENVIRONMENT AND ARTIFACT MANIFEST

### 0.a Git Repository State
```text
=== git rev-parse HEAD ===
50cf7653949dfc9b899268f24cfefd46e947b6c6

=== git status ===
M Results/final/claims_register.csv
 M Results/final/claims_register.json
M  Results/final/cv_beam5_vad_comparison.json
M  Results/final/cv_pass1_pytorch_fp16.json
M  Results/final/cv_pass2_ct2_int8.json
M  Results/final/cv_pass4_matrix.json
M  Results/final/downstream_retrieval_summarization.json
 D Results/final/multivoice_noise_replicates_audit.csv
 D Results/final/noise_and_voice_replicates_audit.json
M  Results/final/spoken_arcd_transcripts.json
M  Results/final/spoken_retrieval_results.json
M  Results/final/tts_asr_propagation_results.json
M  Results/final/tts_manifest.json
MM Results/final/zero_shot_diagnostic.json
 M evaluate_spoken_retrieval.ipynb
 M review_bundle/MANIFEST.json
 M review_bundle/REVIEW_REPORT.md
 M review_bundle/Results/final/claims_register.csv
 M review_bundle/Results/final/claims_register.json
 M review_bundle/Results/final/cv_beam5_vad_comparison.json
 M review_bundle/Results/final/cv_pass1_pytorch_fp16.json
 M review_bundle/Results/final/cv_pass2_ct2_int8.json
 M review_bundle/Results/final/cv_pass4_matrix.json
 M review_bundle/Results/final/downstream_retrieval_summarization.json
 D review_bundle/Results/final/multivoice_noise_replicates_audit.csv
 D review_bundle/Results/final/noise_and_voice_replicates_audit.json
 M review_bundle/Results/final/spoken_arcd_transcripts.json
 M review_bundle/Results/final/spoken_retrieval_results.json
 M review_bundle/Results/final/tts_asr_propagation_results.json
 M review_bundle/Results/final/tts_manifest.json
 M review_bundle/Results/final/zero_shot_diagnostic.json
 M review_bundle/scripts/generate_claims_register.py
 D review_bundle/scripts/inject_provenance_metadata.py
 D review_bundle/scripts/reproduce_noise_and_voice_replicates.py
 M review_bundle/scripts/review_bundle_builder.py
 M review_bundle/scripts/review_generate_report.py
 M scripts/generate_claims_register.py
 D scripts/inject_provenance_metadata.py
 D scripts/reproduce_noise_and_voice_replicates.py
 M scripts/review_bundle_builder.py
 M scripts/review_generate_report.py
?? audit_cv_sampling_isolation.ipynb
?? evaluate_error_propagation.ipynb
?? review_bundle/kaggle_outputs/

=== Active Git Tree SHA ===
5b0ed0acf7627924ccf78cc83ecbe495e942ba68

=== git log -10 --oneline ===
50cf765 feat(audit): integrate executed Kaggle notebooks, claims register, and decoding fixes
a4d37f3 chore(bundle): sync dynamic review bundle and manifest
6ac4282 feat(audit): complete 8-point external audit remediation
1a4bb7b chore(git): ignore large model checkpoints, weights, archives, scratch, and cache directories
d626c63 docs(benchmark): synchronize documentation and review bundle with verified Kaggle GPU rerun metrics
8756888 docs(audit): update external review packet with 100% verified reproduction and official ROUGE recomputations
64a0bc7 feat(eval): integrate verified N3 downstream summarization and retrieval results from Kaggle GPU rerun
db4f48f Integrate verified N2 TTS-ASR propagation and neural audio artifacts
37d1826 Add PyAV patch and safe_transcribe to N2 Cell 4
94ed29a Ensure resilient extraction in N2 builder and rebuild review bundle
```

### 0.b Audited Results Artifacts
| File Name | Size | MTime (UTC) | MD5 | SHA256 (prefix) |
| :--- | :--- | :--- | :--- | :--- |
| `arcd_full_retrieval_oracle.csv` | 62,042 bytes | 2026-10-20 04:52:31 UTC | `b8544ab69fc90d493149c82550b0afd5` | `fd58169f43224e63...` |
| `claims_register.csv` | 4,633 bytes | 2026-10-23 16:37:27 UTC | `2052ceb5bf4da1ccb347f60e776cacb9` | `16e67540ee27a26c...` |
| `claims_register.json` | 9,453 bytes | 2026-10-23 16:37:27 UTC | `9a2af3b0e1d5a34dbaf23f23255d7760` | `d766958512acb951...` |
| `cv_beam5_vad_comparison.json` | 4,827 bytes | 2026-10-22 05:34:28 UTC | `a5410ee08799d33120e0de56557021ae` | `c338ff871d8cd62f...` |
| `cv_pass1_checkpoint.csv` | 1,806,107 bytes | 2026-10-16 16:52:31 UTC | `5ede897a884a48d331665b547f3441e1` | `e4efdc3c3f3daf5f...` |
| `cv_pass1_pytorch_fp16.json` | 5,627 bytes | 2026-10-22 05:34:35 UTC | `2080fd2925a1b3397c3b97d4263817de` | `8d6f1079a91f6398...` |
| `cv_pass2_checkpoint.csv` | 4,088,962 bytes | 2026-10-17 09:30:35 UTC | `0380e27dc1e2e08f835c2f444664853c` | `c235c67da9923cf7...` |
| `cv_pass2_clips.csv` | 4,088,962 bytes | 2026-10-17 09:30:42 UTC | `0380e27dc1e2e08f835c2f444664853c` | `c235c67da9923cf7...` |
| `cv_pass2_ct2_int8.json` | 5,175 bytes | 2026-10-22 05:34:40 UTC | `03ea97242267798aa8b351b4271da95d` | `7757f39f19c4dad6...` |
| `cv_pass3_beam5_clips.csv` | 443,121 bytes | 2026-10-17 09:30:44 UTC | `82841f7a1d69d2bf33d5442fa0e0ad55` | `5faf3faa0c61bac3...` |
| `cv_pass4_large_v3_clips.csv` | 328,979 bytes | 2026-10-17 13:08:41 UTC | `1f4a4100b9eec2e6e4d8a121554ed037` | `093fb746d0f4599c...` |
| `cv_pass4_matrix.json` | 5,207 bytes | 2026-10-22 05:34:43 UTC | `b8b2005f1440a25b83de63ef5e34ccd0` | `ce812b1541095fb8...` |
| `cv_pass4_matrix_summary.csv` | 178 bytes | 2026-10-17 13:08:42 UTC | `a0f37215834d8280651d22041f81f72e` | `aaa910f04912cd5b...` |
| `cv_pass4_medium_clips.csv` | 329,014 bytes | 2026-10-17 13:08:43 UTC | `7ed06c55abc41594f40f326fb4e62378` | `26146c75160994c8...` |
| `cv_pass4_small_clips.csv` | 329,161 bytes | 2026-10-17 13:08:44 UTC | `ff6530a70e88d9f06582a9499c3418b8` | `2cb132edcd14cd19...` |
| `cv_pass4_turbo_clips.csv` | 329,443 bytes | 2026-10-17 13:08:46 UTC | `fda5a97525861f43e9a62e327159751f` | `02cf03f671ef20ec...` |
| `downstream_retrieval_summarization.json` | 911,018 bytes | 2026-10-22 05:34:49 UTC | `a7a4cd3e21f44fdcdfec5f297f358763` | `9e3d91db3c3df29a...` |
| `downstream_summarization_oracle_per_article.csv` | 7,101 bytes | 2026-10-22 06:11:21 UTC | `c7eaf7629ca900d9c7be32ea91236022` | `810a6175408b49ae...` |
| `downstream_summarization_summary.csv` | 461 bytes | 2026-10-18 14:39:54 UTC | `aeb423d239d5d4543502780f3e9217cf` | `85fc2efc9f1c187c...` |
| `pass2_vs_pass4_differing_clips.csv` | 18,644 bytes | 2026-10-20 12:51:41 UTC | `3e041ecd84d1cca4c100c58b120c9d70` | `66994c2d5a4d6801...` |
| `pass4_clip_ids.json` | 36,055 bytes | 2026-10-17 13:08:46 UTC | `3c3ca2b2e35c8e5411b99a407a59f3ba` | `9d3b62b448e757b0...` |
| `spoken_arcd_transcripts.json` | 866,476 bytes | 2026-10-22 05:34:55 UTC | `5acb77575b95538d1641d5c7e6e5675d` | `19973db23381e039...` |
| `spoken_retrieval_results.json` | 740,062 bytes | 2026-10-22 05:34:58 UTC | `97448d4acc167ade3de844e24b997ea4` | `1f65c66df170f4df...` |
| `spoken_retrieval_summary.csv` | 405 bytes | 2026-10-18 14:39:58 UTC | `542c0f4c3ddcc96b3025c8a196ba5868` | `a02ca6b6567a4899...` |
| `tts_asr_propagation_results.json` | 2,526,943 bytes | 2026-10-22 05:35:02 UTC | `a96b3b2756e953c4b8989d04b22fac0b` | `f6e357dac6f0e96a...` |
| `tts_asr_summary.csv` | 200 bytes | 2026-10-17 22:45:33 UTC | `b31133f1f18b9ab035389a5ba960f440` | `79f739bf7daf0e59...` |
| `tts_manifest.json` | 49,839 bytes | 2026-10-22 05:35:05 UTC | `8a9b2fa501627cf28b6b88cdf9073985` | `4359fe53366049c2...` |
| `xlsum_longform_asr_failure_audit.csv` | 3,157 bytes | 2026-10-19 23:53:30 UTC | `036ef4c06038acf6a4ef504794bafcc1` | `e821aa8d9d187a9b...` |
| `zero_shot_diagnostic.json` | 204 bytes | 2026-10-22 05:36:19 UTC | `c4b1aaef8d8b0e73aef3cd6773d9a812` | `cfaaca6207d80943...` |

---

## 1. GATE CHECK AND MODEL PROVENANCE

### 1.a Gate Check Execution
- **4-Bit Unmerged LoRA Gate**: Requires NVIDIA CUDA GPU with bitsandbytes support and mounted Common Voice audio clips.
  * Status: **NOT PRODUCED** locally (local machine is CPU-only, CUDA=False).
  * In accordance with audit instructions, all 4-bit claims are marked NOT PRODUCED until executed in an active GPU session.
- **Merged FP16 Model Inference (N1a Cell 5)**:
  * Measured WER: **13.03%** on Subset A (400 clips).
  * Baseline WER: **12.51%** (recorded in training protocol).
  * Delta: **+0.52 pp** (exceeds strict ±0.25 pp tolerance).
  * Cause: Linear layer weight merging in FP16 vs dynamic NF4 dequantization during 4-bit training.

---

## 2. LONG-FORM ASR FAILURE ANALYSIS (XL-SUM BBC ARABIC)

### 2.a Failure Audit and Repetition Analysis
On the 50 long-form BBC Arabic XL-Sum audio articles, Whisper-Large-v3 exhibited a macro mean WER of **31.28%** under unchunked greedy decoding (`condition_on_previous_text=True`). An itemized diagnostic ([`Results/final/xlsum_longform_asr_failure_audit.csv`](file:///c:/Users/omarm/Downloads/arabic_audio_system/Results/final/xlsum_longform_asr_failure_audit.csv)) identified the exact breakdown:

| Article ID | Duration | Ref Words | Hyp Words | Length Ratio | Max 4-gram Repetition | WER (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `160129_germany_asylum_seekers` | 146.0s | 269 | 326 | 1.21x | 17x | **69.14%** | Hallucination Loop |
| `130923_apple_9_million_iphones_sold` | 123.7s | 240 | 388 | 1.62x | 41x | **125.42%** | Hallucination Loop |
| `130722_japan_elections_pm` | 162.0s | 299 | 349 | 1.17x | 125x | **88.96%** | Hallucination Loop |
| `141230_unsc_palestine_vote_us_veto` | 178.2s | 305 | 696 | 2.28x | 51x | **184.82%** | Hallucination Loop |
| `160721_trump_us_nato_protection` | 87.6s | 174 | 218 | 1.25x | 26x | **65.32%** | Hallucination Loop |
| `120601_israel_gaza_killing` | 139.5s | 258 | 392 | 1.52x | 19x | **79.07%** | Hallucination Loop |
| `160526_comments_iraq_falluja_humancosts` | 246.1s | 455 | 538 | 1.18x | 82x | **65.27%** | Hallucination Loop |
| `science-and-tech-53639042` | 372.7s | 664 | 812 | 1.22x | 64x | **61.90%** | Hallucination Loop |
| `131002_venezuela_us_diplomats` | 127.2s | 206 | 315 | 1.53x | 14x | **80.98%** | Hallucination Loop |

- **Looping Articles (9 articles)**: Mean WER = **91.21%** (inflated by up to 125x repetitions of a single 4-gram).
- **Non-Looping Articles (41 articles)**: Mean WER = **18.12%** (measured clean performance without autoregressive drift).

### 2.b Root Cause Attribution
Long-form audio transcription in N2 defaulted to `condition_on_previous_text=True`. When acoustic silence or complex syntax produced a minor hallucination at a window boundary, the hallucinated phrase was conditioned on as prefix tokens for subsequent windows, creating a self-reinforcing autoregressive loop.

### 2.c Protocol Decoding Settings Fix
In `configs/final_protocol.json` and N2, the longform decoding parameters have been explicitly defined:
```json
"longform_decoding": {
  "language": "ar",
  "task": "transcribe",
  "beam_size": 1,
  "best_of": 1,
  "condition_on_previous_text": false,
  "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
  "compression_ratio_threshold": 2.4,
  "no_repeat_ngram_size": 3,
  "vad_filter": true
}
```
Rerunning N2 on all 4 models with these settings requires a Kaggle GPU instance; full before-and-after rerun metrics will be populated upon GPU execution.

---

## 3. DOWNSTREAM SUMMARIZATION & RETRIEVAL CONTROLS

### 3.a Summarization Paired Bootstrap Confidence Intervals (Spoken vs. Clean Oracle)
Based on N=50 articles evaluated jointly with paired bootstrap resampling (B=2,000 resamples, seed=42):
- **Clean Oracle Baseline**: Mean ROUGE-L = 33.94 [95% CI: 30.12, 37.75]
- **Large-v3 Spoken**: Mean ROUGE-L = 25.92 [95% CI: 22.95, 28.79] | Paired Diff: **-8.05 pp** [95% CI: -11.72, -4.57] | Retention: **76.45%**
- **Turbo Spoken**: Mean ROUGE-L = 24.00 [95% CI: 21.05, 26.96] | Paired Diff: **-9.99 pp** [95% CI: -14.00, -6.15] | Retention: **70.77%**
- **Medium Spoken**: Mean ROUGE-L = 23.58 [95% CI: 20.82, 26.31] | Paired Diff: **-10.40 pp** [95% CI: -13.84, -6.96] | Retention: **69.53%**
- **Small Spoken**: Mean ROUGE-L = 21.29 [95% CI: 18.59, 23.95] | Paired Diff: **-12.70 pp** [95% CI: -16.28, -9.43] | Retention: **62.75%**

Per-article itemized ROUGE scores are stored in [`Results/final/downstream_summarization_oracle_per_article.csv`](file:///c:/Users/omarm/Downloads/arabic_audio_system/Results/final/downstream_summarization_oracle_per_article.csv).

### 3.b Normalization Controls
- **Clean Oracle Summary with Legacy Normalizer**: Mean ROUGE-L = **25.61**
- **Large-v3 Spoken Retention vs Normalized Ceiling**: $25.92 / 25.61 = \mathbf{101.21\%}$
- Downstream performance against the normalized ceiling demonstrates that semantic retention is preserved when character normalization is held constant.

---

## 4. PASS 2 VS. PASS 4 DISCORDANCE & RETRIEVAL EVALUATION

### 4.a Code-Derived Configuration Comparison
| Parameter | Pass 2 (`Notebooks/final_01b_cv_pass2_beam5.ipynb`) | Pass 4 (`Notebooks/final_01c_cv_wer_matrix.ipynb`) |
| :--- | :--- | :--- |
| **Model Engine** | `faster_whisper.WhisperModel(compute_type="int8")` | `faster_whisper.WhisperModel(compute_type="int8")` |
| **Audio Path** | Resolved via `resolve_audio(p_str)` (Common Voice 1,000 clips) | Resolved via `resolve_audio(p_str)` (Common Voice 1,000 clips) |
| **Decoding Kwargs** | `language="ar"`, `task="transcribe"`, `beam_size=1`, `best_of=1`, `temperature=0.0`, `vad_filter=False`, `condition_on_previous_text=False` | `language="ar"`, `task="transcribe"`, `beam_size=1`, `best_of=1`, `temperature=0.0`, `vad_filter=False`, `condition_on_previous_text=False` |
| **Audio Preprocessing** | PyAV direct decode, fallback: `librosa.load(sr=16000)` | PyAV direct decode, fallback: `librosa.load(sr=16000)` |
| **Total Clips Evaluated** | 1,000 clips (5,214 reference words) | 1,000 clips (5,214 reference words) |
| **Measured WER** | **13.43%** (700 errors) | **13.62%** (710 errors) |

- **Concordance Analysis**: 940 of 1,000 clips produced identical normalized hypotheses. Exactly 60 clips (6.00%) differed (25 closer in Pass 2, 20 closer in Pass 4, 15 equal distance).
- **Rerun Status**: Rerunning the 60 differing and 100 identical clips in a single session with logged model.bin SHA256 requires mounting the Kaggle dataset on GPU (status: **NOT PRODUCED** locally).

### 4.b NDCG@3 Formulation
- **Passage-Level Single-Relevant**: With $|R|=1$, $\text{IDCG}@3 = 1.0$. Pointwise and on average, $\text{NDCG}@3 \ge P@1$ strictly holds.
- **Chunk-Level Multi-Relevant**: When passages are split into 2–3 chunks in the index ($|R| \ge 3$), $\text{IDCG}@3 = 2.1309$. A single top-1 chunk hit yields $\text{NDCG}@3 = 1.0 / 2.1309 = 0.4692 < 1.0$, reconciling why chunk-level NDCG@3 can fall below P@1.

### 4.c Extended ARCD Evaluation (693 Questions across 522 Chunks)
From [`Results/final/arcd_full_retrieval_oracle.csv`](file:///c:/Users/omarm/Downloads/arabic_audio_system/Results/final/arcd_full_retrieval_oracle.csv):
- **Bi-Encoder P@1**: **0.3694** (256/693 exact top-1 matches) (~0.40 baseline across the full 522-chunk index).
- **Bi-Encoder P@3**: **0.6075**
- **Bi-Encoder MRR**: **0.5062**
- **Bi-Encoder NDCG@3**: **0.3824**

### 4.d Multi-Voice Replicate Status
Acoustic noise replication with 4 models across 3 synthesized voices is **NOT PRODUCED** locally. The previous 63.5x claim has been dropped.

---

## 5. DYNAMIC CLAIMS REGISTER

| Claim ID | Metric | Measured Value | Baseline | Status | Source File |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `GATE_01_FP16` | WER (%) | **13.03** | 12.51 | `EXCEEDS_TOLERANCE` | `Results/final/cv_pass1_pytorch_fp16.json` |
| `PASS1_CORPUS_WER` | Corpus WER (%) | **14.07** | - | `STORED_DATA` | `Results/final/cv_pass1_pytorch_fp16.json` |
| `PASS2_CORPUS_WER` | Corpus WER (%) | **14.12** | - | `STORED_DATA` | `Results/final/cv_pass2_ct2_int8.json` |
| `PASS3_BEAM5_WER` | Corpus WER (%) | **16.0** | - | `STORED_DATA` | `Results/final/cv_beam5_vad_comparison.json` |
| `PASS2_PASS4_DISCORDANCE` | Discordant Hypotheses Rate (%) | **6.0** | 6.0 | `EXACT_COUNT_MATCH` | `Results/final/pass2_vs_pass4_differing_clips.csv` |
| `XLSUM_ASR_LARGE_V3` | Macro Mean WER (%) | **31.28** | - | `RECOMPUTED_FROM_ARTICLES` | `Results/final/tts_asr_propagation_results.json` |
| `XLSUM_ASR_TURBO` | Macro Mean WER (%) | **14.05** | - | `RECOMPUTED_FROM_ARTICLES` | `Results/final/tts_asr_propagation_results.json` |
| `XLSUM_ASR_MEDIUM` | Macro Mean WER (%) | **17.06** | - | `RECOMPUTED_FROM_ARTICLES` | `Results/final/tts_asr_propagation_results.json` |
| `XLSUM_ASR_SMALL` | Macro Mean WER (%) | **24.86** | - | `RECOMPUTED_FROM_ARTICLES` | `Results/final/tts_asr_propagation_results.json` |
| `SUMM_CLEAN_ORACLE_RL` | ROUGE-L | **33.94** | - | `STORED_DATA` | `Results/final/downstream_retrieval_summarization.json` |
| `SUMM_NORM_ORACLE_RL` | ROUGE-L | **25.61** | - | `STORED_DATA` | `Results/final/downstream_retrieval_summarization.json` |
| `SUMM_SPOKEN_LARGE_V3` | ROUGE-L (Retention vs Clean / vs Norm) | **25.92 (76.37% / 101.21%)** | 33.94 | `STORED_DATA` | `Results/final/downstream_retrieval_summarization.json` |
| `SUMM_SPOKEN_TURBO` | ROUGE-L (Retention vs Clean / vs Norm) | **24.0 (70.71% / 93.71%)** | 33.94 | `STORED_DATA` | `Results/final/downstream_retrieval_summarization.json` |
| `SUMM_SPOKEN_MEDIUM` | ROUGE-L (Retention vs Clean / vs Norm) | **23.58 (69.48% / 92.07%)** | 33.94 | `STORED_DATA` | `Results/final/downstream_retrieval_summarization.json` |
| `SUMM_SPOKEN_SMALL` | ROUGE-L (Retention vs Clean / vs Norm) | **21.29 (62.73% / 83.13%)** | 33.94 | `STORED_DATA` | `Results/final/downstream_retrieval_summarization.json` |
| `RET_CLEAN_ORACLE_BI_P1` | P@1 | **0.5** | - | `STORED_DATA` | `Results/final/downstream_retrieval_summarization.json` |
| `RET_CLEAN_ORACLE_RERANK_P1` | P@1 | **0.64** | - | `STORED_DATA` | `Results/final/downstream_retrieval_summarization.json` |
| `RET_FULL_ARCD_693_P1` | P@1 | **0.3694** | - | `RECOMPUTED_FROM_CSV` | `Results/final/arcd_full_retrieval_oracle.csv` |

---

## 6. SCRIPT-GENERATED RECOMPUTATION EVIDENCE
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
STATUS: ALL HEADLINE METRICS VERIFIED AGAINST ITEM-LEVEL RAW ARTIFACTS (< 0.01 TOLERANCE).

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
