import json
import os
import ast
import shutil

def create_error_propagation_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def add_code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")]
        })

    # 1. Header
    add_md("""# 🎙️ ➔ 📝 Quantization & Cascading Error Propagation Benchmark (Dual-Scale: N=50 & N=25)
### Evaluating ASR Quantization Tiers (FP16 vs. INT8_FP16 vs. Static INT8) on Downstream Arabic Summarization (Whisper-Medium ➔ AraBART)

**Key Research Questions:**
When deploying Arabic Spoken Document Processing on cost-constrained serverless infrastructure:
1. **Quantization Impact on ASR**: How much do precision tiers (**CT2 Float16**, **INT8_FLOAT16**, and **Static INT8**) alter ASR Word Error Rate (WER %)?
2. **Quantization Impact on Downstream Summarization**: Does ASR quantization noise cascade into AraBART ROUGE scores, and by what margin does downstream performance shift across tiers?
3. **End-to-End Trade-Off**: Quantifying the trade-off between model disk footprint reduction and downstream summarization quality retention.
4. **Scale Consistency**: Evaluating across both N=25 and N=50 held-out test articles to verify metric stability across sample counts.

**Benchmark Architecture:**
- **Oracle Pipeline:** Clean Human Article ➔ AraBART ➔ Reference ROUGE (100% Quality Baseline)
- **Pipeline 1 (CT2 Float16):** Audio ➔ Whisper-Medium CT2 FP16 ➔ AraBART ➔ ROUGE
- **Pipeline 2 (CT2 INT8_FLOAT16):** Audio ➔ Whisper-Medium CT2 INT8_FLOAT16 ➔ AraBART ➔ ROUGE
- **Pipeline 3 (CT2 Static INT8):** Audio ➔ Whisper-Medium CT2 Static INT8 ➔ AraBART ➔ ROUGE
- *(Baseline: Vanilla PyTorch HF FP16)*""")

    # 2. Dependencies
    add_code("""# ── 1. Install Dependencies ──────────────────────────────────────────────────
!pip uninstall -y torchaudio
!pip install -q faster-whisper ctranslate2 edge-tts evaluate sacrebleu tabulate pandas soundfile jiwer transformers scipy rouge-score

# Provide robust pyonmttok fallback only if not already installed (preserves tokenization for Arabic)
try:
    import pyonmttok
except ImportError:
    from unittest.mock import MagicMock
    import re
    mock_pyonmttok = MagicMock()
    class MockTokenizer:
        def __init__(self, mode='aggressive'): pass
        def tokenize(self, text):
            return [re.findall(r'\w+|[^\w\s]', text, re.UNICODE)]
    mock_pyonmttok.Tokenizer = MockTokenizer
    import sys
    sys.modules['pyonmttok'] = mock_pyonmttok

import subprocess, os
REPO_DIR = 'xl-sum'
if not os.path.isdir(REPO_DIR):
    subprocess.check_call(['git', 'clone', '--depth', '1', 'https://github.com/csebuetnlp/xl-sum.git', REPO_DIR])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-U', f'./{REPO_DIR}/multilingual_rouge_scoring'])
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
print("[OK] Official XL-Sum multilingual_rouge_scoring installed successfully.")""")

    # 3. Environment & Hardware Diagnostics
    add_code("""# ── 2. Environment Verification ───────────────────────────────────────────────
import os
import sys
import time
import re
import json
import glob
import asyncio
import tarfile
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import jiwer
import edge_tts
from tabulate import tabulate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"GPU Device Name : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Running on CPU.")""")

    # 4. Configuration & Auto-Detection of Models & Dataset
    add_code("""# ── 3. Configuration & Auto-Detection ─────────────────────────────────────────
CFG = {
    "num_samples": 50,               # Number of test articles to evaluate (enables full N=50 and N=25 subset analysis)
    "voice": "ar-SA-HamedNeural",     # High-quality neural Arabic voice
    "beam_size": 1,                  # Whisper greedy decoding (production speed)
    "max_input_length": 512,         # AraBART max input tokens
    "max_summary_length": 100,       # AraBART max output tokens
    "output_dir": "/kaggle/working/error_propagation_results",
    "models_dir": "/kaggle/working/models",
    "audio_dir": "/kaggle/working/error_propagation_results/audio",
}
os.makedirs(CFG["output_dir"], exist_ok=True)
os.makedirs(CFG["models_dir"], exist_ok=True)
os.makedirs(CFG["audio_dir"], exist_ok=True)

# 1. Recursive Auto-detection for Stage 2 Whisper-Medium Model
print("Scanning for Stage 2 Whisper model...")
model_ct2_path = None
model_hf_path = None

search_roots = ["/kaggle/input", "/kaggle/working", "./whisper_medium_output"]
for sroot in search_roots:
    if not os.path.exists(sroot):
        continue
    for root, dirs, files in os.walk(sroot):
        if "model.bin" in files and ("vocabulary.json" in files or "vocabulary.txt" in files):
            if not model_ct2_path or "stage2" in root.lower() or "ct2" in root.lower():
                model_ct2_path = root
                print(f"  [FOUND] CT2 model: {root}")
        if "config.json" in files and any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
            if "model.bin" not in files:
                if not model_hf_path or "stage2" in root.lower() or "merged" in root.lower():
                    model_hf_path = root
                    print(f"  [FOUND] HF model : {root}")

print(f"HF Model Path  : {model_hf_path}")
print(f"CT2 Model Path : {model_ct2_path}")

# 2. Locate Arabic XL-Sum Dataset
print("\\nScanning for Arabic XL-Sum dataset...")
dataset_paths = [
    "/kaggle/input/datasets/omar10lfc/arabic-xl-sum",
    "/kaggle/input/arabic-xl-sum",
    "/kaggle/input/datasets/omar10lfc/*xl-sum*",
    "/kaggle/input/*xl-sum*",
    "/kaggle/input/*arabic_XLSum*",
    "Data",
    "./Data",
]

xlsum_jsonl_file = None
xlsum_archive_file = None

for dpath in dataset_paths:
    for match in glob.glob(dpath):
        if os.path.exists(match):
            for root, dirs, files in os.walk(match):
                for f in files:
                    if f in ["arabic_test.jsonl", "test.jsonl"]:
                        xlsum_jsonl_file = os.path.join(root, f)
                        print(f"  [FOUND] Uncompressed test file: {xlsum_jsonl_file}")
                        break
                    elif f.endswith((".tar.bz2", ".tar.gz")):
                        if "xlsum" in f.lower() or "arabic" in f.lower():
                            xlsum_archive_file = os.path.join(root, f)
                            print(f"  [FOUND] Compressed archive: {xlsum_archive_file}")
                if xlsum_jsonl_file:
                    break
        if xlsum_jsonl_file:
            break

print(f"Test File : {xlsum_jsonl_file}")
print(f"Archive   : {xlsum_archive_file}")""")

    # 5. Load Dataset Articles
    add_code("""# ── 4. Load Held-Out XL-Sum Test Articles ────────────────────────────────────
test_samples = []

if xlsum_jsonl_file and os.path.exists(xlsum_jsonl_file):
    print(f"Loading test samples directly from uncompressed file: {xlsum_jsonl_file}...")
    with open(xlsum_jsonl_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            doc_text = item.get("text", item.get("maintext", ""))
            ref_summary = item.get("summary", "")
            word_count = len(doc_text.split())
            if word_count > 0:
                test_samples.append({
                    "id": item.get("id"),
                    "title": item.get("title", ""),
                    "text": doc_text.strip(),
                    "summary": ref_summary.strip(),
                    "word_count": word_count,
                })
            if len(test_samples) >= CFG["num_samples"]:
                break

elif xlsum_archive_file and os.path.exists(xlsum_archive_file):
    print(f"Extracting test samples from archive: {xlsum_archive_file}...")
    mode = "r:bz2" if xlsum_archive_file.endswith(".bz2") else "r:gz"
    with tarfile.open(xlsum_archive_file, mode) as tar:
        test_member = None
        for member in tar.getmembers():
            if "test.jsonl" in member.name:
                test_member = member
                break
        if test_member:
            f = tar.extractfile(test_member)
            for line in f:
                item = json.loads(line)
                doc_text = item.get("text", item.get("maintext", ""))
                ref_summary = item.get("summary", "")
                word_count = len(doc_text.split())
                if word_count > 0:
                    test_samples.append({
                        "id": item.get("id"),
                        "title": item.get("title", ""),
                        "text": doc_text.strip(),
                        "summary": ref_summary.strip(),
                        "word_count": word_count,
                    })
                if len(test_samples) >= CFG["num_samples"]:
                    break

if not test_samples:
    print("Loading test samples directly via datasets library fallback...")
    from datasets import load_dataset
    ds = load_dataset("csebuetnlp/xl-sum", "arabic", split="test")
    for item in ds:
        doc_text = item.get("text", item.get("maintext", ""))
        ref_summary = item.get("summary", "")
        word_count = len(doc_text.split())
        if word_count > 0:
            test_samples.append({
                "id": item.get("id"),
                "title": item.get("title", ""),
                "text": doc_text.strip(),
                "summary": ref_summary.strip(),
                "word_count": word_count,
            })
        if len(test_samples) >= CFG["num_samples"]:
            break

print(f"[OK] Successfully loaded {len(test_samples)} test articles for error propagation.")
print(f"Average document length: {np.mean([s['word_count'] for s in test_samples]):.1f} words.")""")

    # 6. Quantize & Compile Models
    add_code("""# ── 5. Quantize & Compile Models ──────────────────────────────────────────
CT2_DIRS = {
    "ct2_fp16": model_ct2_path if model_ct2_path else os.path.join(CFG["models_dir"], "whisper_medium_ct2_fp16"),
    "ct2_int8": os.path.join(CFG["models_dir"], "whisper_medium_ct2_int8"),
}

# 1. Convert to CT2 Float16 if not already present
if model_hf_path and not os.path.exists(CT2_DIRS["ct2_fp16"]):
    print(f"Quantizing & Converting HF model to CTranslate2 Float16...")
    !ct2-transformers-converter --model {model_hf_path} --output_dir {CT2_DIRS['ct2_fp16']} --quantization float16
elif os.path.exists(CT2_DIRS["ct2_fp16"]):
    print(f"[OK] CTranslate2 Float16 model ready at: {CT2_DIRS['ct2_fp16']}")
elif not model_ct2_path and not model_hf_path:
    print("Notice: No local Stage-2 model found, falling back to openai/whisper-medium.")
    CT2_DIRS["ct2_fp16"] = "openai/whisper-medium"

# 2. Convert to CT2 Static INT8 (on-disk 8-bit quantization) if HF model exists
if model_hf_path and not os.path.exists(CT2_DIRS["ct2_int8"]):
    print(f"Quantizing & Converting HF model to CTranslate2 Static INT8...")
    !ct2-transformers-converter --model {model_hf_path} --output_dir {CT2_DIRS['ct2_int8']} --quantization int8
elif os.path.exists(CT2_DIRS["ct2_int8"]):
    print(f"[OK] CTranslate2 Static INT8 model ready at: {CT2_DIRS['ct2_int8']}")

def get_dir_size_mb(path):
    if not path or not os.path.exists(path):
        return 0.0
    return sum(os.path.getsize(os.path.join(root, f)) for root, _, files in os.walk(path) for f in files) / 1e6

print("\\nModel Inventory & Disk Footprints:")
for k, p in CT2_DIRS.items():
    if os.path.exists(p):
        print(f"  - {k:10s}: {get_dir_size_mb(p):.1f} MB ({p})")
if model_hf_path and os.path.exists(model_hf_path):
    print(f"  - {'HF Model':10s}: {get_dir_size_mb(model_hf_path):.1f} MB ({model_hf_path})")""")

    # 7. Helper Functions & Metric Scorers
    add_code("""# ── 6. Helper Functions for Pipeline Execution ───────────────────────────────
def normalize_arabic(text: str) -> str:
    # Deterministic Arabic normalization matching Stage 1 & Stage 2 evaluations
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[إأآٱ]', 'ا', text)              # Alef forms -> ا
    text = re.sub(r'ى', 'ي', text)                    # Dotless ya -> ي
    text = re.sub(r'ة', 'ه', text)                    # Ta-marbuta -> ه
    text = re.sub(r'[\\u064B-\\u065F\\u0670]', '', text) # Strip diacritics
    text = re.sub(r'ـ', '', text)                      # Strip tatweel
    text = re.sub(r'[^\\w\\s\\u0600-\\u06FF]', '', text)  # Keep Arabic letters & digits
    text = re.sub(r'\\s+', ' ', text)
    return text.strip()

async def synthesize_speech(text: str, output_path: str, voice: str = CFG["voice"]):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def generate_audio_sync(text: str, output_path: str):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            loop.run_until_complete(synthesize_speech(text, output_path))
        else:
            loop.run_until_complete(synthesize_speech(text, output_path))
    except Exception:
        asyncio.run(synthesize_speech(text, output_path))

# Load Summarizer (Fine-tuned AraBART)
SUMMARIZER_ID = "Omar10lfc/arabart-xlsum-arabic"
print(f"Loading Summarizer: {SUMMARIZER_ID}...")
sum_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_ID)
sum_model = AutoModelForSeq2SeqLM.from_pretrained(
    SUMMARIZER_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)
sum_model.eval()
print("[OK] AraBART loaded on device:", device)

@torch.no_grad()
def run_summarization(text: str) -> str:
    inputs = sum_tokenizer(
        [text],
        max_length=CFG["max_input_length"],
        truncation=True,
        padding="longest",
        return_tensors="pt"
    ).to(device)
    output_ids = sum_model.generate(
        **inputs,
        max_length=CFG["max_summary_length"],
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True,
    )
    return sum_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

from rouge_score import rouge_scorer

# Official XL-Sum paper multilingual ROUGE scorer (with Arabic Snowball stemmer)
scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True,
    lang='arabic'
)

def compute_rouge_scores(pred: str, ref: str) -> dict:
    scores = scorer.score(ref, pred)
    return {
        "rouge1": scores["rouge1"].fmeasure * 100.0,
        "rouge2": scores["rouge2"].fmeasure * 100.0,
        "rougeL": scores["rougeL"].fmeasure * 100.0,
    }""")

    # 8. Step 1: Synthesize Audio & Compute Oracle Summaries
    add_code("""# ── 7. Step 1: Synthesize Speech & Compute Oracle Baseline ─────────────────
print("="*80)
print(f"Step 1: Generating Neural Audio & Oracle Summaries for {len(test_samples)} articles...")
print("="*80)

audio_paths = []
oracle_summaries = []
oracle_rouges = []
audio_durations = []

t0_prep = time.time()

for idx, sample in enumerate(test_samples):
    gold_text = sample["text"]
    human_ref = sample["summary"]
    audio_path = os.path.join(CFG["audio_dir"], f"sample_{idx:03d}.mp3")

    # 1. Synthesize audio if not already generated
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        generate_audio_sync(gold_text, audio_path)
    audio_paths.append(audio_path)

    # 2. Measure audio duration
    try:
        dur = sf.info(audio_path).duration
    except Exception:
        dur = 8.0
    audio_durations.append(dur)

    # 3. Compute Oracle Summary (Clean Ground-Truth Text -> AraBART)
    oracle_sum = run_summarization(gold_text)
    oracle_summaries.append(oracle_sum)
    o_rouge = compute_rouge_scores(oracle_sum, human_ref)
    oracle_rouges.append(o_rouge)

    if (idx + 1) % 5 == 0 or (idx + 1) == len(test_samples):
        print(f"  Processed {idx+1}/{len(test_samples)} articles | Oracle ROUGE-L: {o_rouge['rougeL']:.1f}%")

total_audio_sec = sum(audio_durations)
prep_time = time.time() - t0_prep
print(f"\\n[OK] Step 1 Complete: Total Audio = {total_audio_sec:.1f}s ({total_audio_sec/60:.1f} mins) generated in {prep_time:.1f}s.")
print(f"Oracle Baseline ROUGE-L: {np.mean([r['rougeL'] for r in oracle_rouges]):.2f}%")""")

    # 9. Step 2: Multi-Quantization Cascaded Benchmark Execution
    add_code("""# ── 8. Step 2: Execute Multi-Quantization Cascaded Benchmark ────────────────
from faster_whisper import WhisperModel

# Define quantization configurations to evaluate
TIERS = [
    {
        "name": "CT2 Float16",
        "engine": "faster-whisper (CT2)",
        "precision": "float16",
        "model_path": CT2_DIRS["ct2_fp16"],
        "compute_type": "float16",
    },
    {
        "name": "CT2 INT8_FLOAT16",
        "engine": "faster-whisper (CT2)",
        "precision": "int8_float16",
        "model_path": CT2_DIRS["ct2_fp16"],
        "compute_type": "int8_float16",
    },
    {
        "name": "CT2 Static INT8",
        "engine": "faster-whisper (CT2)",
        "precision": "int8",
        "model_path": CT2_DIRS["ct2_int8"] if os.path.exists(CT2_DIRS["ct2_int8"]) else CT2_DIRS["ct2_fp16"],
        "compute_type": "int8",
    },
]

# If Hugging Face standalone model is available, add Vanilla PyTorch HF FP16
if model_hf_path and os.path.exists(model_hf_path):
    TIERS.insert(0, {
        "name": "Vanilla PyTorch HF",
        "engine": "Transformers HF",
        "precision": "FP16",
        "model_path": model_hf_path,
        "compute_type": "torch_fp16",
    })

import scipy.signal

def load_audio_16k(audio_path):
    speech, sr = sf.read(audio_path)
    if len(speech.shape) > 1:
        speech = speech.mean(axis=1)
    if sr != 16000:
        num_samples = int(len(speech) * 16000 / sr)
        speech = scipy.signal.resample(speech, num_samples)
    return speech.astype(np.float32)

print("Pre-loading audio signals at 16kHz mono (bypassing PyAV)...")
audio_16k_list = [load_audio_16k(p) for p in audio_paths]
total_audio_sec = sum(len(a) / 16000.0 for a in audio_16k_list)
print(f"[OK] Cached {len(audio_16k_list)} clips ({total_audio_sec:.1f}s audio) in memory.")

tier_evaluation_results = []
all_tier_details = {}

for tier in TIERS:
    t_name = tier["name"]
    t_prec = tier["precision"]
    m_path = tier["model_path"]
    c_type = tier["compute_type"]

    print("\\n" + "="*80)
    print(f"🚀 Benchmarking Tier: {t_name} (Precision: {t_prec})")
    print("="*80)

    torch.cuda.empty_cache()

    # Load ASR Model
    t0_load = time.time()
    if c_type == "torch_fp16":
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        whisper_proc = WhisperProcessor.from_pretrained(m_path)
        whisper_mod = WhisperForConditionalGeneration.from_pretrained(
            m_path, torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device).eval()
    else:
        whisper_ct2 = WhisperModel(m_path, device=device, compute_type=c_type)
    load_time = time.time() - t0_load
    print(f"Model loaded in {load_time:.2f}s.")

    # 1. Transcribe all audio files
    transcripts = []
    t0_transcribe = time.time()
    for idx, audio_16k in enumerate(audio_16k_list):
        if c_type == "torch_fp16":
            hf_dtype = torch.float16 if device == "cuda" else torch.float32
            inputs = whisper_proc(audio_16k, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=hf_dtype)
            with torch.no_grad():
                try:
                    forced_ids = whisper_proc.get_decoder_prompt_ids(language="ar", task="transcribe")
                    gen_ids = whisper_mod.generate(inputs, forced_decoder_ids=forced_ids, max_new_tokens=440)
                except Exception:
                    gen_ids = whisper_mod.generate(inputs, language="ar", task="transcribe", max_new_tokens=440)
            text = whisper_proc.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        else:
            segments, _ = whisper_ct2.transcribe(audio_16k, language="ar", beam_size=CFG["beam_size"])
            text = " ".join([seg.text for seg in segments]).strip()
        transcripts.append(text)

    asr_time = time.time() - t0_transcribe
    rtf = asr_time / total_audio_sec
    throughput = total_audio_sec / asr_time

    # 2. Compute ASR WER against gold text
    wers = []
    for idx, sample in enumerate(test_samples):
        w = jiwer.wer(normalize_arabic(sample["text"]), normalize_arabic(transcripts[idx])) * 100.0
        wers.append(w)
    mean_wer = np.mean(wers)

    # 3. Summarize transcripts with AraBART & Score ROUGE
    cascaded_summaries = []
    cascaded_rouges = []
    semantic_preservations = []

    for idx, sample in enumerate(test_samples):
        casc_sum = run_summarization(transcripts[idx])
        cascaded_summaries.append(casc_sum)

        # ROUGE against human reference
        r_score = compute_rouge_scores(casc_sum, sample["summary"])
        cascaded_rouges.append(r_score)

        # Direct ROUGE against Oracle AraBART summary
        inter_r = compute_rouge_scores(casc_sum, oracle_summaries[idx])
        semantic_preservations.append(inter_r["rougeL"])

    # Aggregate Metrics
    r1 = np.mean([r["rouge1"] for r in cascaded_rouges])
    r2 = np.mean([r["rouge2"] for r in cascaded_rouges])
    rl = np.mean([r["rougeL"] for r in cascaded_rouges])
    fidelity = np.mean(semantic_preservations)
    disk_mb = get_dir_size_mb(m_path)

    oracle_rl_mean = np.mean([r["rougeL"] for r in oracle_rouges])
    delta_rl = rl - oracle_rl_mean
    retention_rate = (rl / oracle_rl_mean) * 100.0

    print(f"  [METRICS] ASR WER: {mean_wer:.2f}% | RTF: {rtf:.4f} ({throughput:.1f}x real-time)")
    print(f"  [METRICS] Downstream R-L: {rl:.2f}% (Delta vs Oracle: {delta_rl:+.2f} pp | Retention: {retention_rate:.1f}%)")

    tier_res = {
        "Tier": t_name,
        "Engine": tier["engine"],
        "Precision": t_prec,
        "Disk (MB)": round(disk_mb, 1),
        "ASR WER (%)": round(mean_wer, 2),
        "Inference Time (s)": round(asr_time, 2),
        "RTF": round(rtf, 4),
        "Throughput (xRT)": round(throughput, 1),
        "ROUGE-1": round(r1, 2),
        "ROUGE-2": round(r2, 2),
        "ROUGE-L": round(rl, 2),
        "Delta RL (pp)": round(delta_rl, 2),
        "Quality Retention (%)": round(retention_rate, 1),
        "Summary Fidelity (%)": round(fidelity, 2),
    }
    tier_evaluation_results.append(tier_res)

    all_tier_details[t_name] = {
        "transcripts": transcripts,
        "summaries": cascaded_summaries,
        "wers": wers,
        "rouges": cascaded_rouges,
        "semantic_preservations": semantic_preservations,
        "disk_mb": round(disk_mb, 1),
        "inference_time": round(asr_time, 2),
        "rtf": round(rtf, 4),
        "throughput": round(throughput, 1),
        "engine": tier["engine"],
        "precision": t_prec,
    }

    # Free memory
    if c_type == "torch_fp16":
        del whisper_mod, whisper_proc
    else:
        del whisper_ct2
    torch.cuda.empty_cache()""")

    # 10. Master Comparison Table & Aggregate Analysis
    add_code("""# ── 9. Master Quantization & Cascading Error Comparison (Dual Scale: N=50 & N=25) ───
def bootstrap_ci(scores, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    scores = np.array(scores)
    boot_means = [rng.choice(scores, size=len(scores), replace=True).mean() for _ in range(n_boot)]
    low = np.percentile(boot_means, 100 * (alpha / 2))
    high = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return (round(float(low), 2), round(float(high), 2))

def evaluate_scale(n_samples):
    sliced_samples = test_samples[:n_samples]
    sliced_oracle_rouges = oracle_rouges[:n_samples]
    o_r1 = np.mean([r["rouge1"] for r in sliced_oracle_rouges])
    o_r2 = np.mean([r["rouge2"] for r in sliced_oracle_rouges])
    o_rl = np.mean([r["rougeL"] for r in sliced_oracle_rouges])
    o_rl_ci = bootstrap_ci([r["rougeL"] for r in sliced_oracle_rouges])

    rows = [
        {
            "Configuration": "Oracle Baseline (Clean Text -> AraBART)",
            "Engine": "AraBART Only",
            "Precision": "FP16",
            "Disk (MB)": "-",
            "ASR WER (%)": "0.00%",
            "Throughput": "-",
            "ROUGE-1": round(o_r1, 2),
            "ROUGE-2": round(o_r2, 2),
            "ROUGE-L": round(o_rl, 2),
            "ROUGE-L 95% CI": str(o_rl_ci),
            "Delta RL": "0.00 pp",
            "Quality Retention": "100.0% (Ref)",
        }
    ]

    tier_res_list = []
    for tier in TIERS:
        t_name = tier["name"]
        details = all_tier_details[t_name]
        t_wers = details["wers"][:n_samples]
        t_rouges = details["rouges"][:n_samples]
        m_wer = np.mean(t_wers)
        r1 = np.mean([r["rouge1"] for r in t_rouges])
        r2 = np.mean([r["rouge2"] for r in t_rouges])
        rl = np.mean([r["rougeL"] for r in t_rouges])
        t_ci = bootstrap_ci([r["rougeL"] for r in t_rouges])
        delta_rl = rl - o_rl
        retention = (rl / o_rl) * 100.0
        disk_mb = details.get("disk_mb", "-")
        tp = details.get("throughput", 0.0)

        t_res = {
            "Tier": t_name,
            "Engine": tier["engine"],
            "Precision": tier["precision"],
            "Disk (MB)": disk_mb,
            "ASR WER (%)": round(m_wer, 2),
            "Throughput (xRT)": tp,
            "ROUGE-1": round(r1, 2),
            "ROUGE-2": round(r2, 2),
            "ROUGE-L": round(rl, 2),
            "ROUGE-L 95% CI": t_ci,
            "Delta RL (pp)": round(delta_rl, 2),
            "Quality Retention (%)": round(retention, 1),
        }
        tier_res_list.append(t_res)

        rows.append({
            "Configuration": t_name,
            "Engine": tier["engine"],
            "Precision": tier["precision"],
            "Disk (MB)": disk_mb,
            "ASR WER (%)": f"{m_wer:.2f}%",
            "Throughput": f"{tp:.1f}x" if isinstance(tp, (int, float)) else str(tp),
            "ROUGE-1": round(r1, 2),
            "ROUGE-2": round(r2, 2),
            "ROUGE-L": round(rl, 2),
            "ROUGE-L 95% CI": str(t_ci),
            "Delta RL": f"{delta_rl:+.2f} pp",
            "Quality Retention": f"{retention:.1f}%",
        })

    oracle_meta = {
        "rouge1": round(o_r1, 2),
        "rouge2": round(o_r2, 2),
        "rougeL": round(o_rl, 2),
        "rougeL_ci": o_rl_ci,
    }
    return rows, tier_res_list, oracle_meta

n_total = len(test_samples)
rows_primary, tiers_primary, oracle_primary = evaluate_scale(n_total)
print("\\n" + "="*95)
print(f"📊 PRIMARY BENCHMARK: QUANTIZATION & CASCADING ERROR PROPAGATION (N={n_total} ARTICLES)")
print("="*95)
print(tabulate(rows_primary, headers="keys", tablefmt="github"))

if n_total >= 50:
    rows_25, tiers_25, oracle_25 = evaluate_scale(25)
    print("\\n" + "="*95)
    print(f"📊 SUBSET BENCHMARK: REPRODUCIBILITY & CONSISTENCY CHECK (N=25 ARTICLES)")
    print("="*95)
    print(tabulate(rows_25, headers="keys", tablefmt="github"))

    stability_rows = [
        {
            "Configuration": "Oracle Baseline",
            "WER (N=25)": "0.00%",
            "WER (N=50)": "0.00%",
            "Δ WER": "0.00 pp",
            "ROUGE-L (N=25)": f"{oracle_25['rougeL']:.2f}",
            "ROUGE-L (N=50)": f"{oracle_primary['rougeL']:.2f}",
            "Δ R-L": f"{oracle_primary['rougeL'] - oracle_25['rougeL']:+.2f} pp",
            "Retention (N=25)": "100.0%",
            "Retention (N=50)": "100.0%",
            "Δ Ret": "0.0 pp",
        }
    ]
    for t25, t50 in zip(tiers_25, tiers_primary):
        stability_rows.append({
            "Configuration": t50["Tier"],
            "WER (N=25)": f"{t25['ASR WER (%)']:.2f}%",
            "WER (N=50)": f"{t50['ASR WER (%)']:.2f}%",
            "Δ WER": f"{t50['ASR WER (%)'] - t25['ASR WER (%)']:+.2f} pp",
            "ROUGE-L (N=25)": f"{t25['ROUGE-L']:.2f}",
            "ROUGE-L (N=50)": f"{t50['ROUGE-L']:.2f}",
            "Δ R-L": f"{t50['ROUGE-L'] - t25['ROUGE-L']:+.2f} pp",
            "Retention (N=25)": f"{t25['Quality Retention (%)']:.1f}%",
            "Retention (N=50)": f"{t50['Quality Retention (%)']:.1f}%",
            "Δ Ret": f"{t50['Quality Retention (%)'] - t25['Quality Retention (%)']:+.1f} pp",
        })

    print("\\n" + "="*95)
    print("⚖️ SCALE STABILITY COMPARISON: N=25 vs. N=50 EVALUATION SAMPLES")
    print("="*95)
    print(tabulate(stability_rows, headers="keys", tablefmt="github"))

# ── Save Full CSV and JSON Artifacts ────────────────────────────────────────
df_primary = pd.DataFrame(tiers_primary)
csv_primary_path = os.path.join(CFG["output_dir"], f"quantization_error_propagation_summary_n{n_total}.csv")
df_primary.to_csv(csv_primary_path, index=False)

dual_records = []
if n_total >= 50:
    dual_records.append({
        "Experiment": "Experiment_A", "Sample_Count_N": 25,
        "Configuration": "Oracle Baseline (Clean Text -> AraBART)", "Engine": "AraBART Only",
        "Precision": "FP16", "Disk_MB": "-", "ASR_WER_pct": 0.0, "Throughput_xRT": "-",
        "ROUGE_1": oracle_25["rouge1"], "ROUGE_2": oracle_25["rouge2"], "ROUGE_L": oracle_25["rougeL"],
        "Delta_RL_pp": 0.0, "Quality_Retention_pct": "100.0%"
    })
    for t in tiers_25:
        dual_records.append({
            "Experiment": "Experiment_A", "Sample_Count_N": 25,
            "Configuration": t["Tier"], "Engine": t["Engine"], "Precision": t["Precision"],
            "Disk_MB": t["Disk (MB)"], "ASR_WER_pct": t["ASR WER (%)"], "Throughput_xRT": t["Throughput (xRT)"],
            "ROUGE_1": t["ROUGE-1"], "ROUGE_2": t["ROUGE-2"], "ROUGE_L": t["ROUGE-L"],
            "Delta_RL_pp": t["Delta RL (pp)"], "Quality_Retention_pct": f"{t['Quality Retention (%)']:.1f}%"
        })

dual_records.append({
    "Experiment": "Experiment_B", "Sample_Count_N": n_total,
    "Configuration": "Oracle Baseline (Clean Text -> AraBART)", "Engine": "AraBART Only",
    "Precision": "FP16", "Disk_MB": "-", "ASR_WER_pct": 0.0, "Throughput_xRT": "-",
    "ROUGE_1": oracle_primary["rouge1"], "ROUGE_2": oracle_primary["rouge2"], "ROUGE_L": oracle_primary["rougeL"],
    "Delta_RL_pp": 0.0, "Quality_Retention_pct": "100.0%"
})
for t in tiers_primary:
    dual_records.append({
        "Experiment": "Experiment_B", "Sample_Count_N": n_total,
        "Configuration": t["Tier"], "Engine": t["Engine"], "Precision": t["Precision"],
        "Disk_MB": t["Disk (MB)"], "ASR_WER_pct": t["ASR WER (%)"], "Throughput_xRT": t["Throughput (xRT)"],
        "ROUGE_1": t["ROUGE-1"], "ROUGE_2": t["ROUGE-2"], "ROUGE_L": t["ROUGE-L"],
        "Delta_RL_pp": t["Delta RL (pp)"], "Quality_Retention_pct": f"{t['Quality Retention (%)']:.1f}%"
    })

csv_dual_path = os.path.join(CFG["output_dir"], "quantization_error_propagation_dual_scale.csv")
pd.DataFrame(dual_records).to_csv(csv_dual_path, index=False)

def format_json_experiment(tiers_data, oracle_meta, n_count):
    res_list = [
        {
            "configuration": "Oracle Baseline (Clean Text -> AraBART)",
            "engine": "AraBART Only",
            "precision": "FP16",
            "disk_mb": None,
            "asr_wer_pct": 0.0,
            "throughput_xrt": None,
            "rouge1": oracle_meta["rouge1"],
            "rouge2": oracle_meta["rouge2"],
            "rougeL": oracle_meta["rougeL"],
            "delta_rl_pp": 0.0,
            "quality_retention_pct": 100.0
        }
    ]
    for t in tiers_data:
        res_list.append({
            "configuration": t["Tier"],
            "engine": t["Engine"],
            "precision": t["Precision"],
            "disk_mb": t["Disk (MB)"] if isinstance(t["Disk (MB)"], (int, float)) else None,
            "asr_wer_pct": t["ASR WER (%)"],
            "throughput_xrt": t["Throughput (xRT)"] if isinstance(t["Throughput (xRT)"], (int, float)) else None,
            "rouge1": t["ROUGE-1"],
            "rouge2": t["ROUGE-2"],
            "rougeL": t["ROUGE-L"],
            "delta_rl_pp": t["Delta RL (pp)"],
            "quality_retention_pct": t["Quality Retention (%)"]
        })
    return {
        "sample_count": n_count,
        "results": res_list
    }

json_export_data = {
    "benchmark": "End-to-End Cascading Error Propagation & Quantization Benchmark",
    "dataset": "Arabic XL-Sum Held-Out Test Set (BBC Arabic)",
    "scorer": "Official XL-Sum multilingual_rouge_scoring (NLTK Arabic Snowball stemmer)",
    "experiments": {
        "experiment_b_n50": format_json_experiment(tiers_primary, oracle_primary, n_total)
    }
}
if n_total >= 50:
    json_export_data["experiments"]["experiment_a_n25"] = format_json_experiment(tiers_25, oracle_25, 25)

json_results_path = os.path.join(CFG["output_dir"], "error_propagation_results.json")
with open(json_results_path, "w", encoding="utf-8") as f:
    json.dump(json_export_data, f, ensure_ascii=False, indent=2)

export_records = []
for idx, s in enumerate(test_samples):
    rec = {
        "id": s["id"],
        "title": s["title"],
        "gold_text": s["text"],
        "human_ref": s["summary"],
        "oracle_summary": oracle_summaries[idx],
        "oracle_rouge": oracle_rouges[idx],
        "tiers": {}
    }
    for t_name, details in all_tier_details.items():
        rec["tiers"][t_name] = {
            "transcript": details["transcripts"][idx],
            "summary": details["summaries"][idx],
            "wer": details["wers"][idx],
            "rouge": details["rouges"][idx],
        }
    export_records.append(rec)

json_records_path = os.path.join(CFG["output_dir"], "error_propagation_records.json")
with open(json_records_path, "w", encoding="utf-8") as f:
    json.dump(export_records, f, ensure_ascii=False, indent=2)

print(f"\\n[OK] Results saved to:")
print(f"  - {csv_primary_path}")
print(f"  - {csv_dual_path}")
print(f"  - {json_results_path}")
print(f"  - {json_records_path}")""")

    # 11. Qualitative Case Study Display across Tiers
    add_code("""# ── 10. Qualitative Case Study: Cross-Quantization Error Resilience ───────────
print("\\n" + "="*90)
print("🔍 QUALITATIVE INSPECTION: RECOVERY & RESILIENCE ACROSS QUANTIZATION TIERS")
print("="*90)

sample_idx = 0
s = test_samples[sample_idx]
print(f"Headline: {s['title']}")
print(f"Human Reference Summary:")
print(f"  {s['summary']}\\n")

print(f"--- 1. ORACLE SUMMARY (From Clean Ground Truth Text) ---")
print(f"  {oracle_summaries[sample_idx]}\\n")

for tier in TIERS:
    t_name = tier["name"]
    t_wer = all_tier_details[t_name]["wers"][sample_idx]
    t_sum = all_tier_details[t_name]["summaries"][sample_idx]
    t_rl = all_tier_details[t_name]["rouges"][sample_idx]["rougeL"]
    print(f"--- 2. {t_name.upper()} (WER: {t_wer:.1f}% | ROUGE-L: {t_rl:.1f}%) ---")
    print(f"  Summary: {t_sum}\\n")

print("="*90)
eval_tiers = tiers_primary if "tiers_primary" in locals() else tiers_25
if eval_tiers:
    sorted_tiers = sorted(eval_tiers, key=lambda x: x["ROUGE-L"])
    worst_tier = sorted_tiers[0]
    best_tier = sorted_tiers[-1]
    delta_rl = best_tier["ROUGE-L"] - worst_tier["ROUGE-L"]
    print(f"Quantization Tier ROUGE-L Span (N={len(test_samples)}): Best tier is '{best_tier['Tier']}' ({best_tier['ROUGE-L']:.2f}), "
          f"lowest tier is '{worst_tier['Tier']}' ({worst_tier['ROUGE-L']:.2f}), "
          f"yielding a delta of {delta_rl:.2f} pp across evaluated quantization tiers.")
print("="*90)""")

    # Validate Python syntax of all code cells
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code_str = "".join(cell["source"])
            py_lines = [l for l in code_str.splitlines() if not l.strip().startswith("!") and not l.strip().startswith("%")]
            py_str = "\n".join(py_lines)
            try:
                ast.parse(py_str)
            except SyntaxError as e:
                print(f"Syntax error in cell {idx}: {e}")
                raise

    return nb

if __name__ == "__main__":
    notebook = create_error_propagation_notebook()
    target_path = os.path.abspath("evaluate_error_propagation.ipynb")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"[OK] Generated notebook successfully at: {target_path}")

    # Also sync to Notebooks/ directory
    notebooks_target = os.path.abspath("Notebooks/evaluate_error_propagation.ipynb")
    shutil.copyfile(target_path, notebooks_target)
    print(f"[OK] Synced copy to: {notebooks_target}")
