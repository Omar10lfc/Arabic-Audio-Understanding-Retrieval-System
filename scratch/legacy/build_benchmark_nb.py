import json
import os
import ast

def create_notebook():
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
    add_md("""# 📊 Whisper-Medium Arabic: Quantization & Performance Benchmark
### Evaluating Precision Tiers (FP16 vs INT8 vs INT4) on Mozilla Common Voice 25.0 Arabic

**Objective:**
Empirically measure the impact of post-training quantization and CTranslate2 optimization on:
1. **Word Error Rate (WER %)**: Prove zero/near-zero quality regression.
2. **Inference Latency & Real-Time Factor (RTF)**: Seconds of compute per second of audio.
3. **Throughput (Speedup factor)**: Multiple over real-time and over vanilla PyTorch.
4. **VRAM Footprint & On-Disk Model Size**: Memory efficiency.

**Target Hardware:** NVIDIA T4 GPU (16 GB) on Kaggle.""")

    # 2. Pip Install
    add_code("""# ── 1. Install Inference Dependencies ──────────────────────────────────────────
!pip install -q faster-whisper ctranslate2 jiwer evaluate soundfile pandas tabulate
print("✅ Dependencies installed successfully.")""")

    # 3. Environment Check
    add_code("""# ── 2. Environment Verification ───────────────────────────────────────────────
import os
import sys
import time
import re
import glob
import json
import shutil
import torch
import soundfile as sf
import pandas as pd
import numpy as np
import jiwer
from tabulate import tabulate

print(f"PyTorch     : {torch.__version__}")
print(f"CUDA Avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM Total  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ WARNING: Running on CPU. Benchmarks are designed for GPU.")""")

    # 4. Configuration & Auto-Detection
    add_code("""# ── 3. Configuration & Auto-Detection of Models and Dataset ────────────────────
CFG = {
    "n_test": 400,                   # Number of held-out test clips
    "beam_size": 1,                  # Greedy decoding (standard fast inference)
    "language": "ar",
    "seed": 42,
    "output_dir": "/kaggle/working/quantization_benchmarks",
}
os.makedirs(CFG["output_dir"], exist_ok=True)

# 1. Print all mounted inputs to verify attached datasets/models
print("── Mounted Inputs in /kaggle/input ──")
input_dirs = os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else []
if input_dirs:
    for item in input_dirs:
        print(f"  📁 /kaggle/input/{item}")
else:
    print("  ⚠️ No inputs mounted in /kaggle/input")

# 2. Auto-detect Common Voice Arabic dataset directory
DATASET_CANDIDATES = [
    "/kaggle/input/datasets/omar10lfc/common-voice-scripted-speech-25-0-arabic",
    "/kaggle/input/common-voice-scripted-speech-25-0-arabic",
    "/kaggle/input/*common-voice*arabic*",
]
data_root = None
for cand in DATASET_CANDIDATES:
    matches = glob.glob(cand)
    if matches and os.path.exists(matches[0]):
        data_root = matches[0]
        break

if not data_root and os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        if "test.tsv" in files:
            data_root = root
            break

if not data_root:
    raise FileNotFoundError("Could not find Common Voice Arabic dataset! Please attach omar10lfc/common-voice-scripted-speech-25-0-arabic via '+ Add Input'.")
print(f"\\n✅ Found Dataset Root: {data_root}")

# 3. Recursive auto-detect for Stage 2 models (CT2 & Hugging Face)
print("\\nScanning /kaggle/input and /kaggle/working for fine-tuned models...")
model_ct2_path = None
model_hf_path = None

search_roots = ["/kaggle/input", "/kaggle/working"]
for sroot in search_roots:
    if not os.path.exists(sroot):
        continue
    for root, dirs, files in os.walk(sroot):
        # Detect CTranslate2 model: contains model.bin and vocabulary.json/txt
        if "model.bin" in files and ("vocabulary.json" in files or "vocabulary.txt" in files):
            if not model_ct2_path or "stage2" in root.lower() or "ct2" in root.lower():
                model_ct2_path = root
                print(f"  ✅ Found CT2 model at: {root}")

        # Detect Standalone HF model: contains config.json and (.safetensors or .bin)
        if "config.json" in files and any(f.endswith(".safetensors") or f.endswith(".bin") for f in files):
            if "model.bin" not in files: # Exclude CT2 directory
                if not model_hf_path or "stage2" in root.lower() or "merged" in root.lower():
                    model_hf_path = root
                    print(f"  ✅ Found HF Standalone model at: {root}")

print(f"\\nDetected Models:")
print(f"  - HF Standalone Model : {model_hf_path}")
print(f"  - CT2 Pre-built Model : {model_ct2_path}")

if not model_hf_path and not model_ct2_path:
    print("\\n" + "!"*72)
    print("⚠️  ACTION REQUIRED: STAGE-2 MODEL WAS NOT FOUND IN /kaggle/input!")
    print("To attach your fine-tuned Stage-2 model to this notebook:")
    print("  1. In the right panel of this notebook, click '+ Add Input'")
    print("  2. In the search box, click the 'Your Work' or 'Notebooks' tab")
    print("  3. Find 'stage-2' (or search 'stage-2')")
    print("  4. Click the '+' button next to it to attach its output")
    print("  5. Re-run this cell!")
    print("!"*72 + "\\n")
    raise FileNotFoundError("Stage-2 model not found. Please attach the 'stage-2' kernel output using the steps above.")""")

    # 5. Arabic Normalizer & Data Loader
    add_code("""# ── 4. Arabic Normalization & Test Dataset Loader ─────────────────────────────
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

print("Locating test.tsv...")
tsv_path = None
for root, _, files in os.walk(data_root):
    if "test.tsv" in files:
        tsv_path = os.path.join(root, "test.tsv")
        break

if not tsv_path:
    raise FileNotFoundError("test.tsv not found in dataset!")

print(f"Found test.tsv at: {tsv_path}")
df_raw = pd.read_csv(tsv_path, sep='\\t', low_memory=False)
path_col = 'path' if 'path' in df_raw.columns else df_raw.columns[1]
text_col = 'sentence' if 'sentence' in df_raw.columns else 'text'

# Locate audio clips directory
tsv_dir = os.path.dirname(tsv_path)
clips_candidate = os.path.join(tsv_dir, "clips")
if not os.path.isdir(clips_candidate):
    clips_candidate = os.path.join(data_root, "clips")

if os.path.isdir(clips_candidate):
    print(f"Found clips directory at: {clips_candidate}")
    def resolve_clip(p):
        full = os.path.join(clips_candidate, str(p))
        if os.path.exists(full):
            return full
        if not full.endswith(".mp3") and os.path.exists(full + ".mp3"):
            return full + ".mp3"
        return None
    df_raw['audio_path'] = df_raw[path_col].apply(resolve_clip)
else:
    print("Indexing audio files recursively...")
    audio_index = {}
    for root, _, files in os.walk(data_root):
        for f in files:
            if f.endswith(('.mp3', '.wav', '.ogg')):
                audio_index[f] = os.path.join(root, f)
                audio_index[os.path.splitext(f)[0]] = os.path.join(root, f)
    def resolve_audio(p):
        fname = os.path.basename(str(p))
        stem = os.path.splitext(fname)[0]
        return audio_index.get(fname) or audio_index.get(stem)
    df_raw['audio_path'] = df_raw[path_col].apply(resolve_audio)

df_raw['clean_ref'] = df_raw[text_col].apply(normalize_arabic)
df_test = df_raw.dropna(subset=['audio_path'])
df_test = df_test[df_test['clean_ref'].str.len() > 2].reset_index(drop=True)

# Select test slice
df_test = df_test.sample(n=min(CFG['n_test'], len(df_test)), random_state=CFG['seed']).reset_index(drop=True)
print(f"✅ Loaded {len(df_test)} test audio samples for evaluation.")""")

    # 6. Quantization Exports
    add_code("""# ── 5. Generate CTranslate2 Models (FP16, INT8, INT8_FP16, INT4) ──────────────
CT2_DIRS = {
    "ct2_fp16": model_ct2_path if model_ct2_path else os.path.join(CFG["output_dir"], "whisper-medium-ct2-fp16"),
    "ct2_int8": os.path.join(CFG["output_dir"], "whisper-medium-ct2-int8"),
    "ct2_int4": os.path.join(CFG["output_dir"], "whisper-medium-ct2-int4"),
}

# 1. If CT2 FP16 doesn't exist yet and HF model available, convert it
if not os.path.exists(CT2_DIRS["ct2_fp16"]) and model_hf_path:
    print("Converting HF model to CTranslate2 FP16...")
    !ct2-transformers-converter --model {model_hf_path} --output_dir {CT2_DIRS['ct2_fp16']} --quantization float16
elif os.path.exists(CT2_DIRS["ct2_fp16"]):
    print(f"✅ CTranslate2 FP16 model ready at: {CT2_DIRS['ct2_fp16']}")

# 2. Convert to Static INT8 on disk if HF model available
if not os.path.exists(CT2_DIRS["ct2_int8"]) and model_hf_path:
    print("Converting HF model to CTranslate2 Static INT8 (On-Disk Quantization)...")
    !ct2-transformers-converter --model {model_hf_path} --output_dir {CT2_DIRS['ct2_int8']} --quantization int8
elif os.path.exists(CT2_DIRS["ct2_int8"]):
    print(f"✅ CTranslate2 Static INT8 model ready at: {CT2_DIRS['ct2_int8']}")

# 3. Convert to Static INT4 on disk if HF model available
if not os.path.exists(CT2_DIRS["ct2_int4"]) and model_hf_path:
    print("Converting HF model to CTranslate2 Static INT4 (Aggressive Quantization)...")
    !ct2-transformers-converter --model {model_hf_path} --output_dir {CT2_DIRS['ct2_int4']} --quantization int4
elif os.path.exists(CT2_DIRS["ct2_int4"]):
    print(f"✅ CTranslate2 Static INT4 model ready at: {CT2_DIRS['ct2_int4']}")

print("\\nModel directories on disk:")
for k, p in CT2_DIRS.items():
    if os.path.exists(p):
        size_mb = sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))) / 1e6
        print(f"  - {k:10s}: {size_mb:.1f} MB  ({p})")""")

    # 7. Benchmarking Engine
    add_code("""# ── 6. Benchmark Execution Harness ───────────────────────────────────────────
from faster_whisper import WhisperModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# Metric function
def compute_wer(predictions, references):
    return jiwer.wer(references, predictions) * 100.0

def get_audio_duration(path):
    try:
        info = sf.info(path)
        return info.duration
    except Exception:
        return 5.0 # fallback

print("Calculating total audio duration for test set...")
durations = [get_audio_duration(p) for p in df_test['audio_path']]
total_audio_sec = sum(durations)
print(f"Total audio duration: {total_audio_sec:.1f} seconds ({total_audio_sec/60:.1f} minutes).")

results = []

def run_ct2_benchmark(tier_name, model_dir, compute_type, device="cuda"):
    print(f"\\n{'='*60}\\nRunning Benchmark: {tier_name} (compute_type={compute_type})...\\n{'='*60}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    t0_load = time.time()
    model = WhisperModel(model_dir, device=device, compute_type=compute_type)
    load_time = time.time() - t0_load
    
    # Warmup
    for i in range(2):
        _ = list(model.transcribe(df_test['audio_path'].iloc[i], language='ar', beam_size=1)[0])
        
    t0_infer = time.time()
    preds = []
    
    for i, path in enumerate(df_test['audio_path']):
        segments, _ = model.transcribe(path, language='ar', beam_size=1)
        text = " ".join([seg.text for seg in segments])
        preds.append(normalize_arabic(text))
        if (i + 1) % 50 == 0 or (i + 1) == len(df_test):
            elapsed = time.time() - t0_infer
            print(f"  Processed {i+1}/{len(df_test)} clips in {elapsed:.1f}s...")
            
    total_time = time.time() - t0_infer
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1e6
    wer = compute_wer(preds, df_test['clean_ref'].tolist())
    rtf = total_time / total_audio_sec
    speedup = total_audio_sec / total_time
    
    # Measure disk size
    disk_mb = sum(os.path.getsize(os.path.join(model_dir, f)) for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))) / 1e6
    
    res = {
        "Configuration": tier_name,
        "Engine": "faster-whisper (CT2)",
        "Precision": compute_type,
        "Disk Size (MB)": round(disk_mb, 1),
        "Peak VRAM (MB)": round(peak_vram_mb, 1),
        "Inference Time (s)": round(total_time, 2),
        "RTF (Compute/Audio)": round(rtf, 4),
        "Throughput (xRealtime)": round(speedup, 1),
        "WER (%)": round(wer, 2),
    }
    print(f"\\n🎯 Result for {tier_name}: WER = {wer:.2f}% | RTF = {rtf:.4f} ({speedup:.1f}x real-time) | VRAM = {peak_vram_mb:.1f} MB")
    del model
    torch.cuda.empty_cache()
    return res""")

    # 8. Run All Benchmarks
    add_code("""# ── 7. Execute All Quantization Tiers ─────────────────────────────────────────

# 1. CTranslate2 FP16
if os.path.exists(CT2_DIRS["ct2_fp16"]):
    res_fp16 = run_ct2_benchmark("CT2 Float16", CT2_DIRS["ct2_fp16"], compute_type="float16")
    results.append(res_fp16)

# 2. CTranslate2 INT8_FLOAT16 (Dynamic 8-bit weights, FP16 compute - optimal for GPU Tensor Cores)
if os.path.exists(CT2_DIRS["ct2_fp16"]):
    res_int8_fp16 = run_ct2_benchmark("CT2 INT8_FLOAT16", CT2_DIRS["ct2_fp16"], compute_type="int8_float16")
    results.append(res_int8_fp16)

# 3. CTranslate2 Static INT8 (Pure 8-bit)
if os.path.exists(CT2_DIRS["ct2_int8"]):
    res_int8 = run_ct2_benchmark("CT2 Static INT8", CT2_DIRS["ct2_int8"], compute_type="int8")
    results.append(res_int8)

# 4. CTranslate2 Static INT4 (Aggressive 4-bit)
if os.path.exists(CT2_DIRS["ct2_int4"]):
    res_int4 = run_ct2_benchmark("CT2 Static INT4", CT2_DIRS["ct2_int4"], compute_type="int4")
    results.append(res_int4)""")

    # 9. Optional: Vanilla Hugging Face Baseline
    add_code("""# ── 8. Vanilla PyTorch Transformers (FP16 Baseline) ───────────────────────────
if model_hf_path and os.path.exists(model_hf_path):
    print(f"\\n{'='*60}\\nRunning Baseline: Vanilla PyTorch Transformers (FP16)...\\n{'='*60}")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        processor = WhisperProcessor.from_pretrained(model_hf_path)
    except Exception:
        processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language='ar', task='transcribe')
        
    hf_model = WhisperForConditionalGeneration.from_pretrained(model_hf_path, torch_dtype=torch.float16).to("cuda")
    hf_model.eval()
    
    # Warmup
    with torch.no_grad():
        dummy = torch.randn(1, 80, 3000, dtype=torch.float16, device="cuda")
        _ = hf_model.generate(dummy, max_new_tokens=20)
        
    t0_hf = time.time()
    hf_preds = []
    
    from faster_whisper.audio import decode_audio
    with torch.no_grad():
        for i, path in enumerate(df_test['audio_path']):
            audio = decode_audio(path, sampling_rate=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", dtype=torch.float16)
            pred_ids = hf_model.generate(inputs, language='ar', task='transcribe', max_new_tokens=80)
            text = processor.batch_decode(pred_ids, skip_special_tokens=True)[0]
            hf_preds.append(normalize_arabic(text))
            
            if (i + 1) % 50 == 0 or (i + 1) == len(df_test):
                elapsed = time.time() - t0_hf
                print(f"  Processed {i+1}/{len(df_test)} clips in {elapsed:.1f}s...")
                
    total_time_hf = time.time() - t0_hf
    peak_vram_hf = torch.cuda.max_memory_allocated() / 1e6
    wer_hf = compute_wer(hf_preds, df_test['clean_ref'].tolist())
    rtf_hf = total_time_hf / total_audio_sec
    speedup_hf = total_audio_sec / total_time_hf
    disk_hf = sum(os.path.getsize(os.path.join(model_hf_path, f)) for f in os.listdir(model_hf_path) if os.path.isfile(os.path.join(model_hf_path, f))) / 1e6
    
    res_hf = {
        "Configuration": "Vanilla PyTorch HF",
        "Engine": "Transformers",
        "Precision": "FP16",
        "Disk Size (MB)": round(disk_hf, 1),
        "Peak VRAM (MB)": round(peak_vram_hf, 1),
        "Inference Time (s)": round(total_time_hf, 2),
        "RTF (Compute/Audio)": round(rtf_hf, 4),
        "Throughput (xRealtime)": round(speedup_hf, 1),
        "WER (%)": round(wer_hf, 2),
    }
    results.insert(0, res_hf)
    print(f"\\n🎯 Vanilla PyTorch HF: WER = {wer_hf:.2f}% | RTF = {rtf_hf:.4f} | VRAM = {peak_vram_hf:.1f} MB")
    del hf_model, processor
    torch.cuda.empty_cache()""")

    # 10. Summary Table & Report Generation
    add_code("""# ── 9. Final Benchmark Report & Resume Metrics ────────────────────────────────
df_results = pd.DataFrame(results)

# Calculate Relative Speedup vs Vanilla PyTorch baseline (or CT2 Float16)
if "Vanilla PyTorch HF" in df_results["Configuration"].values:
    baseline_time = df_results.loc[df_results["Configuration"] == "Vanilla PyTorch HF", "Inference Time (s)"].values[0]
    df_results["Speedup vs PyTorch"] = (baseline_time / df_results["Inference Time (s)"]).round(2).astype(str) + "x"
    baseline_wer = df_results.loc[df_results["Configuration"] == "Vanilla PyTorch HF", "WER (%)"].values[0]
    df_results["WER Delta"] = (df_results["WER (%)"] - baseline_wer).round(2).astype(str) + " pp"
elif "CT2 Float16" in df_results["Configuration"].values:
    baseline_time = df_results.loc[df_results["Configuration"] == "CT2 Float16", "Inference Time (s)"].values[0]
    df_results["Speedup vs CT2 FP16"] = (baseline_time / df_results["Inference Time (s)"]).round(2).astype(str) + "x"
    baseline_wer = df_results.loc[df_results["Configuration"] == "CT2 Float16", "WER (%)"].values[0]
    df_results["WER Delta"] = (df_results["WER (%)"] - baseline_wer).round(2).astype(str) + " pp"
else:
    df_results["Speedup"] = "-"
    df_results["WER Delta"] = "-"

print("\\n" + "="*80)
print("📊 FINAL QUANTIZATION & INFERENCE BENCHMARK RESULTS")
print("="*80)
print(tabulate(df_results, headers='keys', tablefmt='github', showindex=False))

# Export to CSV and JSON
csv_path = os.path.join(CFG["output_dir"], "quantization_benchmark_results.csv")
json_path = os.path.join(CFG["output_dir"], "quantization_benchmark_results.json")
df_results.to_csv(csv_path, index=False)
df_results.to_json(json_path, orient='records', indent=2)

print(f"\\n✅ Results saved to:\\n  - {csv_path}\\n  - {json_path}")""")

    # Validate Python syntax of all code cells
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code_str = "".join(cell["source"])
            py_lines = [l for l in code_str.splitlines() if not l.strip().startswith("!") and not l.strip().startswith("%")]
            py_str = "\n".join(py_lines)
            try:
                ast.parse(py_str)
            except SyntaxError as e:
                print(f"❌ Syntax error in cell {idx}: {e}")
                raise

    return nb

if __name__ == "__main__":
    notebook = create_notebook()
    target_path = os.path.abspath("benchmark_quantization.ipynb")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"[OK] Generated notebook successfully at: {target_path}")
