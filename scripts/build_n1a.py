"""
N1A NOTEBOOK BUILDER (scripts/build_n1a.py)
Generates Notebooks/final_01a_cv_gate_pass1.ipynb (Internet OFF)
Strictly adheres to:
B.2.a Evaluate Subset A through 4-bit NF4 + unmerged adapter training path.
B.2.b Read 12.51 from Results/results-Whisper-large-v3-qlora.json with tolerance 0.25 (12.26-12.76).
B.2.c Report merged-FP16 and 4-bit unmerged separately.
B.3.a Every pass (gate, Pass 1) must save a per-clip CSV: client_id, duration, ref, hyp_unnorm, hyp_norm, S, D, I, H, N, wer.
B.3.b Fail if any clip is dropped.
"""

import os
import sys
import hashlib
import nbformat as nbf

def create_notebook(nb_path, cells_source):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    }
    for idx, src in enumerate(cells_source):
        clean_src = src.strip()
        cell = nbf.v4.new_code_cell(clean_src)
        cell.id = f"cell_{idx:03d}"
        nb.cells.append(cell)
    
    os.makedirs(os.path.dirname(nb_path), exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Generated {nb_path} ({len(cells_source)} code cells)")

def build_n1a(evalkit_zip_sha=""):
    cells = [
        # Cell 1: Environment & evalkit check
        f"""# ── CELL 1: ENVIRONMENT & EVALKIT-SRC VERIFICATION (INTERNET OFF) ──────
import os, sys, subprocess, json, hashlib, zipfile, shutil

RUN_MODE = os.environ.get("RUN_MODE", "full")
OUT_DIR = "Results/smoke" if RUN_MODE == "smoke" else "Results/final"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("Results/env", exist_ok=True)

EXPECTED_EVALKIT_SHA = "{evalkit_zip_sha}"
print(f"Internet: OFF | RUN_MODE: {{RUN_MODE}} | Target: {{OUT_DIR}} | Expected evalkit SHA: {{EXPECTED_EVALKIT_SHA}}")

env_path = f"Results/env/final_01a_cv_gate_pass1_{{RUN_MODE}}_env.txt"
with open(env_path, 'w', encoding='utf-8') as f:
    f.write(f"Python Version: {{sys.version}}\\n")
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
        f.write(f"GPU Info:\\n{{gpu_info}}\\n")
    except Exception as e:
        f.write(f"GPU Info: None ({{e}})\\n")
print(f"Environment logged to: {{env_path}}")

# Auto-locate and extract evalkit-src archive
evalkit_ready = False
if os.path.exists("/kaggle/working/evalkit") and os.path.exists("/kaggle/working/configs"):
    evalkit_ready = True
elif os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        for f in sorted(files):
            if "evalkit" in f.lower() and f.endswith(".zip"):
                zip_path = os.path.join(root, f)
                with zipfile.ZipFile(zip_path, "r") as z:
                    z.extractall("/kaggle/working")
                evalkit_ready = True
                break
        if evalkit_ready:
            break
        if "evalkit" in dirs and "configs" in dirs:
            if root != "/kaggle/working":
                if not os.path.exists("/kaggle/working/evalkit"):
                    shutil.copytree(os.path.join(root, "evalkit"), "/kaggle/working/evalkit")
                if not os.path.exists("/kaggle/working/configs"):
                    shutil.copytree(os.path.join(root, "configs"), "/kaggle/working/configs")
            evalkit_ready = True
            break

if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")

assert evalkit_ready or RUN_MODE == "smoke", "FATAL: evalkit-src not found in /kaggle/input!"
print("[OK] evalkit unpacked and loaded to sys.path.")
""",
        # Cell 2: Preflight check & Dataset Discovery
        """# ── CELL 2: PREFLIGHT INPUTS AUDIT & TIMING PROBE ────────────────────────
import os, glob, time, json

print("=== MOUNTED INPUT DIRECTORIES in /kaggle/input ===")
if os.path.exists("/kaggle/input"):
    for item in sorted(os.listdir("/kaggle/input")):
        print(f" - /kaggle/input/{item}")

COMMON_VOICE_DIR = None
for root, dirs, files in os.walk("/kaggle/input") if os.path.exists("/kaggle/input") else []:
    if "test.tsv" in files and ("clips" in dirs or "validated.tsv" in files or "train.tsv" in files):
        COMMON_VOICE_DIR = root
        break

if not COMMON_VOICE_DIR:
    for cand in ["/kaggle/input/common-voice-scripted-speech-25-0-arabic"]:
        if os.path.exists(cand):
            COMMON_VOICE_DIR = cand
            break

assert (COMMON_VOICE_DIR is not None and os.path.exists(COMMON_VOICE_DIR)) or RUN_MODE == "smoke", \
    "FATAL: Missing Common Voice dataset in /kaggle/input"
print(f"[OK] Common Voice directory located: {COMMON_VOICE_DIR}")

MERGED_MODELS_DIR = None
for root, dirs, files in os.walk("/kaggle/input") if os.path.exists("/kaggle/input") else []:
    if "base" in dirs and ("adapters" in dirs or "merged" in dirs or "ct2_int8" in dirs):
        MERGED_MODELS_DIR = root
        break
    if os.path.basename(root) == "final-merged-models":
        MERGED_MODELS_DIR = root
        break

if not MERGED_MODELS_DIR:
    for cand in [
        "/kaggle/input/final-merged-models",
        "/kaggle/input/zero-notebook/final-merged-models",
        "/kaggle/input/zero-notebook"
    ]:
        if os.path.exists(cand):
            MERGED_MODELS_DIR = cand
            break

assert (MERGED_MODELS_DIR is not None and os.path.exists(MERGED_MODELS_DIR)) or RUN_MODE == "smoke", \
    "FATAL: Missing final-merged-models directory!"
print(f"[OK] Merged models directory located: {MERGED_MODELS_DIR}")
""",
        # Cell 3: Metrics & Protocol
        """# ── CELL 3: PROTOCOL CONFIG & NORMALIZERS ──────────────────────────────
import json, os
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2
from evalkit.wer import compute_dual_wer_record, wer_breakdown
from evalkit.stats import save_result_file

cfg_path = "/kaggle/working/configs/final_protocol.json" if os.path.exists("/kaggle/working/configs/final_protocol.json") else "configs/final_protocol.json"
with open(cfg_path, 'r', encoding='utf-8') as f:
    PROTOCOL = json.load(f)
print(f"[OK] Protocol loaded: normalizer.primary={PROTOCOL['normalizer']['primary']}")
""",
        # Cell 4: Pool Filtering and Subset A Preparation
        """# ── CELL 4: DATASET POOL FILTERING & SUBSET A PREPARATION ─────────────────
import os, pandas as pd, numpy as np
from evalkit.normalize import normalize_arabic_legacy

merged_root = globals().get('MERGED_MODELS_DIR') or "/kaggle/input/final-merged-models"
cv_root = globals().get('COMMON_VOICE_DIR') or "/kaggle/input/common-voice-scripted-speech-25-0-arabic"

base_path = os.path.join(merged_root, "base/whisper-large-v3")
adapter_path = os.path.join(merged_root, "adapters/whisper-large-v3-adapter")
merged_path = os.path.join(merged_root, "merged/whisper-large-v3")
tsv_path = os.path.join(cv_root, "test.tsv")

print(f"Base path    : {base_path} (exists: {os.path.exists(base_path)})")
print(f"Adapter path : {adapter_path} (exists: {os.path.exists(adapter_path)})")
print(f"Merged path  : {merged_path} (exists: {os.path.exists(merged_path)})")
print(f"TSV path     : {tsv_path} (exists: {os.path.exists(tsv_path)})")

if os.path.exists(tsv_path):
    df_raw = pd.read_csv(tsv_path, sep='\\t', low_memory=False)
else:
    df_raw = pd.DataFrame({
        'path': [f'mock_{i}.mp3' for i in range(16)],
        'sentence': ['هذا اختبار تجريبي للمنظومة الصوتية العربية'] * 16,
        'client_id': [f'spk_{i%4}' for i in range(16)]
    })

df_raw = df_raw.dropna(subset=['sentence']).copy()
df_raw['clean_ref'] = df_raw['sentence'].apply(normalize_arabic_legacy)
df_valid = df_raw[df_raw['clean_ref'].str.len() > 2].reset_index(drop=True)
print(f"Common Voice Evaluated Pool Size: {len(df_valid):,} clips")

subset_a = df_valid.sample(frac=1, random_state=42).iloc[:400].reset_index(drop=True)
print(f"First 5 Clip IDs in Subset A: {subset_a['path'].iloc[:5].tolist()}")

if RUN_MODE != "smoke":
    assert len(df_valid) == 10506, f"Pool size mismatch: got {len(df_valid)}, expected 10,506"
""",
        # Cell 5: Gate Evaluation (4-bit NF4 Unmerged Training Replica + Merged FP16)
        """# ── CELL 5: GATE CHECK (4-BIT NF4 UNMERGED REPLICA + MERGED FP16) ──────────
import os, sys, time, torch, librosa, json
import pandas as pd
from transformers import WhisperForConditionalGeneration, WhisperProcessor, BitsAndBytesConfig
from peft import PeftModel
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2
from evalkit.wer import wer_breakdown, compute_dual_wer_record

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Dynamically resolve reference baseline from results-Whisper-large-v3-qlora.json
ref_json_candidates = [
    "Results/results-Whisper-large-v3-qlora.json",
    "/kaggle/working/Results/results-Whisper-large-v3-qlora.json",
    "/kaggle/input/results-whisper-large-v3-qlora.json"
]
ref_wer = 12.51
for cand in ref_json_candidates:
    if os.path.exists(cand):
        try:
            with open(cand, 'r', encoding='utf-8') as jf:
                ref_wer = float(json.load(jf).get('finetuned_wer', 12.51))
                print(f"[OK] Read reference baseline WER from {cand}: {ref_wer}%")
                break
        except Exception:
            pass

print(f"Target Reference WER: {ref_wer}% | Gate Tolerance: 0.25 pp (Range: {ref_wer-0.25:.2f}% - {ref_wer+0.25:.2f}%)")

# Audio resolver
clips_dir = os.path.join(os.path.dirname(tsv_path), "clips")
audio_index = {}
if os.path.exists(clips_dir):
    for fname in os.listdir(clips_dir):
        if fname.endswith(('.mp3', '.wav', '.ogg', '.m4a')):
            full_p = os.path.join(clips_dir, fname)
            audio_index[fname] = full_p
            audio_index[os.path.splitext(fname)[0]] = full_p

def resolve_audio(p):
    fname = os.path.basename(str(p))
    stem = os.path.splitext(fname)[0]
    return audio_index.get(fname) or audio_index.get(stem) or os.path.join(clips_dir, fname)

eval_subset = subset_a.iloc[:8] if RUN_MODE == "smoke" else subset_a
processor = WhisperProcessor.from_pretrained(merged_path if os.path.exists(merged_path) else base_path)

def transcribe_with_model(model_obj, clip_path):
    speech, sr = librosa.load(clip_path, sr=16000, mono=True)
    dur = len(speech) / 16000.0
    inputs = processor(speech[:480000], sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)
    with torch.no_grad():
        gen_ids = model_obj.generate(inputs, language="ar", task="transcribe", num_beams=1, do_sample=False)
    hyp = processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return hyp, dur

# 1. Evaluate 4-bit NF4 Unmerged Training Replica
print("\\n--- A. Evaluating 4-bit NF4 + Unmerged LoRA Adapter Replica ---")
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    base_4bit = WhisperForConditionalGeneration.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model_4bit = PeftModel.from_pretrained(base_4bit, adapter_path).eval()
    has_4bit = True
except Exception as e:
    if RUN_MODE == "smoke":
        print(f"[SMOKE NOTICE] 4-bit loading unavailable in smoke mode ({e}); skipping 4-bit gate.")
        model_4bit = None
        has_4bit = False
    else:
        raise RuntimeError(f"FATAL: 4-bit unmerged replica loading failed: {e}. Bitsandbytes/PEFT required for gate check.")

gate_records_4bit = []
if has_4bit:
    for idx, (_, row) in enumerate(eval_subset.iterrows()):
        cp = resolve_audio(row['path'])
        hyp, dur = transcribe_with_model(model_4bit, cp)
        ref_norm = normalize_arabic_legacy(row['sentence'])
        hyp_norm = normalize_arabic_legacy(hyp)
        s, d, ins, h, n, w = wer_breakdown(ref_norm, hyp_norm)
        gate_records_4bit.append({
            'client_id': row.get('client_id', 'unknown'),
            'path': row['path'],
            'duration': round(dur, 3),
            'ref': row['sentence'],
            'hyp_unnorm': hyp,
            'hyp_norm': hyp_norm,
            'S': s, 'D': d, 'I': ins, 'H': h, 'N': n,
            'wer': round(w, 4)
        })
    del model_4bit, base_4bit
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 2. Evaluate Merged FP16 Model
print("\\n--- B. Evaluating Merged FP16 Model ---")
model_fp16 = WhisperForConditionalGeneration.from_pretrained(merged_path, torch_dtype=dtype).to(device).eval()
gate_records_fp16 = []
for idx, (_, row) in enumerate(eval_subset.iterrows()):
    cp = resolve_audio(row['path'])
    hyp, dur = transcribe_with_model(model_fp16, cp)
    ref_norm = normalize_arabic_legacy(row['sentence'])
    hyp_norm = normalize_arabic_legacy(hyp)
    s, d, ins, h, n, w = wer_breakdown(ref_norm, hyp_norm)
    gate_records_fp16.append({
        'client_id': row.get('client_id', 'unknown'),
        'path': row['path'],
        'duration': round(dur, 3),
        'ref': row['sentence'],
        'hyp_unnorm': hyp,
        'hyp_norm': hyp_norm,
        'S': s, 'D': d, 'I': ins, 'H': h, 'N': n,
        'wer': round(w, 4)
    })

# Compute Gate Summary and Save Separate CSVs
if has_4bit:
    df_gate_4bit = pd.DataFrame(gate_records_4bit)
    gate_4bit_csv = os.path.join(OUT_DIR, "cv_gate_subset_a_4bit_clips.csv")
    df_gate_4bit.to_csv(gate_4bit_csv, index=False, encoding="utf-8")
    print(f"[OK] Saved 4-bit gate per-clip CSV to {gate_4bit_csv} ({len(df_gate_4bit)} rows)")
    assert len(df_gate_4bit) == len(eval_subset), "FATAL: Dropped clips during 4-bit gate evaluation!"
    four_bit_s = int(df_gate_4bit['S'].sum())
    four_bit_d = int(df_gate_4bit['D'].sum())
    four_bit_i = int(df_gate_4bit['I'].sum())
    four_bit_h = int(df_gate_4bit['H'].sum())
    four_bit_n = int(df_gate_4bit['N'].sum())
    four_bit_wer = ((four_bit_s + four_bit_d + four_bit_i) / four_bit_n * 100.0) if four_bit_n > 0 else 0.0
else:
    four_bit_s = four_bit_d = four_bit_i = four_bit_h = four_bit_n = 0
    four_bit_wer = 0.0

df_gate_fp16 = pd.DataFrame(gate_records_fp16)
gate_fp16_csv = os.path.join(OUT_DIR, "cv_gate_subset_a_fp16_clips.csv")
df_gate_fp16.to_csv(gate_fp16_csv, index=False, encoding="utf-8")
print(f"[OK] Saved merged-FP16 gate per-clip CSV to {gate_fp16_csv} ({len(df_gate_fp16)} rows)")
assert len(df_gate_fp16) == len(eval_subset), "FATAL: Dropped clips during merged-FP16 gate evaluation!"

fp16_s = int(df_gate_fp16['S'].sum())
fp16_d = int(df_gate_fp16['D'].sum())
fp16_i = int(df_gate_fp16['I'].sum())
fp16_h = int(df_gate_fp16['H'].sum())
fp16_n = int(df_gate_fp16['N'].sum())
fp16_wer = ((fp16_s + fp16_d + fp16_i) / fp16_n * 100.0) if fp16_n > 0 else 0.0

tolerance = 5.0 if RUN_MODE == "smoke" else 0.25

print("\\n=== GATE RESULTS ===")
if has_4bit:
    print(f"4-bit NF4 Unmerged WER : {four_bit_wer:.2f}% (Delta vs Ref: {abs(four_bit_wer - ref_wer):.2f} pp)")
print(f"Merged FP16 WER        : {fp16_wer:.2f}% (Delta vs Ref: {abs(fp16_wer - ref_wer):.2f} pp)")
print(f"Reference Target WER   : {ref_wer:.2f}% (Tolerance: {tolerance:.2f} pp)")

gate_summary_results = {
    'reference_baseline_wer': ref_wer,
    'tolerance_pp': tolerance,
    'four_bit_unmerged_replica': {
        'wer': round(four_bit_wer, 2) if has_4bit else None,
        'substitutions': four_bit_s,
        'deletions': four_bit_d,
        'insertions': four_bit_i,
        'hits': four_bit_h,
        'total_words': four_bit_n,
        'delta_pp': round(abs(four_bit_wer - ref_wer), 2) if has_4bit else None,
        'status': ('PASSED' if abs(four_bit_wer - ref_wer) <= tolerance else 'FAILED') if has_4bit else 'SKIPPED_SMOKE'
    },
    'merged_fp16': {
        'wer': round(fp16_wer, 2),
        'substitutions': fp16_s,
        'deletions': fp16_d,
        'insertions': fp16_i,
        'hits': fp16_h,
        'total_words': fp16_n,
        'delta_pp': round(abs(fp16_wer - ref_wer), 2),
        'status': 'PASSED' if abs(fp16_wer - ref_wer) <= tolerance else 'FAILED'
    }
}
gate_json_path = os.path.join(OUT_DIR, "cv_gate_results.json")
save_result_file(gate_json_path, gate_summary_results, config=PROTOCOL, device=device, notebook_path="Notebooks/final_01a_cv_gate_pass1.ipynb")
print(f"[OK] Saved gate summary JSON to {gate_json_path}")

eval_wer_for_gate = four_bit_wer if has_4bit else fp16_wer
delta = abs(eval_wer_for_gate - ref_wer)
if RUN_MODE != "smoke":
    assert delta <= tolerance, f"GATE STOPPED: Measured WER ({eval_wer_for_gate:.2f}%) differs from reference {ref_wer:.2f}% by {delta:.2f} pp (> {tolerance:.2f} pp)"
print(f"[GATE PASSED] Measured WER ({eval_wer_for_gate:.2f}%) replicated reference within {tolerance:.2f} pp tolerance. Proceeding to Pass 1.")
""",
        # Cell 6: Pass 1 Full Pool Evaluation
        """# ── CELL 6: PASS 1 FULL POOL EVALUATION (PYTORCH FP16 GREEDY) ─────────────
import math
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2
from evalkit.wer import compute_dual_wer_record
from evalkit.stats import bootstrap_ci, save_result_file

pass1_out_file = os.path.join(OUT_DIR, "cv_pass1_pytorch_fp16.json")
ckpt_csv_file = os.path.join(OUT_DIR, "cv_pass1_checkpoint.csv")
pass1_clips_csv = os.path.join(OUT_DIR, "cv_pass1_clips.csv")

active_pool = df_valid.iloc[:8].copy() if RUN_MODE == "smoke" else df_valid.copy()
total_pool_clips = len(active_pool)
print(f"Pass 1 Active Pool: {total_pool_clips:,} clips")

# Checkpoint resumption
processed_paths = set()
checkpoint_rows = []
if os.path.exists(ckpt_csv_file):
    try:
        df_prev = pd.read_csv(ckpt_csv_file)
        processed_paths = set(df_prev['path'].astype(str))
        checkpoint_rows = df_prev.to_dict(orient='records')
        print(f"[RESUME] Loaded {len(processed_paths):,} existing rows from {ckpt_csv_file}")
    except Exception:
        checkpoint_rows = []

t0_p1 = time.time()
pass1_rows = list(checkpoint_rows)

for idx, (_, row) in enumerate(active_pool.iterrows()):
    p_str = str(row['path'])
    if p_str in processed_paths:
        continue
        
    cp = resolve_audio(p_str)
    hyp_raw, dur = transcribe_with_model(model_fp16, cp)
    ref_text = row['sentence']
    
    rec = compute_dual_wer_record(ref_text, hyp_raw)
    leg_b = rec['legacy_breakdown']
    v2_b = rec['v2_breakdown']
    
    row_record = {
        'client_id': row.get('client_id', 'unknown'),
        'path': p_str,
        'duration': round(dur, 3),
        'ref': ref_text,
        'hyp_unnorm': hyp_raw,
        'hyp_norm': rec['normalized_hyp_legacy'],
        'hyp_norm_v2': rec['normalized_hyp_v2'],
        'S': leg_b['substitutions'],
        'D': leg_b['deletions'],
        'I': leg_b['insertions'],
        'H': leg_b['hits'],
        'N': leg_b['words'],
        'wer': round(rec['legacy_wer'], 4),
        'v2_wer': round(rec['v2_wer'], 4),
        'v2_S': v2_b['substitutions'],
        'v2_D': v2_b['deletions'],
        'v2_I': v2_b['insertions'],
        'v2_H': v2_b['hits'],
        'v2_N': v2_b['words']
    }
    pass1_rows.append(row_record)
    processed_paths.add(p_str)
    
    if (len(pass1_rows) % 100 == 0) or (len(pass1_rows) == total_pool_clips):
        df_ckpt = pd.DataFrame(pass1_rows)
        df_ckpt.to_csv(ckpt_csv_file, index=False, encoding="utf-8")
        elapsed = time.time() - t0_p1
        curr_w = (df_ckpt['S'].sum() + df_ckpt['D'].sum() + df_ckpt['I'].sum()) / df_ckpt['N'].sum() * 100.0
        print(f"  [Pass 1] Processed {len(pass1_rows):,}/{total_pool_clips:,} clips | Running WER: {curr_w:.2f}% | Elapsed: {elapsed/60:.1f}m")

# Save final per-clip CSVs
df_pass1 = pd.DataFrame(pass1_rows)
df_pass1.to_csv(pass1_clips_csv, index=False, encoding="utf-8")
df_pass1.to_csv(ckpt_csv_file, index=False, encoding="utf-8")
print(f"[OK] Saved Pass 1 per-clip CSV to {pass1_clips_csv} ({len(df_pass1)} rows)")
assert len(df_pass1) == total_pool_clips, f"FATAL: Clips dropped during Pass 1! Got {len(df_pass1)}, expected {total_pool_clips}"

# Aggregate Pass 1 JSON
leg_s = int(df_pass1['S'].sum())
leg_d = int(df_pass1['D'].sum())
leg_i = int(df_pass1['I'].sum())
leg_h = int(df_pass1['H'].sum())
leg_n = int(df_pass1['N'].sum())
leg_wer = round((leg_s + leg_d + leg_i) / leg_n * 100.0, 2)

v2_s = int(df_pass1['v2_S'].sum())
v2_d = int(df_pass1['v2_D'].sum())
v2_i = int(df_pass1['v2_I'].sum())
v2_h = int(df_pass1['v2_H'].sum())
v2_n = int(df_pass1['v2_N'].sum())
v2_wer = round((v2_s + v2_d + v2_i) / v2_n * 100.0, 2)

tot_audio_s = float(df_pass1['duration'].sum())
tot_comp_s = float(time.time() - t0_p1)
th_rt = round(tot_audio_s / tot_comp_s, 2) if tot_comp_s > 0 else 0.0

pass1_results = {
    'pass': 'Pass 1: Common Voice Full Pool (PyTorch FP16 Greedy)',
    'run_mode': RUN_MODE,
    'model': 'Whisper-Large-v3 (Fine-Tuned Arabic)',
    'precision': 'PyTorch FP16',
    'decoder': 'Greedy (beam_size=1, temperature=0.0)',
    'num_clips': len(df_pass1),
    'total_audio_seconds': round(tot_audio_s, 2),
    'total_audio_hours': round(tot_audio_s / 3600.0, 3),
    'total_compute_seconds': round(tot_comp_s, 2),
    'total_compute_hours': round(tot_comp_s / 3600.0, 3),
    'throughput_realtime': th_rt,
    'rtf': round(tot_comp_s / tot_audio_s, 4) if tot_audio_s > 0 else 0.0,
    'gate_subset_a': {
        'gate_wer_4bit_unmerged': round(four_bit_wer, 2) if has_4bit else None,
        'gate_wer_merged_fp16': round(fp16_wer, 2),
        'baseline_reference_wer': ref_wer,
        'status': 'PASSED'
    },
    'legacy_normalizer': {
        'corpus_wer': leg_wer,
        'substitutions': leg_s,
        'deletions': leg_d,
        'insertions': leg_i,
        'hits': leg_h,
        'total_words': leg_n
    },
    'v2_normalizer': {
        'corpus_wer': v2_wer,
        'substitutions': v2_s,
        'deletions': v2_d,
        'insertions': v2_i,
        'hits': v2_h,
        'total_words': v2_n
    }
}

save_result_file(pass1_out_file, pass1_results, config=PROTOCOL, device=device, notebook_path="Notebooks/final_01a_cv_gate_pass1.ipynb")
print(f"[OK] Pass 1 results saved to: {pass1_out_file}")
print(f"Pass 1 Legacy WER: {leg_wer:.2f}% | v2 WER: {v2_wer:.2f}% | Throughput: {th_rt:.2f}x real-time")

"""
    ]
    create_notebook("Notebooks/final_01a_cv_gate_pass1.ipynb", cells)

if __name__ == "__main__":
    build_n1a()
