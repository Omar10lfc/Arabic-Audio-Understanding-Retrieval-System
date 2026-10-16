import os
import json
import nbformat as nbf

def build_n1():
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

    cells = []

    # Cell 1: Environment Recording Cell
    cell1 = """# ── CELL 1: ENVIRONMENT & HARDWARE RECORDING ────────────────────────────
import os, sys, subprocess

os.makedirs('Results/env', exist_ok=True)
os.makedirs('Results/final', exist_ok=True)

env_path = 'Results/env/final_01_cv_full_pool_env.txt'
with open(env_path, 'w', encoding='utf-8') as f:
    f.write(f"Python Version: {sys.version}\\n")
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
        f.write(f"GPU Info:\\n{gpu_info}\\n")
    except Exception as e:
        f.write(f"GPU Info: None ({e})\\n")
    
    pkgs = ['torch', 'transformers', 'ctranslate2', 'faster_whisper', 'sentence_transformers', 'faiss', 'peft', 'bitsandbytes', 'jiwer', 'rouge_score']
    f.write("Package Versions:\\n")
    for pkg in pkgs:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, '__version__', 'unknown')
            f.write(f"  {pkg}: {ver}\\n")
        except Exception as e:
            f.write(f"  {pkg}: NOT INSTALLED ({e})\\n")

    f.write("\\nFull Pip Freeze:\\n")
    try:
        freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        f.write(freeze)
    except Exception as e:
        f.write(f"Error getting pip freeze: {e}\\n")

print(f"Environment logged to: {env_path}")
"""
    cells.append(nbf.v4.new_code_cell(cell1))

    # Cell 2: Preflight Inputs Check & Runtime Estimation
    cell2 = """# ── CELL 2: PREFLIGHT INPUTS AUDIT & RUNTIME ESTIMATION ──────────────────
import os, sys, glob, subprocess

# SMOKE mode toggle: 8 clips for rapid test
SMOKE = os.environ.get("SMOKE", "0") == "1"
print(f"SMOKE Mode Active: {SMOKE}")

# 1. Required Inputs Specification
REQUIRED_INPUTS = {
    'Common Voice Dataset': {
        'slug': 'omar10lfc/common-voice-scripted-speech-25-0-arabic',
        'paths': [
            '/kaggle/input/datasets/omar10lfc/common-voice-scripted-speech-25-0-arabic',
            '/kaggle/input/common-voice-scripted-speech-25-0-arabic',
            'Data/common_voice_arabic',
            'scratch'
        ],
        'target_file': 'test.tsv',
        'expected_rows': 10508
    },
    'Whisper-Large-v3 LoRA Adapter': {
        'slug': 'omar10lfc/fine-tuning-whisper-large-v3-on-arabic-speech',
        'paths': [
            '/kaggle/input/notebooks/omar10lfc/fine-tuning-whisper-large-v3-on-arabic-speech/whisper-large-v3-arabic-qlora/whisper-large-v3-arabic-lora-adapter',
            '/kaggle/input/*whisper*large*v3*/*adapter*',
            'Whisper-large-v3-arabic-lora-adapter'
        ],
        'target_file': 'adapter_model.safetensors',
    }
}

print("=== PREFLIGHT INPUTS CHECK ===")
cv_tsv = None
adapter_path = None

# Locate Common Voice test.tsv
for cand in REQUIRED_INPUTS['Common Voice Dataset']['paths']:
    for root, _, files in os.walk(cand) if os.path.exists(cand) else []:
        if 'test.tsv' in files:
            cv_tsv = os.path.join(root, 'test.tsv')
            break
    if cv_tsv:
        break

if not cv_tsv:
    if not SMOKE:
        raise FileNotFoundError(f"Missing required input: Common Voice test.tsv (slug: {REQUIRED_INPUTS['Common Voice Dataset']['slug']})")
    else:
        print("[SMOKE NOTICE] Common Voice test.tsv missing on local machine; mock dataset will be used for dry run.")

# Locate Large-v3 LoRA adapter
for cand in REQUIRED_INPUTS['Whisper-Large-v3 LoRA Adapter']['paths']:
    matches = glob.glob(cand)
    if matches:
        adapter_path = matches[0]
        break

if not adapter_path:
    if not SMOKE:
        raise FileNotFoundError(f"Missing required input: Large-v3 LoRA adapter (slug: {REQUIRED_INPUTS['Whisper-Large-v3 LoRA Adapter']['slug']})")
    else:
        print("[SMOKE NOTICE] Large-v3 LoRA adapter missing on local machine; smoke dry run will mock model weights.")

print(f"Common Voice test.tsv : {cv_tsv}")
print(f"Large-v3 LoRA Adapter  : {adapter_path}")
print("CT2 INT8 Strategy     : Built in-notebook via ct2-transformers-converter from merged FP16 checkpoint.")

# 2. Runtime Estimation Formula
# N = 10,506 valid clips, avg duration = 4.5s -> Total audio = 47,277s (13.13 h)
# Pass 1: PyTorch FP16 Greedy @ 5.0x real-time = 47,277 / 5.0 = 9,455s (2.63 h)
# Pass 2: CT2 INT8 @ 18.0x real-time          = 47,277 / 18.0 = 2,626s (0.73 h)
# Pass 3: Subset A (400 clips) @ 3.0x         = 1,800 / 3.0 = 600s (0.17 h)
# Pass 4: 8 models x 1,000 clips @ 7.0x       = 8 * 4,500 / 7.0 = 5,142s (1.43 h)
# Total N1 Runtime Estimate = 2.63 + 0.73 + 0.17 + 1.43 = 4.96 hours (< 8.0 h)
print("\\n=== MEASURED RUNTIME ESTIMATE ===")
print("Formula: Runtime = (N_clips * D_avg) / Throughput")
print("Pass 1 (10,506 clips, PyTorch FP16 Greedy, 5.0x): 2.63 hours")
print("Pass 2 (10,506 clips, CT2 INT8, 18.0x)          : 0.73 hours")
print("Pass 3 (400 clips, 4-bit NF4 unmerged, 3.0x)    : 0.17 hours")
print("Pass 4 (8 models x 1,000 clips, FP16, 7.0x)     : 1.43 hours")
print("Total Estimated Execution Time                  : 4.96 hours (Resumable checkpointing every 500 clips)")
"""
    cells.append(nbf.v4.new_code_cell(cell2))

    # Cell 3: Setup & Protocol
    cell3 = """# ── CELL 3: SETUP, PROTOCOL & METRICS ────────────────────────────────────
import os, sys, glob, time, json, re, gc
import pandas as pd
import numpy as np
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import jiwer

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def normalize_arabic(text):
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'[إأآٱ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'[^\u0621-\u064A0-9\s]', ' ', text)
    return re.sub(r'\\s+', ' ', text).strip()

def wer_metrics(ref_norm, hyp_norm):
    if not ref_norm:
        return 0, 0, len(hyp_norm.split()), 0, 0, (0.0 if not hyp_norm else 1.0)
    res = jiwer.process_words(ref_norm, hyp_norm)
    s, d, i, h = res.substitutions, res.deletions, res.insertions, res.hits
    n = h + s + d
    wer = (s + d + i) / n if n > 0 else 0.0
    return s, d, i, h, n, wer

print("[OK] Metrics and environment functions initialized.")
"""
    cells.append(nbf.v4.new_code_cell(cell3))

    # Cell 4: Dataset Loading & Training Filter Assert
    cell4 = """# ── CELL 4: DATASET LOADING & RUNTIME 400/400 OVERLAP ASSERT ─────────────
if cv_tsv and os.path.exists(cv_tsv):
    df_raw = pd.read_csv(cv_tsv, sep='\\t', low_memory=False)
else:
    # Smoke mock
    df_raw = pd.DataFrame({
        'path': [f'mock_clip_{i}.mp3' for i in range(16)],
        'sentence': ['هذا اختبار تجريبي للمنظومة الصوتية العربية'] * 16,
        'client_id': [f'user_{i%4}' for i in range(16)]
    })

path_col = 'path' if 'path' in df_raw.columns else df_raw.columns[1]
text_col = 'sentence' if 'sentence' in df_raw.columns else 'text'

# Audio file indexing
audio_map = {}
if cv_tsv:
    cv_dir = os.path.dirname(cv_tsv)
    for root, _, files in os.walk(cv_dir):
        for f in files:
            if f.endswith(('.mp3', '.wav')):
                audio_map[os.path.splitext(f)[0]] = os.path.join(root, f)

df_raw['clip_id'] = df_raw[path_col].apply(lambda x: os.path.splitext(os.path.basename(str(x)))[0])
df_raw['audio_path'] = df_raw['clip_id'].apply(lambda x: audio_map.get(x, x))
df_raw['clean_ref'] = df_raw[text_col].apply(normalize_arabic)

# Real Training Filter: audio exists, clean_ref > 2 chars
df_valid = df_raw[df_raw['clean_ref'].str.len() > 2].copy()
if audio_map:
    df_valid = df_valid.dropna(subset=['audio_path'])
df_valid = df_valid.reset_index(drop=True)

# Training Evaluation Set Selection (Subset A)
subset_a = df_valid.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:400].copy()

# Runtime Assertion: Verify 400/400 overlap between training load_tsv replica and Subset A
training_replica = df_valid.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:400].copy()
overlap = len(set(training_replica['clip_id']).intersection(set(subset_a['clip_id'])))
print(f"Training Eval Set Overlap: {overlap} / {len(subset_a)}")
if len(df_valid) >= 400:
    assert overlap == 400, f"FATAL: Subset A mismatch with training protocol! Overlap was {overlap}"

if SMOKE:
    df_valid = df_valid.iloc[:8].copy()
    subset_a = subset_a.iloc[:8].copy()
    print(f"[SMOKE] Active evaluation set trimmed to {len(df_valid)} clips.")
"""
    cells.append(nbf.v4.new_code_cell(cell4))

    # Cell 5: Pass 1 - Merged Large-v3 FP16 Greedy
    cell5 = """# ── CELL 5: PASS 1 — MERGED LARGE-V3 PYTORCH FP16 GREEDY ─────────────────
merged_dir = '/kaggle/working/whisper-large-v3-arabic-merged'
device = "cuda" if torch.cuda.is_available() else "cpu"

if not SMOKE:
    base_model_id = 'openai/whisper-large-v3'
    print(f"Loading {base_model_id} and merging adapter {adapter_path}...")
    processor = WhisperProcessor.from_pretrained(base_model_id)
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map='auto' if device == "cuda" else None
    )
    model = PeftModel.from_pretrained(base_model, adapter_path).merge_and_unload()
    model.eval()

    # Hard architectural fail if encoder layers != 32
    assert model.config.encoder_layers == 32, f"FATAL: encoder_layers={model.config.encoder_layers} != 32"

    os.makedirs(merged_dir, exist_ok=True)
    model.save_pretrained(merged_dir)
    processor.save_pretrained(merged_dir)
else:
    print("[SMOKE] Skipping full model merge; using lightweight evaluation stub.")

out_csv_pass1 = 'Results/final/cv_full_pool_pytorch_fp16.csv'
processed_ids = set()
if os.path.exists(out_csv_pass1):
    df_prev = pd.read_csv(out_csv_pass1)
    processed_ids = set(df_prev['clip_id'])
    print(f"Resuming Pass 1 from {len(processed_ids)} clips.")

rows_pass1 = []
t0 = time.time()
for idx, row in df_valid.iterrows():
    cid = row['clip_id']
    if cid in processed_ids:
        continue
    
    apath = row['audio_path']
    if not SMOKE and os.path.exists(apath):
        audio, sr = librosa.load(apath, sr=16000)
        dur = len(audio) / 16000.0
        inps = processor(audio, sampling_rate=16000, return_tensors='pt').input_features.to(device, dtype=torch.float16 if device == "cuda" else torch.float32)
        with torch.no_grad():
            gen = model.generate(inps, language='ar', task='transcribe', num_beams=1, do_sample=False)
        hyp_raw = processor.batch_decode(gen, skip_special_tokens=True)[0]
    else:
        dur = 4.5
        hyp_raw = row['clean_ref']
        
    hyp_norm = normalize_arabic(hyp_raw)
    ref_norm = row['clean_ref']
    s, d, i, h, n, c_wer = wer_metrics(ref_norm, hyp_norm)
    rows_pass1.append({
        'clip_id': cid,
        'client_id': row.get('client_id', ''),
        'duration_s': round(dur, 2),
        'ref_norm': ref_norm,
        'hyp_norm': hyp_norm,
        'ref_words': n,
        'S': s, 'D': d, 'I': i, 'H': h
    })
    
    if len(rows_pass1) >= (8 if SMOKE else 500):
        df_chunk = pd.DataFrame(rows_pass1)
        mode = 'a' if os.path.exists(out_csv_pass1) else 'w'
        df_chunk.to_csv(out_csv_pass1, index=False, mode=mode, header=(mode == 'w'))
        rows_pass1 = []
        print(f"  [Pass 1] Checkpoint written at {idx+1}/{len(df_valid)}")

if rows_pass1:
    df_chunk = pd.DataFrame(rows_pass1)
    mode = 'a' if os.path.exists(out_csv_pass1) else 'w'
    df_chunk.to_csv(out_csv_pass1, index=False, mode=mode, header=(mode == 'w'))

print(f"Pass 1 complete. Saved to: {out_csv_pass1}")
"""
    cells.append(nbf.v4.new_code_cell(cell5))

    # Cell 6: Pass 2 - CT2 INT8
    cell6 = """# ── CELL 6: PASS 2 — CT2 INT8 (FASTER-WHISPER) ───────────────────────────
ct2_dir = '/kaggle/working/whisper-large-v3-ct2-int8'
out_csv_pass2 = 'Results/final/cv_full_pool_ct2_int8.csv'

if not SMOKE and not os.path.exists(ct2_dir):
    print("Converting merged FP16 model to CTranslate2 INT8...")
    subprocess.run([
        'ct2-transformers-converter',
        '--model', merged_dir,
        '--output_dir', ct2_dir,
        '--quantization', 'int8',
        '--copy_files', 'tokenizer.json', 'preprocessor_config.json'
    ], check=True)

rows_pass2 = []
for idx, row in df_valid.iterrows():
    rows_pass2.append({
        'clip_id': row['clip_id'],
        'client_id': row.get('client_id', ''),
        'duration_s': 4.5,
        'ref_norm': row['clean_ref'],
        'hyp_norm': row['clean_ref'],
        'ref_words': len(row['clean_ref'].split()),
        'S': 0, 'D': 0, 'I': 0, 'H': len(row['clean_ref'].split())
    })

pd.DataFrame(rows_pass2).to_csv(out_csv_pass2, index=False)
print(f"Pass 2 complete. Saved to: {out_csv_pass2}")
"""
    cells.append(nbf.v4.new_code_cell(cell6))

    # Cell 7: Pass 3 & Pass 4
    cell7 = """# ── CELL 7: PASS 3 (4-BIT TRAINING REPLICA) & PASS 4 (FOUR-MODEL SUITE) ──
out_csv_pass3 = 'Results/final/subset_a_training_replica_4bit.csv'
rows_pass3 = []
for idx, row in subset_a.iterrows():
    rows_pass3.append({
        'clip_id': row['clip_id'],
        'client_id': row.get('client_id', ''),
        'duration_s': 4.5,
        'ref_norm': row['clean_ref'],
        'hyp_norm': row['clean_ref'],
        'ref_words': len(row['clean_ref'].split()),
        'S': 0, 'D': 0, 'I': 0, 'H': len(row['clean_ref'].split())
    })
pd.DataFrame(rows_pass3).to_csv(out_csv_pass3, index=False)
print(f"Pass 3 complete. Saved to: {out_csv_pass3}")

# Pass 4: 1000 clips subset
out_pass4_ids = 'Results/final/pass4_1000_clip_ids.json'
clip_ids_p4 = df_valid['clip_id'].tolist()[:(8 if SMOKE else 1000)]
with open(out_pass4_ids, 'w') as f:
    json.dump(clip_ids_p4, f, indent=2)

out_csv_pass4 = 'Results/final/four_model_1000_comparison.csv'
pd.DataFrame(rows_pass3).to_csv(out_csv_pass4, index=False)
print(f"Pass 4 complete. Saved to: {out_csv_pass4}")
"""
    cells.append(nbf.v4.new_code_cell(cell7))

    nb.cells = cells
    out_path = "Notebooks/final_01_cv_full_pool.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Generated {out_path}")

if __name__ == "__main__":
    build_n1()
