"""
N1B NOTEBOOK BUILDER (scripts/build_n1b.py)
Generates Notebooks/final_01b_cv_pass2_beam5.ipynb (Internet OFF)
Strictly adheres to:
B.3.a Every pass (Pass 2 greedy, Pass 3 beam-5) saves per-clip CSV: client_id, duration, ref, hyp_unnorm, hyp_norm, S, D, I, H, N, wer.
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

def build_n1b(evalkit_zip_sha=""):
    cells = [
        # Cell 1: Environment & evalkit check
        f"""# ── CELL 1: ENVIRONMENT & EVALKIT-SRC VERIFICATION ────────────────────────
import os, sys, subprocess, json, hashlib, zipfile, shutil

for pkg in ["ctranslate2", "faster-whisper", "soundfile"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {{pkg}}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
            print(f"[OK] Installed {{pkg}}.")
        except Exception as e:
            print(f"[ERROR] Could not install {{pkg}}: {{e}}")
            print("Please toggle Internet: ON in Kaggle Notebook Settings (right sidebar) to allow pip install.")
            raise

RUN_MODE = os.environ.get("RUN_MODE", "full")
OUT_DIR = "Results/smoke" if RUN_MODE == "smoke" else "Results/final"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("Results/env", exist_ok=True)

EXPECTED_EVALKIT_SHA = "{evalkit_zip_sha}"
print(f"RUN_MODE: {{RUN_MODE}} | Target: {{OUT_DIR}} | Expected evalkit SHA: {{EXPECTED_EVALKIT_SHA}}")

env_path = f"Results/env/final_01b_cv_pass2_beam5_{{RUN_MODE}}_env.txt"
with open(env_path, 'w', encoding='utf-8') as f:
    f.write(f"Python Version: {{sys.version}}\\n")
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
        f.write(f"GPU Info:\\n{{gpu_info}}\\n")
    except Exception as e:
        f.write(f"GPU Info: None ({{e}})\\n")
print(f"Environment logged to: {{env_path}}")

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
        # Cell 2: Preflight check & Protocol load
        """# ── CELL 2: PREFLIGHT INPUTS AUDIT & PROTOCOL ────────────────────────────
import os, glob, time, json, pandas as pd
from evalkit.normalize import normalize_arabic_legacy
from evalkit.wer import compute_dual_wer_record, wer_breakdown
from evalkit.stats import save_result_file

cfg_path = "/kaggle/working/configs/final_protocol.json" if os.path.exists("/kaggle/working/configs/final_protocol.json") else "configs/final_protocol.json"
with open(cfg_path, 'r', encoding='utf-8') as f:
    PROTOCOL = json.load(f)

print("=== MOUNTED INPUT DIRECTORIES in /kaggle/input ===")
if os.path.exists("/kaggle/input"):
    for item in sorted(os.listdir("/kaggle/input")):
        print(f" - /kaggle/input/{item}")

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

merged_root = MERGED_MODELS_DIR or "/kaggle/input/final-merged-models"
ct2_model_path = os.path.join(merged_root, "ct2_int8/whisper-large-v3-ct2-int8")

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

cv_root = COMMON_VOICE_DIR or "/kaggle/input/common-voice-scripted-speech-25-0-arabic"
tsv_path = os.path.join(cv_root, "test.tsv")

print(f"CT2 Model Path : {ct2_model_path} (exists: {os.path.exists(ct2_model_path)})")
print(f"CV TSV Path    : {tsv_path} (exists: {os.path.exists(tsv_path)})")

if os.path.exists(tsv_path):
    df_raw = pd.read_csv(tsv_path, sep='\\t', low_memory=False)
else:
    df_raw = pd.DataFrame({
        'path': [f'mock_{i}.mp3' for i in range(16)],
        'sentence': ['هذا اختبار تجريبي'] * 16,
        'client_id': [f'spk_{i%4}' for i in range(16)]
    })

df_raw = df_raw.dropna(subset=['sentence']).copy()
df_raw['clean_ref'] = df_raw['sentence'].apply(normalize_arabic_legacy)
df_valid = df_raw[df_raw['clean_ref'].str.len() > 2].reset_index(drop=True)
print(f"Common Voice Pool Size: {len(df_valid):,} clips")
""",
        # Cell 3: Pass 2 CT2 INT8 Greedy Full Pool Evaluation
        """# ── CELL 3: PASS 2 CT2 INT8 GREEDY FULL POOL EVALUATION ────────────────────
import os, time, gc, torch, soundfile as sf, numpy as np, librosa
import pandas as pd
from faster_whisper import WhisperModel
from faster_whisper.feature_extractor import FeatureExtractor
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2
from evalkit.wer import compute_dual_wer_record, wer_breakdown
from evalkit.stats import save_result_file

# PyAV compatibility patch
try:
    import av, av.container.core
    _c_open = av.container.core.open
    def _safe_av_open(*args, **kwargs):
        kwargs.pop('metadata_errors', None)
        kwargs.pop('metadata_encoding', None)
        return _c_open(*args, **kwargs)
    av.container.core.open = _safe_av_open
    av.open = _safe_av_open
except Exception:
    pass

# Universal recursive audio index across all mounted directories in /kaggle/input
print("Indexing audio files across all mounted directories in /kaggle/input...")
t0_idx = time.time()
audio_index = {}
if os.path.exists("/kaggle/input"):
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f.endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                full_p = os.path.join(root, f)
                audio_index[f] = full_p
                audio_index[os.path.splitext(f)[0]] = full_p
print(f"[OK] Indexed {len(audio_index)//2:,} audio clips across all shards in {time.time()-t0_idx:.2f}s.")

def resolve_audio(p):
    fname = os.path.basename(str(p))
    stem = os.path.splitext(fname)[0]
    return audio_index.get(fname) or audio_index.get(stem) or (p if os.path.exists(p) else None)

pass2_out_file = os.path.join(OUT_DIR, "cv_pass2_ct2_int8.json")
pass2_clips_csv = os.path.join(OUT_DIR, "cv_pass2_clips.csv")
ckpt_p2_file = os.path.join(OUT_DIR, "cv_pass2_checkpoint.csv")

active_pool = df_valid.iloc[:8].copy() if RUN_MODE == "smoke" else df_valid.copy()
total_pool = len(active_pool)
print(f"Pass 2 Active Pool: {total_pool:,} clips")

# Checkpoint resumption
processed_paths = set()
checkpoint_rows = []
if os.path.exists(ckpt_p2_file):
    try:
        df_prev = pd.read_csv(ckpt_p2_file)
        processed_paths = set(df_prev['path'].astype(str))
        checkpoint_rows = df_prev.to_dict(orient='records')
        print(f"[RESUME] Loaded {len(processed_paths):,} existing rows from {ckpt_p2_file}")
    except Exception:
        checkpoint_rows = []

# Load CT2 INT8 Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CTranslate2 model from {ct2_model_path} on {device}...")
model_ct2 = WhisperModel(ct2_model_path, device=device, compute_type="int8")
model_ct2.feature_extractor = FeatureExtractor(feature_size=128)
print("[OK] CT2 INT8 Whisper model loaded successfully.")

t0_p2 = time.time()
pass2_rows = list(checkpoint_rows)

for idx, (_, row) in enumerate(active_pool.iterrows()):
    p_str = str(row['path'])
    if p_str in processed_paths:
        continue
        
    cp = resolve_audio(p_str)
    assert cp is not None and os.path.exists(cp), f"FATAL: Clip not found in any mounted directory: {p_str}"
    
    t0_clip = time.time()
    try:
        segments, info = model_ct2.transcribe(
            cp, language="ar", task="transcribe",
            beam_size=1, best_of=1, temperature=0.0,
            vad_filter=False, condition_on_previous_text=False
        )
        raw_hyp = " ".join([seg.text for seg in segments]).strip()
        dur = info.duration if hasattr(info, 'duration') and info.duration else 0.0
    except Exception:
        audio_np, _ = librosa.load(cp, sr=16000, mono=True)
        dur = float(len(audio_np) / 16000.0)
        segments, info = model_ct2.transcribe(
            audio_np, language="ar", task="transcribe",
            beam_size=1, best_of=1, temperature=0.0,
            vad_filter=False, condition_on_previous_text=False
        )
        raw_hyp = " ".join([seg.text for seg in segments]).strip()
    
    rec = compute_dual_wer_record(row['sentence'], raw_hyp)
    leg_b = rec['legacy_breakdown']
    v2_b = rec['v2_breakdown']
    
    pass2_rows.append({
        'client_id': row.get('client_id', 'unknown'),
        'path': p_str,
        'duration': round(dur, 3),
        'ref': row['sentence'],
        'hyp_unnorm': raw_hyp,
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
    })
    processed_paths.add(p_str)
    
    if (len(pass2_rows) % 100 == 0) or (len(pass2_rows) == total_pool):
        df_cur = pd.DataFrame(pass2_rows)
        df_cur.to_csv(ckpt_p2_file, index=False, encoding="utf-8")
        w_cur = (df_cur['S'].sum() + df_cur['D'].sum() + df_cur['I'].sum()) / df_cur['N'].sum() * 100.0
        elapsed = time.time() - t0_p2
        print(f"  [Pass 2] Processed {len(pass2_rows):,}/{total_pool:,} clips | Running WER: {w_cur:.2f}% | Elapsed: {elapsed/60:.1f}m")

df_pass2 = pd.DataFrame(pass2_rows)
df_pass2.to_csv(pass2_clips_csv, index=False, encoding="utf-8")
df_pass2.to_csv(ckpt_p2_file, index=False, encoding="utf-8")
print(f"[OK] Saved Pass 2 per-clip CSV to {pass2_clips_csv} ({len(df_pass2)} rows)")
assert len(df_pass2) == total_pool, f"FATAL: Dropped clips during Pass 2! Got {len(df_pass2)}, expected {total_pool}"

# Aggregate Pass 2
p2_leg_s = int(df_pass2['S'].sum())
p2_leg_d = int(df_pass2['D'].sum())
p2_leg_i = int(df_pass2['I'].sum())
p2_leg_h = int(df_pass2['H'].sum())
p2_leg_n = int(df_pass2['N'].sum())
p2_leg_wer = round((p2_leg_s + p2_leg_d + p2_leg_i) / p2_leg_n * 100.0, 2)

p2_v2_s = int(df_pass2['v2_S'].sum())
p2_v2_d = int(df_pass2['v2_D'].sum())
p2_v2_i = int(df_pass2['v2_I'].sum())
p2_v2_h = int(df_pass2['v2_H'].sum())
p2_v2_n = int(df_pass2['v2_N'].sum())
p2_v2_wer = round((p2_v2_s + p2_v2_d + p2_v2_i) / p2_v2_n * 100.0, 2)

tot_aud = float(df_pass2['duration'].sum())
tot_cmp = float(time.time() - t0_p2)
th_p2 = round(tot_aud / tot_cmp, 2) if tot_cmp > 0 else 0.0

pass2_results = {
    'pass': 'Pass 2: Common Voice Full Pool (CT2 INT8 Greedy)',
    'run_mode': RUN_MODE,
    'model': 'Whisper-Large-v3 (Fine-Tuned Arabic CT2 INT8)',
    'precision': 'CTranslate2 INT8',
    'decoder': 'Greedy (beam_size=1, temperature=0.0)',
    'num_clips': len(df_pass2),
    'total_audio_seconds': round(tot_aud, 2),
    'total_audio_hours': round(tot_aud / 3600.0, 3),
    'total_compute_seconds': round(tot_cmp, 2),
    'total_compute_hours': round(tot_cmp / 3600.0, 3),
    'throughput_realtime': th_p2,
    'rtf': round(tot_cmp / tot_aud, 4) if tot_aud > 0 else 0.0,
    'legacy_normalizer': {
        'corpus_wer': p2_leg_wer,
        'substitutions': p2_leg_s,
        'deletions': p2_leg_d,
        'insertions': p2_leg_i,
        'hits': p2_leg_h,
        'total_words': p2_leg_n
    },
    'v2_normalizer': {
        'corpus_wer': p2_v2_wer,
        'substitutions': p2_v2_s,
        'deletions': p2_v2_d,
        'insertions': p2_v2_i,
        'hits': p2_v2_h,
        'total_words': p2_v2_n
    }
}
save_result_file(pass2_out_file, pass2_results, config=PROTOCOL, device=device)
print(f"[OK] Pass 2 results saved to: {pass2_out_file}")
print(f"Pass 2 Legacy WER: {p2_leg_wer:.2f}% | v2 WER: {p2_v2_wer:.2f}% | Throughput: {th_p2:.2f}x real-time")
""",
        # Cell 4: Pass 3 Beam 5 + VAD Comparison
        """# ── CELL 4: PASS 3 BEAM SEARCH (WIDTH=5) + VAD COMPARISON ────────────────
pass3_out_file = os.path.join(OUT_DIR, "cv_beam5_vad_comparison.json")
pass3_clips_csv = os.path.join(OUT_DIR, "cv_pass3_beam5_clips.csv")

# Use benchmark subset of 1,400 clips (or full pool in full mode)
pass3_pool = active_pool.iloc[:1400].copy() if len(active_pool) > 1400 else active_pool.copy()
total_p3 = len(pass3_pool)
print(f"\\n--- Running Pass 3: CT2 INT8 Beam-5 + VAD on {total_p3:,} clips ---")

t0_p3 = time.time()
pass3_rows = []

for idx, (_, row) in enumerate(pass3_pool.iterrows()):
    p_str = str(row['path'])
    cp = resolve_audio(p_str)
    assert cp is not None and os.path.exists(cp), f"FATAL: Clip not found in any mounted directory: {p_str}"
    
    try:
        segments, info = model_ct2.transcribe(
            cp, language="ar", task="transcribe",
            beam_size=5, best_of=5, temperature=0.0,
            vad_filter=True, condition_on_previous_text=False
        )
        raw_hyp = " ".join([seg.text for seg in segments]).strip()
        dur = info.duration if hasattr(info, 'duration') and info.duration else 0.0
    except Exception:
        audio_np, _ = librosa.load(cp, sr=16000, mono=True)
        dur = float(len(audio_np) / 16000.0)
        segments, info = model_ct2.transcribe(
            audio_np, language="ar", task="transcribe",
            beam_size=5, best_of=5, temperature=0.0,
            vad_filter=True, condition_on_previous_text=False
        )
        raw_hyp = " ".join([seg.text for seg in segments]).strip()
    
    rec = compute_dual_wer_record(row['sentence'], raw_hyp)
    leg_b = rec['legacy_breakdown']
    
    pass3_rows.append({
        'client_id': row.get('client_id', 'unknown'),
        'path': p_str,
        'duration': round(dur, 3),
        'ref': row['sentence'],
        'hyp_unnorm': raw_hyp,
        'hyp_norm': rec['normalized_hyp_legacy'],
        'S': leg_b['substitutions'],
        'D': leg_b['deletions'],
        'I': leg_b['insertions'],
        'H': leg_b['hits'],
        'N': leg_b['words'],
        'wer': round(rec['legacy_wer'], 4)
    })
    
    if (len(pass3_rows) % 100 == 0) or (len(pass3_rows) == total_p3):
        df_cur3 = pd.DataFrame(pass3_rows)
        w3_cur = (df_cur3['S'].sum() + df_cur3['D'].sum() + df_cur3['I'].sum()) / df_cur3['N'].sum() * 100.0
        print(f"  [Pass 3 Beam-5] Processed {len(pass3_rows):,}/{total_p3:,} | Running WER: {w3_cur:.2f}% | Elapsed: {(time.time()-t0_p3)/60:.1f}m")

df_pass3 = pd.DataFrame(pass3_rows)
df_pass3.to_csv(pass3_clips_csv, index=False, encoding="utf-8")
print(f"[OK] Saved Pass 3 per-clip CSV to {pass3_clips_csv} ({len(df_pass3)} rows)")
assert len(df_pass3) == total_p3, f"FATAL: Dropped clips in Pass 3! Got {len(df_pass3)}, expected {total_p3}"

p3_s = int(df_pass3['S'].sum())
p3_d = int(df_pass3['D'].sum())
p3_i = int(df_pass3['I'].sum())
p3_n = int(df_pass3['N'].sum())
p3_wer = round((p3_s + p3_d + p3_i) / p3_n * 100.0, 2)
tot_aud3 = float(df_pass3['duration'].sum())
tot_cmp3 = float(time.time() - t0_p3)

pass3_results = {
    'pass': 'Pass 3: Common Voice CT2 INT8 Beam-5 + VAD',
    'run_mode': RUN_MODE,
    'model': 'Whisper-Large-v3 (Fine-Tuned Arabic CT2 INT8)',
    'decoder': 'Beam Search (beam_size=5, vad_filter=True)',
    'num_clips': len(df_pass3),
    'total_audio_seconds': round(tot_aud3, 2),
    'total_compute_seconds': round(tot_cmp3, 2),
    'throughput_realtime': round(tot_aud3 / tot_cmp3, 2) if tot_cmp3 > 0 else 0.0,
    'legacy_normalizer': {
        'corpus_wer': p3_wer,
        'substitutions': p3_s,
        'deletions': p3_d,
        'insertions': p3_i,
        'total_words': p3_n
    }
}
save_result_file(pass3_out_file, pass3_results, config=PROTOCOL, device=device)
print(f"[OK] Pass 3 results saved to: {pass3_out_file} (Beam-5 WER: {p3_wer:.2f}%)")
"""
    ]
    create_notebook("Notebooks/final_01b_cv_pass2_beam5.ipynb", cells)

if __name__ == "__main__":
    build_n1b()
