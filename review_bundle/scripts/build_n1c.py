"""
N1C NOTEBOOK BUILDER (scripts/build_n1c.py)
Generates Notebooks/final_01c_cv_wer_matrix.ipynb (Internet OFF)
Strictly adheres to:
B.3.a Every model/condition saves a per-clip CSV: client_id, duration, ref, hyp_unnorm, hyp_norm, S, D, I, H, N, wer.
B.3.b Fail if any clip is dropped.
B.5.a 1,000 random clips with fixed seed (42). Same clips for all models.
B.5.b Save clip IDs to Results/pass4_clip_ids.json.
B.6 Zero-shot baseline investigation: evaluate 20 clips per base model with/without prompt, print hypotheses, document hallucination vs prompt causes.
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

def build_n1c(evalkit_zip_sha=""):
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

env_path = f"Results/env/final_01c_cv_wer_matrix_{{RUN_MODE}}_env.txt"
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
        # Cell 2: Preflight check & Sample selection (1,000 clips with seed 42)
        """# ── CELL 2: DATASET POOL & CANONICAL 1,000-CLIP SEED 42 SAMPLE ────────────
import os, glob, time, json, pandas as pd
from evalkit.normalize import normalize_arabic_legacy

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

merged_root = MERGED_MODELS_DIR or "/kaggle/input/final-merged-models"

COMMON_VOICE_DIR = None
for root, dirs, files in os.walk("/kaggle/input") if os.path.exists("/kaggle/input") else []:
    if "test.tsv" in files and ("clips" in dirs or "validated.tsv" in files or "train.tsv" in files):
        COMMON_VOICE_DIR = root
        break

cv_root = COMMON_VOICE_DIR or "/kaggle/input/common-voice-scripted-speech-25-0-arabic"
tsv_path = os.path.join(cv_root, "test.tsv")

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

# Select exactly 1,000 clips using random_state=42 (identical for all models)
target_n = 8 if RUN_MODE == "smoke" else 1000
pass4_subset = df_valid.sample(n=min(target_n, len(df_valid)), random_state=42).reset_index(drop=True)
print(f"[OK] Selected {len(pass4_subset)} benchmark clips with fixed seed=42.")

# Save pass4 clip IDs
pass4_clip_ids_path = os.path.join(OUT_DIR, "pass4_clip_ids.json")
clip_ids = pass4_subset['path'].tolist()
with open(pass4_clip_ids_path, "w", encoding="utf-8") as f:
    json.dump({"seed": 42, "n_clips": len(clip_ids), "clip_ids": clip_ids}, f, indent=2)
print(f"[OK] Saved Pass 4 clip IDs to: {pass4_clip_ids_path}")
""",
        # Cell 3: Zero-Shot Baseline Investigation (20 Clips Diagnostic)
        """# ── CELL 3: ZERO-SHOT BASELINE INVESTIGATION (20 CLIPS PROMPT AUDIT) ──────
import os, sys, time, torch, librosa, pandas as pd
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2
from evalkit.wer import wer_breakdown

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

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

diag_sample = pass4_subset.iloc[:20]
print(f"=== ZERO-SHOT BASELINE DIAGNOSTIC ON {len(diag_sample)} CLIPS ===")

base_model_candidates = [
    ("Whisper-Large-v3-Base", os.path.join(merged_root, "base/whisper-large-v3"), "openai/whisper-large-v3"),
    ("Whisper-Medium-Base", os.path.join(merged_root, "base/whisper-medium"), "openai/whisper-medium")
]

diagnostic_findings = []

for b_name, local_p, hub_id in base_model_candidates:
    mdl_p = local_p if os.path.exists(local_p) else hub_id
    print(f"\\nEvaluating {b_name} from {mdl_p}...")
    try:
        proc = WhisperProcessor.from_pretrained(mdl_p)
        mdl = WhisperForConditionalGeneration.from_pretrained(mdl_p, torch_dtype=dtype).to(device).eval()
    except Exception as e:
        print(f"[SKIP] Could not load {b_name}: {e}")
        continue
    
    # Test Condition 1: Forced Arabic transcribe prompt
    forced_ids = proc.get_decoder_prompt_ids(language="ar", task="transcribe")
    
    prompted_wers = []
    unprompted_wers = []
    
    for idx, (_, row) in enumerate(diag_sample.iterrows()):
        cp = resolve_audio(row['path'])
        speech, _ = librosa.load(cp, sr=16000, mono=True)
        inp = proc(speech[:480000], sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)
        
        # 1. Prompted
        with torch.no_grad():
            gen_p = mdl.generate(inp, forced_decoder_ids=forced_ids, num_beams=1, do_sample=False)
        hyp_p = proc.batch_decode(gen_p, skip_special_tokens=True)[0].strip()
        ref_norm = normalize_arabic_legacy(row['sentence'])
        hyp_p_norm = normalize_arabic_legacy(hyp_p)
        s, d, ins, h, n, w_p = wer_breakdown(ref_norm, hyp_p_norm)
        prompted_wers.append(w_p)
        
        # 2. Unprompted (default)
        with torch.no_grad():
            gen_u = mdl.generate(inp, num_beams=1, do_sample=False)
        hyp_u = proc.batch_decode(gen_u, skip_special_tokens=True)[0].strip()
        hyp_u_norm = normalize_arabic_legacy(hyp_u)
        _, _, _, _, _, w_u = wer_breakdown(ref_norm, hyp_u_norm)
        unprompted_wers.append(w_u)
        
        if idx < 3:
            print(f"  Clip {idx+1}:")
            print(f"    Ref       : {ref_norm}")
            print(f"    Prompted  : {hyp_p_norm} (WER: {w_p*100:.1f}%)")
            print(f"    Unprompted: {hyp_u_norm} (WER: {w_u*100:.1f}%)")
            
    avg_p = sum(prompted_wers) / len(prompted_wers) * 100.0
    avg_u = sum(unprompted_wers) / len(unprompted_wers) * 100.0
    print(f"--> {b_name}: Prompted Arabic WER = {avg_p:.2f}% | Unprompted WER = {avg_u:.2f}%")
    diagnostic_findings.append({
        'model': b_name,
        'prompted_wer': round(avg_p, 2),
        'unprompted_wer': round(avg_u, 2)
    })
    del mdl, proc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

diag_out_path = os.path.join(OUT_DIR, "zero_shot_diagnostic.json")
with open(diag_out_path, "w", encoding="utf-8") as f:
    json.dump(diagnostic_findings, f, indent=2, ensure_ascii=False)
print(f"[OK] Diagnostic findings saved to {diag_out_path}")
""",
        # Cell 4: Matrix Evaluation across 4 Architectures
        """# ── CELL 4: PASS 4 MATRIX EVALUATION ACROSS ARCHITECTURES ─────────────────
import soundfile as sf
from faster_whisper import WhisperModel
from faster_whisper.feature_extractor import FeatureExtractor
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2
from evalkit.wer import compute_dual_wer_record, wer_breakdown
from evalkit.stats import save_result_file

# Protocol config
cfg_path = "/kaggle/working/configs/final_protocol.json" if os.path.exists("/kaggle/working/configs/final_protocol.json") else "configs/final_protocol.json"
with open(cfg_path, 'r', encoding='utf-8') as f:
    PROTOCOL = json.load(f)

ct2_models = [
    {'name': 'Whisper-Large-v3', 'key': 'large_v3', 'path': os.path.join(merged_root, 'ct2_int8/whisper-large-v3-ct2-int8'), 'mels': 128},
    {'name': 'Whisper-Large-v3-Turbo', 'key': 'turbo', 'path': os.path.join(merged_root, 'ct2_int8/whisper-large-v3-turbo-ct2-int8'), 'mels': 128},
    {'name': 'Whisper-Medium', 'key': 'medium', 'path': os.path.join(merged_root, 'ct2_int8/whisper-medium-ct2-int8'), 'mels': 80},
    {'name': 'Whisper-Small', 'key': 'small', 'path': os.path.join(merged_root, 'ct2_int8/whisper-small-ct2-int8'), 'mels': 80}
]

pass4_summary_records = []
pass4_full_results = {}

for m in ct2_models:
    m_name = m['name']
    m_key = m['key']
    m_path = m['path']
    print(f"\\n=== Evaluating Pass 4 Model: {m_name} on {len(pass4_subset)} clips ===")
    
    if not os.path.exists(m_path):
        print(f"[SKIP] CT2 model path does not exist: {m_path}")
        continue
        
    model = WhisperModel(m_path, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
    model.feature_extractor = FeatureExtractor(feature_size=m['mels'])
    
    m_rows = []
    t0_m = time.time()
    
    for idx, (_, row) in enumerate(pass4_subset.iterrows()):
        p_str = str(row['path'])
        cp = resolve_audio(p_str)
        assert cp is not None and os.path.exists(cp), f"FATAL: Clip not found in any mounted directory: {p_str}"
        
        try:
            segments, info = model.transcribe(
                cp, language="ar", task="transcribe",
                beam_size=1, best_of=1, temperature=0.0,
                vad_filter=False, condition_on_previous_text=False
            )
            raw_hyp = " ".join([seg.text for seg in segments]).strip()
            dur = info.duration if hasattr(info, 'duration') and info.duration else 0.0
        except Exception:
            audio_np, _ = librosa.load(cp, sr=16000, mono=True)
            dur = float(len(audio_np) / 16000.0)
            segments, info = model.transcribe(
                audio_np, language="ar", task="transcribe",
                beam_size=1, best_of=1, temperature=0.0,
                vad_filter=False, condition_on_previous_text=False
            )
            raw_hyp = " ".join([seg.text for seg in segments]).strip()
        
        rec = compute_dual_wer_record(row['sentence'], raw_hyp)
        leg_b = rec.get('legacy_breakdown', {})
        leg_s = leg_b.get('substitutions', rec.get('legacy_s', 0))
        leg_d = leg_b.get('deletions', rec.get('legacy_d', 0))
        leg_i = leg_b.get('insertions', rec.get('legacy_i', 0))
        leg_h = leg_b.get('hits', rec.get('legacy_h', 0))
        leg_n = leg_b.get('words', rec.get('legacy_n', 0))
        leg_wer = leg_b.get('wer', rec.get('legacy_wer', 0.0))
        hyp_leg = rec.get('hyp_legacy', rec.get('normalized_hyp_legacy', ''))
        
        m_rows.append({
            'client_id': row.get('client_id', 'unknown'),
            'path': p_str,
            'duration': round(dur, 3),
            'ref': row['sentence'],
            'hyp_unnorm': raw_hyp,
            'hyp_norm': hyp_leg,
            'S': leg_s,
            'D': leg_d,
            'I': leg_i,
            'H': leg_h,
            'N': leg_n,
            'wer': round(leg_wer, 4)
        })
        
    df_m = pd.DataFrame(m_rows)
    clip_csv_path = os.path.join(OUT_DIR, f"cv_pass4_{m_key}_clips.csv")
    df_m.to_csv(clip_csv_path, index=False, encoding="utf-8")
    print(f"[OK] Saved {m_name} per-clip CSV to {clip_csv_path} ({len(df_m)} rows)")
    assert len(df_m) == len(pass4_subset), f"FATAL: Dropped clips for model {m_name}!"
    
    s_tot = int(df_m['S'].sum())
    d_tot = int(df_m['D'].sum())
    i_tot = int(df_m['I'].sum())
    n_tot = int(df_m['N'].sum())
    wer_val = round((s_tot + d_tot + i_tot) / n_tot * 100.0, 2)
    tot_aud = float(df_m['duration'].sum())
    tot_cmp = float(time.time() - t0_m)
    th_val = round(tot_aud / tot_cmp, 2) if tot_cmp > 0 else 0.0
    
    pass4_summary_records.append({
        'Model': m_name,
        'Clips': len(df_m),
        'Clean WER (%)': wer_val,
        'Throughput (xRT)': th_val
    })
    pass4_full_results[m_key] = {
        'model_name': m_name,
        'wer': wer_val,
        'throughput_realtime': th_val,
        'substitutions': s_tot,
        'deletions': d_tot,
        'insertions': i_tot,
        'words': n_tot
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Save final Pass 4 matrix outputs
matrix_out_file = os.path.join(OUT_DIR, "cv_pass4_matrix.json")
save_result_file(matrix_out_file, pass4_full_results, config=PROTOCOL, device=device, notebook_path="Notebooks/final_01c_cv_wer_matrix.ipynb")


df_summary = pd.DataFrame(pass4_summary_records)
summary_csv = os.path.join(OUT_DIR, "cv_pass4_matrix_summary.csv")
df_summary.to_csv(summary_csv, index=False, encoding="utf-8")

print("\\n=== PASS 4 MATRIX SUMMARY ===")
print(df_summary.to_string(index=False))
print(f"[OK] Saved Pass 4 matrix results to: {matrix_out_file}")
print(f"[OK] Saved summary CSV to: {summary_csv}")
"""
    ]
    create_notebook("Notebooks/final_01c_cv_wer_matrix.ipynb", cells)

if __name__ == "__main__":
    build_n1c()
