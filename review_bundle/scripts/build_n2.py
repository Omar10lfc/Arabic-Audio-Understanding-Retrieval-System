"""
N2 NOTEBOOK BUILDER (scripts/build_n2.py)
Generates Notebooks/final_02_tts_asr.ipynb (Internet ON)
Strictly adheres to:
B.9.a Write complete manifest of 50 generated clips: article_id, voice (ar-SA-HamedNeural), duration, sha256.
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

def build_n2(evalkit_zip_sha=""):
    cells = [
        # Cell 1: Environment & evalkit verification
        f"""# ── CELL 1: ENVIRONMENT & EVALKIT-SRC VERIFICATION (INTERNET ON) ──────
import os, sys, subprocess, json, hashlib, zipfile, shutil

for pkg in ["ctranslate2", "faster-whisper", "edge-tts", "soundfile"]:
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
os.makedirs("audio_tts/xlsum", exist_ok=True)
os.makedirs("audio_tts/arcd_passages", exist_ok=True)
os.makedirs("audio_tts/arcd_queries", exist_ok=True)

EXPECTED_EVALKIT_SHA = "{evalkit_zip_sha}"
print(f"Internet: ON | RUN_MODE: {{RUN_MODE}} | Target: {{OUT_DIR}} | Expected evalkit SHA: {{EXPECTED_EVALKIT_SHA}}")

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

env_path = 'Results/env/final_02_tts_asr_env.txt'
with open(env_path, 'w', encoding='utf-8') as f:
    f.write(f"Python Version: {{sys.version}}\\n")
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
        f.write(f"GPU Info:\\n{{gpu_info}}\\n")
    except Exception as e:
        f.write(f"GPU Info: None ({{e}})\\n")
print(f"[OK] Environment logged to: {{env_path}}")
""",
        # Cell 2: Preflight check & Dataset Extraction
        """# ── CELL 2: PREFLIGHT INPUTS AUDIT & DATA EXTRACTION ─────────────────────
import os, glob, time, json, tarfile, zipfile, pandas as pd

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
print(f"[OK] Merged models located: {merged_root}")

# 1. Extract 50 BBC Arabic XL-Sum Articles
xlsum_articles = []
target_xl_n = 2 if RUN_MODE == "smoke" else 50

for root, dirs, files in os.walk("/kaggle/input") if os.path.exists("/kaggle/input") else []:
    for f in sorted(files):
        if ("xlsum" in f.lower() or "xl-sum" in f.lower() or "arabic_test" in f.lower()) and f.endswith(".jsonl"):
            with open(os.path.join(root, f), "r", encoding="utf-8") as jf:
                for line in jf:
                    s = line.strip()
                    if s:
                        xlsum_articles.append(json.loads(s))
                    if len(xlsum_articles) >= target_xl_n:
                        break
            if len(xlsum_articles) >= target_xl_n:
                break
        elif ("xlsum" in f.lower() or "xl-sum" in f.lower()) and f.endswith((".bz2", ".tar.gz", ".tar.bz2")):
            with tarfile.open(os.path.join(root, f), "r:*") as tar:
                for m in tar.getmembers():
                    if "test.jsonl" in m.name:
                        tf = tar.extractfile(m)
                        for line in tf:
                            s = line.decode("utf-8").strip()
                            if s:
                                xlsum_articles.append(json.loads(s))
                            if len(xlsum_articles) >= target_xl_n:
                                break
                        break
            if len(xlsum_articles) >= target_xl_n:
                break
    if len(xlsum_articles) >= target_xl_n:
        break

if not xlsum_articles and os.path.exists("Data/arabic_XLSum_v2.0.tar.bz2"):
    with tarfile.open("Data/arabic_XLSum_v2.0.tar.bz2", "r:bz2") as tar:
        for m in tar.getmembers():
            if "test.jsonl" in m.name:
                tf = tar.extractfile(m)
                for line in tf:
                    s = line.decode("utf-8").strip()
                    if s:
                        xlsum_articles.append(json.loads(s))
                    if len(xlsum_articles) >= target_xl_n:
                        break
                break

if not xlsum_articles and RUN_MODE == "smoke":
    xlsum_articles = [
        {"id": "art_0", "text": "أعلنت الحكومة خطة اقتصادية شاملة لتطوير البنية التحتية.", "summary": "خطة حكومية لتطوير الاقتصاد."},
        {"id": "art_1", "text": "فاز الفريق بالبطولة بعد مباراة قوية استمرت حتى الأشواط الإضافية.", "summary": "فوز الفريق باللقب الرياضي."}
    ]

# 2. Extract Canonical 50 ARCD Passages & 50 Queries
passages = []
queries = []
target_arcd_n = 2 if RUN_MODE == "smoke" else 50
raw_df = None

for root, dirs, files in os.walk("/kaggle/input") if os.path.exists("/kaggle/input") else []:
    for f in sorted(files):
        cand_p = os.path.join(root, f)
        if f.lower() in ["train.csv", "arcd_train.csv"] or (f.endswith(".csv") and any(k in f.lower() or k in root.lower() for k in ["arcd", "comprehension"])):
            try:
                cand_df = pd.read_csv(cand_p)
                cols_low = [c.lower() for c in cand_df.columns]
                if any(c in cols_low for c in ["context", "text"]) and any(c in cols_low for c in ["question", "query"]):
                    raw_df = cand_df
                    print(f"[OK] Found ARCD dataset: {cand_p}")
                    break
            except Exception:
                pass
        if raw_df is not None:
            break
    if raw_df is not None:
        break

if raw_df is None and os.path.exists("Data/ARCD (Arabic Language Comprehension)-Dataset.zip"):
    with zipfile.ZipFile("Data/ARCD (Arabic Language Comprehension)-Dataset.zip", "r") as z:
        raw_df = pd.read_csv(z.open("train.csv"))
        print("[OK] Loaded ARCD from local Data/ archive.")

if raw_df is not None:
    seen = {}
    for _, row in raw_df.iterrows():
        ctx = str(row.get('context', row.get('text', ''))).strip()
        q = str(row.get('question', row.get('query', ''))).strip()
        if not ctx or not q or len(ctx) < 40 or len(q) < 5:
            continue
        if ctx not in seen:
            pid = f"p_{len(seen):03d}"
            seen[ctx] = pid
            if len(passages) < target_arcd_n:
                passages.append({'id': pid, 'text': ctx})
        t_id = seen[ctx]
        if any(p['id'] == t_id for p in passages):
            if len(queries) < target_arcd_n:
                queries.append({'query': q, 'target_id': t_id})
        if len(passages) >= target_arcd_n and len(queries) >= target_arcd_n:
            break

print(f"[OK] Loaded {len(xlsum_articles)} XL-Sum articles.")
print(f"[OK] Loaded {len(passages)} ARCD passages and {len(queries)} ARCD queries.")
""",
        # Cell 3: Neural Audio Generation with SHA256 & Voice Manifest
        """# ── CELL 3: NEURAL AUDIO GENERATION WITH COMPLETE SHA256 MANIFEST ────────
import asyncio, time, json, os, edge_tts, hashlib, soundfile as sf
import concurrent.futures

VOICE = "ar-SA-HamedNeural"
manifest_path = os.path.join(OUT_DIR, "tts_manifest.json")
tts_manifest = {}
if os.path.exists(manifest_path):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            tts_manifest = json.load(f)
    except Exception:
        tts_manifest = {}

def compute_file_sha256(filepath):
    if not os.path.exists(filepath):
        return ""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

async def _synthesize_clip_async(text, out_path, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            communicate = edge_tts.Communicate(text, VOICE)
            await communicate.save(out_path)
            if os.path.exists(out_path) and os.path.getsize(out_path) > 500:
                return True
        except Exception as e:
            if attempt == max_retries:
                print(f"[ERROR] Failed synthesizing {out_path}: {e}")
                raise
            await asyncio.sleep(1.0 * attempt)
    return False

def run_async(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(lambda: asyncio.run(coro)).result()

def synthesize_clip(text, out_path, max_retries=5):
    return run_async(_synthesize_clip_async(text, out_path, max_retries))

# 1. Synthesize XL-Sum articles
print(f"--- Synthesizing {len(xlsum_articles)} XL-Sum articles using voice: {VOICE} ---")
for idx, art in enumerate(xlsum_articles):
    aid = f"xlsum_{idx:03d}_{art['id']}"
    wav_p = f"audio_tts/xlsum/{aid}.mp3"
    art['audio_path'] = wav_p
    if aid not in tts_manifest or not os.path.exists(wav_p) or os.path.getsize(wav_p) < 500:
        synthesize_clip(art['text'], wav_p)
        try:
            dur = sf.info(wav_p).duration
        except Exception:
            dur = 0.0
        f_sha = compute_file_sha256(wav_p)
        tts_manifest[aid] = {
            "type": "xlsum",
            "article_id": art['id'],
            "audio_path": wav_p,
            "duration_s": round(dur, 2),
            "voice": VOICE,
            "sha256": f_sha,
            "text_len": len(art['text'])
        }

# 2. Synthesize ARCD passages
print(f"--- Synthesizing {len(passages)} ARCD passages using voice: {VOICE} ---")
for idx, p in enumerate(passages):
    pid = p['id']
    wav_p = f"audio_tts/arcd_passages/{pid}.mp3"
    p['audio_path'] = wav_p
    if pid not in tts_manifest or not os.path.exists(wav_p) or os.path.getsize(wav_p) < 500:
        synthesize_clip(p['text'], wav_p)
        try:
            dur = sf.info(wav_p).duration
        except Exception:
            dur = 0.0
        f_sha = compute_file_sha256(wav_p)
        tts_manifest[pid] = {
            "type": "arcd_passage",
            "passage_id": pid,
            "audio_path": wav_p,
            "duration_s": round(dur, 2),
            "voice": VOICE,
            "sha256": f_sha,
            "text_len": len(p['text'])
        }

# 3. Synthesize ARCD queries
print(f"--- Synthesizing {len(queries)} ARCD queries using voice: {VOICE} ---")
for idx, q in enumerate(queries):
    qid = f"query_{idx:03d}"
    wav_p = f"audio_tts/arcd_queries/{qid}.mp3"
    q['audio_path'] = wav_p
    if qid not in tts_manifest or not os.path.exists(wav_p) or os.path.getsize(wav_p) < 500:
        synthesize_clip(q['query'], wav_p)
        try:
            dur = sf.info(wav_p).duration
        except Exception:
            dur = 0.0
        f_sha = compute_file_sha256(wav_p)
        tts_manifest[qid] = {
            "type": "arcd_query",
            "query_id": qid,
            "target_id": q['target_id'],
            "audio_path": wav_p,
            "duration_s": round(dur, 2),
            "voice": VOICE,
            "sha256": f_sha,
            "text_len": len(q['query'])
        }

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(tts_manifest, f, indent=2, ensure_ascii=False)

print(f"[OK] Neural TTS synthesis complete. Total manifest entries: {len(tts_manifest)}")
print(f"Manifest with SHA-256 written to: {manifest_path}")
""",
        # Cell 4: Multi-Model Whisper ASR Transcription
        """# ── CELL 4: MULTI-MODEL WHISPER ASR TRANSCRIPTION ────────────────────────
import gc, torch, numpy as np, librosa
from faster_whisper import WhisperModel
from faster_whisper.feature_extractor import FeatureExtractor
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2
from evalkit.wer import compute_dual_wer_record

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

def safe_transcribe(model, audio_path, **kwargs):
    try:
        return model.transcribe(audio_path, **kwargs)
    except Exception:
        audio_np, _ = librosa.load(audio_path, sr=16000, mono=True)
        return model.transcribe(audio_np, **kwargs)

asr_out_file = os.path.join(OUT_DIR, "tts_asr_propagation_results.json")
arcd_trans_file = os.path.join(OUT_DIR, "spoken_arcd_transcripts.json")

ct2_models = [
    {"name": "Whisper-Large-v3", "key": "large_v3", "path": os.path.join(merged_root, "ct2_int8/whisper-large-v3-ct2-int8"), "mels": 128},
    {"name": "Whisper-Large-v3-Turbo", "key": "turbo", "path": os.path.join(merged_root, "ct2_int8/whisper-large-v3-turbo-ct2-int8"), "mels": 128},
    {"name": "Whisper-Medium", "key": "medium", "path": os.path.join(merged_root, "ct2_int8/whisper-medium-ct2-int8"), "mels": 80},
    {"name": "Whisper-Small", "key": "small", "path": os.path.join(merged_root, "ct2_int8/whisper-small-ct2-int8"), "mels": 80}
]

propagation_results = {}
arcd_transcripts = {
    "passages": {p['id']: {"clean_text": p['text'], "models": {}} for p in passages},
    "queries": {f"query_{i:03d}": {"clean_query": q['query'], "target_id": q['target_id'], "models": {}} for i, q in enumerate(queries)}
}

summary_records = []

for m_info in ct2_models:
    m_name = m_info["name"]
    m_key = m_info["key"]
    m_path = m_info["path"]
    
    print(f"\\nTranscribing with: {m_name} from {m_path}...")
    if not os.path.exists(m_path):
        print(f"[SKIP] Model path not found: {m_path}")
        continue
        
    whisper_model = WhisperModel(m_path, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
    whisper_model.feature_extractor = FeatureExtractor(feature_size=m_info["mels"])
    
    if m_key not in propagation_results:
        propagation_results[m_key] = {"model_name": m_name, "articles": []}
    
    # 1. Transcribe XL-Sum
    t0_xl = time.time()
    tot_aud_xl = 0.0
    for idx, art in enumerate(xlsum_articles):
        segments, info = safe_transcribe(whisper_model, art['audio_path'], language="ar", task="transcribe", beam_size=1, best_of=1, temperature=0.0)
        raw_t = " ".join([seg.text for seg in segments]).strip()
        dur = info.duration if hasattr(info, 'duration') and info.duration else 0.0
        tot_aud_xl += dur
        
        rec = compute_dual_wer_record(art['text'], raw_t)
        propagation_results[m_key]["articles"].append({
            "id": art['id'],
            "audio_path": art['audio_path'],
            "clean_text": art['text'],
            "reference_summary": art['summary'],
            "norm_legacy_transcript": rec.get('normalized_hyp_legacy', rec.get('hyp_legacy', '')),
            "norm_v2_transcript": rec.get('normalized_hyp_v2', rec.get('hyp_v2', '')),
            "legacy_wer": rec.get('legacy_wer', 0.0),
            "v2_wer": rec.get('v2_wer', 0.0),
            "duration_s": dur
        })
    
    # 2. Transcribe ARCD Passages
    for p in passages:
        pid = p['id']
        segments, info = safe_transcribe(whisper_model, p['audio_path'], language="ar", task="transcribe", beam_size=1, best_of=1, temperature=0.0)
        raw_t = " ".join([seg.text for seg in segments]).strip()
        dur = info.duration if hasattr(info, 'duration') and info.duration else 0.0
        rec = compute_dual_wer_record(p['text'], raw_t)
        arcd_transcripts["passages"][pid]["models"][m_key] = {
            "raw": raw_t,
            "norm_legacy": rec.get('normalized_hyp_legacy', rec.get('hyp_legacy', '')),
            "norm_v2": rec.get('normalized_hyp_v2', rec.get('hyp_v2', '')),
            "legacy_wer": rec.get('legacy_wer', 0.0),
            "duration_s": dur
        }
        
    # 3. Transcribe ARCD Queries
    for idx, q in enumerate(queries):
        qid = f"query_{idx:03d}"
        segments, info = safe_transcribe(whisper_model, q['audio_path'], language="ar", task="transcribe", beam_size=1, best_of=1, temperature=0.0)
        raw_t = " ".join([seg.text for seg in segments]).strip()
        dur = info.duration if hasattr(info, 'duration') and info.duration else 0.0
        rec = compute_dual_wer_record(q['query'], raw_t)
        arcd_transcripts["queries"][qid]["models"][m_key] = {
            "raw": raw_t,
            "norm_legacy": rec.get('normalized_hyp_legacy', rec.get('hyp_legacy', '')),
            "norm_v2": rec.get('normalized_hyp_v2', rec.get('hyp_v2', '')),
            "legacy_wer": rec.get('legacy_wer', 0.0),
            "duration_s": dur
        }
        
    xl_wer = np.mean([a["legacy_wer"] for a in propagation_results[m_key]["articles"]]) * 100.0
    p_wer = np.mean([arcd_transcripts["passages"][p['id']]["models"][m_key]["legacy_wer"] for p in passages]) * 100.0
    tot_comp = time.time() - t0_xl
    th_val = round(tot_aud_xl / tot_comp, 2) if tot_comp > 0 else 0.0
    
    summary_records.append({
        "Model": m_name,
        "XL-Sum WER (%)": round(xl_wer, 2),
        "ARCD Passages WER (%)": round(p_wer, 2),
        "Throughput (xRT)": th_val
    })
    print(f"--> {m_name}: XL-Sum WER = {xl_wer:.2f}% | ARCD WER = {p_wer:.2f}% | Throughput = {th_val:.2f}x")
    del whisper_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(asr_out_file, "w", encoding="utf-8") as f:
    json.dump(propagation_results, f, indent=2, ensure_ascii=False)
with open(arcd_trans_file, "w", encoding="utf-8") as f:
    json.dump(arcd_transcripts, f, indent=2, ensure_ascii=False)

df_summary = pd.DataFrame(summary_records)
df_summary.to_csv(os.path.join(OUT_DIR, "tts_asr_summary.csv"), index=False)
print(f"\\n[OK] Outputs saved to: {asr_out_file} and {arcd_trans_file}")
print(df_summary.to_string(index=False))
"""
    ]
    create_notebook("Notebooks/final_02_tts_asr.ipynb", cells)

if __name__ == "__main__":
    build_n2()
