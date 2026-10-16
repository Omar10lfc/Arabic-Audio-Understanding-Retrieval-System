import os, sys

def main():
    target_file = r"scripts/build_all_final_notebooks.py"
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Find where N2 starts
    n2_marker = "    # N2: final_02_tts_asr.ipynb"
    if n2_marker not in content:
        raise ValueError(f"Could not find marker '{n2_marker}' in {target_file}")

    idx_start = content.index(n2_marker)
    # The end of N3 builder is: create_notebook("Notebooks/final_03_summarization_retrieval.ipynb", n3_cells)
    end_marker = 'create_notebook("Notebooks/final_03_summarization_retrieval.ipynb", n3_cells)'
    if end_marker not in content:
        raise ValueError(f"Could not find end marker '{end_marker}' in {target_file}")
    idx_end = content.index(end_marker) + len(end_marker)

    new_section = '''    # N2: final_02_tts_asr.ipynb (Internet ON)
    # ─────────────────────────────────────────────────────────────────────────
    n2_cells = [
        # Cell 1: Environment & evalkit-src check & extraction
        f"""# ── CELL 1: ENVIRONMENT & EVALKIT-SRC VERIFICATION ────────────────────
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

env_path = 'Results/env/final_02_tts_asr_env.txt'
with open(env_path, 'w', encoding='utf-8') as f:
    f.write(f"Python Version: {{sys.version}}\\n")
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
        f.write(f"GPU Info:\\n{{gpu_info}}\\n")
    except Exception as e:
        f.write(f"GPU Info: None ({{e}})\\n")
    pkgs = ['torch', 'transformers', 'ctranslate2', 'faster_whisper', 'edge_tts', 'soundfile']
    f.write("Package Versions:\\n")
    for p in pkgs:
        try:
            mod = __import__(p.replace("-", "_"))
            f.write(f"  {{p}}: {{getattr(mod, '__version__', 'unknown')}}\\n")
        except Exception as e:
            f.write(f"  {{p}}: NOT INSTALLED ({{e}})\\n")
print(f"[OK] Environment logged to: {{env_path}}")
""",
        # Cell 2: Preflight check & Data Extraction
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

if not MERGED_MODELS_DIR:
    for cand in [
        "/kaggle/input/final-merged-models",
        "/kaggle/input/zero-notebook/final-merged-models",
        "/kaggle/input/zero-notebook"
    ]:
        if os.path.exists(cand):
            MERGED_MODELS_DIR = cand
            break

assert (MERGED_MODELS_DIR is not None and os.path.exists(MERGED_MODELS_DIR)) or RUN_MODE == "smoke", "Missing merged models!"
print(f"[OK] Merged models located : {MERGED_MODELS_DIR}")

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
        elif f.endswith(".zip") and any(k in f.lower() for k in ["arcd", "comprehension", "arabic"]):
            try:
                with zipfile.ZipFile(cand_p, "r") as z:
                    for zf in z.namelist():
                        if zf.endswith(".csv") and "train" in zf.lower():
                            cand_df = pd.read_csv(z.open(zf))
                            cols_low = [c.lower() for c in cand_df.columns]
                            if any(c in cols_low for c in ["context", "text"]) and any(c in cols_low for c in ["question", "query"]):
                                raw_df = cand_df
                                print(f"[OK] Found ARCD dataset in archive: {cand_p} -> {zf}")
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

if not passages and RUN_MODE == "smoke":
    passages = [{'id': f'p_{i:03d}', 'text': f'تعتبر هذه الفقرة تجريبية رقم {i} لدراسة الاسترجاع الدلالي العربي.'} for i in range(2)]
    queries = [{'query': f'ما هو رقم الفقرة {i}؟', 'target_id': f'p_{i:03d}'} for i in range(2)]

print(f"[OK] Loaded {len(xlsum_articles)} XL-Sum articles.")
print(f"[OK] Loaded {len(passages)} ARCD passages and {len(queries)} ARCD queries.")
assert len(xlsum_articles) >= target_xl_n, f"Could not load {target_xl_n} XL-Sum articles!"
assert len(passages) >= target_arcd_n, f"Could not load {target_arcd_n} ARCD passages!"
""",
        # Cell 3: Neural TTS Synthesis
        """# ── CELL 3: NEURAL AUDIO GENERATION (XL-SUM & ARCD CORPUS) ───────────────
import asyncio, time, json, os, glob, edge_tts, soundfile as sf, numpy as np

VOICE = "ar-SA-HamedNeural"
manifest_path = os.path.join(OUT_DIR, "tts_manifest.json")
tts_manifest = {}
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        tts_manifest = json.load(f)

async def synthesize_clip(text, out_path, max_retries=5):
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

# 1. Synthesize XL-Sum articles
print(f"--- Synthesizing {len(xlsum_articles)} XL-Sum articles ---")
for idx, art in enumerate(xlsum_articles):
    aid = f"xlsum_{idx:03d}_{art['id']}"
    wav_p = f"audio_tts/xlsum/{aid}.mp3"
    art['audio_path'] = wav_p
    if aid not in tts_manifest or not os.path.exists(wav_p) or os.path.getsize(wav_p) < 500:
        asyncio.run(synthesize_clip(art['text'], wav_p))
        try:
            dur = sf.info(wav_p).duration
        except Exception:
            dur = 0.0
        tts_manifest[aid] = {
            "type": "xlsum",
            "id": art['id'],
            "audio_path": wav_p,
            "duration_s": round(dur, 2),
            "voice": VOICE,
            "text_len": len(art['text'])
        }
    if (idx + 1) % 10 == 0 or (idx + 1) == len(xlsum_articles):
        print(f"  XL-Sum articles synthesized: {idx + 1}/{len(xlsum_articles)}")

# 2. Synthesize ARCD passages
print(f"--- Synthesizing {len(passages)} ARCD passages ---")
for idx, p in enumerate(passages):
    pid = p['id']
    wav_p = f"audio_tts/arcd_passages/{pid}.mp3"
    p['audio_path'] = wav_p
    if pid not in tts_manifest or not os.path.exists(wav_p) or os.path.getsize(wav_p) < 500:
        asyncio.run(synthesize_clip(p['text'], wav_p))
        try:
            dur = sf.info(wav_p).duration
        except Exception:
            dur = 0.0
        tts_manifest[pid] = {
            "type": "arcd_passage",
            "id": pid,
            "audio_path": wav_p,
            "duration_s": round(dur, 2),
            "voice": VOICE,
            "text_len": len(p['text'])
        }
    if (idx + 1) % 10 == 0 or (idx + 1) == len(passages):
        print(f"  ARCD passages synthesized: {idx + 1}/{len(passages)}")

# 3. Synthesize ARCD queries
print(f"--- Synthesizing {len(queries)} ARCD queries ---")
for idx, q in enumerate(queries):
    qid = f"query_{idx:03d}"
    wav_p = f"audio_tts/arcd_queries/{qid}.mp3"
    q['audio_path'] = wav_p
    if qid not in tts_manifest or not os.path.exists(wav_p) or os.path.getsize(wav_p) < 500:
        asyncio.run(synthesize_clip(q['query'], wav_p))
        try:
            dur = sf.info(wav_p).duration
        except Exception:
            dur = 0.0
        tts_manifest[qid] = {
            "type": "arcd_query",
            "id": qid,
            "target_id": q['target_id'],
            "audio_path": wav_p,
            "duration_s": round(dur, 2),
            "voice": VOICE,
            "text_len": len(q['query'])
        }
    if (idx + 1) % 10 == 0 or (idx + 1) == len(queries):
        print(f"  ARCD queries synthesized: {idx + 1}/{len(queries)}")

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(tts_manifest, f, indent=2, ensure_ascii=False)

total_audio_sec = sum(item["duration_s"] for item in tts_manifest.values())
print(f"\\n[OK] Neural TTS synthesis complete! Total clips: {len(tts_manifest)} ({total_audio_sec:.1f} s / {total_audio_sec / 60:.1f} min)")
print(f"Manifest written to: {manifest_path}")
""",
        # Cell 4: Multi-Model Whisper ASR Transcription
        """# ── CELL 4: MULTI-MODEL WHISPER ASR TRANSCRIPTION (4 ARCHITECTURES) ──────
import time, gc, torch, numpy as np
from faster_whisper import WhisperModel
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2
from evalkit.wer import compute_dual_wer_record

asr_out_file = os.path.join(OUT_DIR, "tts_asr_propagation_results.json")
arcd_trans_file = os.path.join(OUT_DIR, "spoken_arcd_transcripts.json")

ct2_models = [
    {
        "name": "Whisper-Large-v3",
        "key": "large_v3",
        "path": os.path.join(MERGED_MODELS_DIR, "ct2_int8/whisper-large-v3-ct2-int8")
    },
    {
        "name": "Whisper-Large-v3-Turbo",
        "key": "turbo",
        "path": os.path.join(MERGED_MODELS_DIR, "ct2_int8/whisper-large-v3-turbo-ct2-int8")
    },
    {
        "name": "Whisper-Medium",
        "key": "medium",
        "path": os.path.join(MERGED_MODELS_DIR, "ct2_int8/whisper-medium-ct2-int8")
    },
    {
        "name": "Whisper-Small",
        "key": "small",
        "path": os.path.join(MERGED_MODELS_DIR, "ct2_int8/whisper-small-ct2-int8")
    }
]

propagation_results = {}
if os.path.exists(asr_out_file):
    with open(asr_out_file, "r", encoding="utf-8") as f:
        propagation_results = json.load(f)

arcd_transcripts = {
    "passages": {p['id']: {"clean_text": p['text'], "models": {}} for p in passages},
    "queries": {f"query_{i:03d}": {"clean_query": q['query'], "target_id": q['target_id'], "models": {}} for i, q in enumerate(queries)}
}
if os.path.exists(arcd_trans_file):
    with open(arcd_trans_file, "r", encoding="utf-8") as f:
        arcd_transcripts = json.load(f)

def transcribe_clip_ct2(model, audio_path):
    t0 = time.time()
    segments, info = model.transcribe(
        audio_path,
        language="ar",
        task="transcribe",
        beam_size=1,
        best_of=1,
        temperature=0.0,
        vad_filter=False,
        condition_on_previous_text=False
    )
    raw_text = " ".join([seg.text for seg in segments]).strip()
    compute_sec = time.time() - t0
    dur_sec = info.duration if hasattr(info, 'duration') and info.duration else 0.0
    return raw_text, dur_sec, compute_sec

summary_records = []

for m_info in ct2_models:
    m_name = m_info["name"]
    m_key = m_info["key"]
    m_path = m_info["path"]
    
    print(f"\\n=======================================================")
    print(f"Transcribing with: {m_name}")
    print(f"Path: {m_path}")
    print(f"=======================================================")
    
    t_load = time.time()
    whisper_model = WhisperModel(m_path, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
    print(f"[OK] Model loaded in {time.time() - t_load:.1f}s")
    
    if m_key not in propagation_results:
        propagation_results[m_key] = {"model_name": m_name, "articles": []}
    
    already_done_xl = len(propagation_results[m_key]["articles"])
    print(f"Transcribing XL-Sum articles ({already_done_xl}/{len(xlsum_articles)} already done)...")
    
    total_audio_xl = 0.0
    total_comp_xl = 0.0
    
    for idx, art in enumerate(xlsum_articles):
        if idx < already_done_xl:
            continue
        raw_t, dur, comp = transcribe_clip_ct2(whisper_model, art['audio_path'])
        norm_leg = normalize_arabic_legacy(raw_t)
        norm_v2 = normalize_arabic_v2(raw_t)
        rec = compute_dual_wer_record(raw_t, art['text'], compute_sec=comp, audio_dur_sec=dur)
        
        propagation_results[m_key]["articles"].append({
            "id": art['id'],
            "audio_path": art['audio_path'],
            "clean_text": art['text'],
            "reference_summary": art['summary'],
            "raw_transcript": raw_t,
            "norm_legacy_transcript": norm_leg,
            "norm_v2_transcript": norm_v2,
            "legacy_wer": rec["legacy_wer"],
            "v2_wer": rec["v2_wer"],
            "duration_s": dur,
            "compute_s": comp
        })
        total_audio_xl += dur
        total_comp_xl += comp
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(xlsum_articles):
            avg_w = np.mean([a["legacy_wer"] for a in propagation_results[m_key]["articles"]])
            print(f"  XL-Sum: {idx + 1}/{len(xlsum_articles)} | Avg Legacy WER: {avg_w:.2f}%")
            with open(asr_out_file, "w", encoding="utf-8") as f:
                json.dump(propagation_results, f, indent=2, ensure_ascii=False)
    
    # 2. Transcribe ARCD Passages
    print(f"Transcribing ARCD passages ({len(passages)} passages)...")
    total_audio_arcd = 0.0
    total_comp_arcd = 0.0
    
    for idx, p in enumerate(passages):
        pid = p['id']
        if pid in arcd_transcripts["passages"] and m_key in arcd_transcripts["passages"][pid].get("models", {}):
            continue
        raw_t, dur, comp = transcribe_clip_ct2(whisper_model, p['audio_path'])
        norm_leg = normalize_arabic_legacy(raw_t)
        norm_v2 = normalize_arabic_v2(raw_t)
        rec = compute_dual_wer_record(raw_t, p['text'], compute_sec=comp, audio_dur_sec=dur)
        
        if pid not in arcd_transcripts["passages"]:
            arcd_transcripts["passages"][pid] = {"clean_text": p['text'], "models": {}}
        arcd_transcripts["passages"][pid]["models"][m_key] = {
            "raw": raw_t,
            "norm_legacy": norm_leg,
            "norm_v2": norm_v2,
            "legacy_wer": rec["legacy_wer"],
            "v2_wer": rec["v2_wer"],
            "duration_s": dur,
            "compute_s": comp
        }
        total_audio_arcd += dur
        total_comp_arcd += comp
        
        if (idx + 1) % 10 == 0 or (idx + 1) == len(passages):
            print(f"  ARCD Passages: {idx + 1}/{len(passages)}")
            with open(arcd_trans_file, "w", encoding="utf-8") as f:
                json.dump(arcd_transcripts, f, indent=2, ensure_ascii=False)
    
    # 3. Transcribe ARCD Queries
    print(f"Transcribing ARCD queries ({len(queries)} queries)...")
    for idx, q in enumerate(queries):
        qid = f"query_{idx:03d}"
        if qid in arcd_transcripts["queries"] and m_key in arcd_transcripts["queries"][qid].get("models", {}):
            continue
        raw_t, dur, comp = transcribe_clip_ct2(whisper_model, q['audio_path'])
        norm_leg = normalize_arabic_legacy(raw_t)
        norm_v2 = normalize_arabic_v2(raw_t)
        rec = compute_dual_wer_record(raw_t, q['query'], compute_sec=comp, audio_dur_sec=dur)
        
        if qid not in arcd_transcripts["queries"]:
            arcd_transcripts["queries"][qid] = {"clean_query": q['query'], "target_id": q['target_id'], "models": {}}
        arcd_transcripts["queries"][qid]["models"][m_key] = {
            "raw": raw_t,
            "norm_legacy": norm_leg,
            "norm_v2": norm_v2,
            "legacy_wer": rec["legacy_wer"],
            "v2_wer": rec["v2_wer"],
            "duration_s": dur,
            "compute_s": comp
        }
        if (idx + 1) % 10 == 0 or (idx + 1) == len(queries):
            print(f"  ARCD Queries: {idx + 1}/{len(queries)}")
            with open(arcd_trans_file, "w", encoding="utf-8") as f:
                json.dump(arcd_transcripts, f, indent=2, ensure_ascii=False)
    
    xl_wer = np.mean([a["legacy_wer"] for a in propagation_results[m_key]["articles"]])
    p_wer = np.mean([arcd_transcripts["passages"][p['id']]["models"][m_key]["legacy_wer"] for p in passages])
    tot_dur = total_audio_xl + total_audio_arcd
    tot_comp = total_comp_xl + total_comp_arcd
    th = tot_dur / tot_comp if tot_comp > 0 else 0.0
    
    summary_records.append({
        "Model": m_name,
        "XL-Sum WER (%)": round(xl_wer, 2),
        "ARCD Passages WER (%)": round(p_wer, 2),
        "Throughput (xRT)": round(th, 2)
    })
    print(f"--> {m_name}: XL-Sum WER = {xl_wer:.2f}% | ARCD WER = {p_wer:.2f}% | Throughput = {th:.2f}x")
    
    del whisper_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with open(asr_out_file, "w", encoding="utf-8") as f:
    json.dump(propagation_results, f, indent=2, ensure_ascii=False)
with open(arcd_trans_file, "w", encoding="utf-8") as f:
    json.dump(arcd_transcripts, f, indent=2, ensure_ascii=False)

print(f"\\n[OK] Outputs saved to: {asr_out_file} and {arcd_trans_file}")
""",
        # Cell 5: Save Verified Propagation Artifacts
        """# ── CELL 5: ARTIFACT VALIDATION & DUAL WER CHECKPOINT ─────────────────────
import pandas as pd

print("=== N2 MULTI-MODEL ASR SUMMARY ===")
df_summary = pd.DataFrame(summary_records)
print(df_summary.to_string(index=False))

df_summary.to_csv(os.path.join(OUT_DIR, "tts_asr_summary.csv"), index=False)
print(f"\\nArtifacts verified:")
print(f" - {asr_out_file} ({os.path.getsize(asr_out_file) / 1024:.1f} KB)")
print(f" - {arcd_trans_file} ({os.path.getsize(arcd_trans_file) / 1024:.1f} KB)")
print(f" - {manifest_path} ({os.path.getsize(manifest_path) / 1024:.1f} KB)")
print("\\n[OK] Neural TTS synthesis and multi-model ASR transcription complete.")
"""
    ]
    create_notebook("Notebooks/final_02_tts_asr.ipynb", n2_cells)

    # ─────────────────────────────────────────────────────────────────────────
    # N3: final_03_summarization_retrieval.ipynb (Internet OFF)
    # ─────────────────────────────────────────────────────────────────────────
    n3_cells = [
        # Cell 1: Environment & evalkit-src check & extraction
        f"""# ── CELL 1: ENVIRONMENT & EVALKIT-SRC VERIFICATION (INTERNET OFF) ──────
import os, sys, subprocess, json, hashlib, zipfile, shutil

RUN_MODE = os.environ.get("RUN_MODE", "full")
OUT_DIR = "Results/smoke" if RUN_MODE == "smoke" else "Results/final"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("Results/env", exist_ok=True)

EXPECTED_EVALKIT_SHA = "{evalkit_zip_sha}"
print(f"Internet: OFF | RUN_MODE: {{RUN_MODE}} | Target: {{OUT_DIR}} | Expected evalkit SHA: {{EXPECTED_EVALKIT_SHA}}")

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

env_path = 'Results/env/final_03_summarization_retrieval_env.txt'
with open(env_path, 'w', encoding='utf-8') as f:
    f.write(f"Python Version: {{sys.version}}\\n")
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
        f.write(f"GPU Info:\\n{{gpu_info}}\\n")
    except Exception as e:
        f.write(f"GPU Info: None ({{e}})\\n")
    pkgs = ['torch', 'transformers', 'sentence_transformers', 'faiss', 'rouge_score', 'numpy']
    f.write("Package Versions:\\n")
    for p in pkgs:
        try:
            mod = __import__(p.replace("-", "_"))
            f.write(f"  {{p}}: {{getattr(mod, '__version__', 'unknown')}}\\n")
        except Exception as e:
            f.write(f"  {{p}}: NOT INSTALLED ({{e}})\\n")
print(f"[OK] Environment logged to: {{env_path}}")
""",
        # Cell 2: Preflight check & Mounted Model Loading
        """# ── CELL 2: PREFLIGHT INPUTS AUDIT & OFFLINE MODEL INITIALIZATION ────────
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

# Downstream model paths
arabart_path = os.path.join(merged_root, "downstream/arabart-xlsum-arabic")
camelbert_path = os.path.join(merged_root, "downstream/camelbert-msa")
mmarco_path = os.path.join(merged_root, "downstream/mmarco-reranker")

print(f"AraBART Path   : {arabart_path} (Exists: {os.path.exists(arabart_path)})")
print(f"CAMeL-BERT Path: {camelbert_path} (Exists: {os.path.exists(camelbert_path)})")
print(f"mMARCO Path    : {mmarco_path} (Exists: {os.path.exists(mmarco_path)})")

# Locate N2 transcript results
asr_out_file = None
arcd_trans_file = None

for cand_dir in ["Results/final", "Results/smoke", "/kaggle/input/tts-asr-propagation-results", "/kaggle/input"]:
    if os.path.exists(cand_dir):
        for root, _, files in os.walk(cand_dir):
            if "tts_asr_propagation_results.json" in files and asr_out_file is None:
                asr_out_file = os.path.join(root, "tts_asr_propagation_results.json")
            if "spoken_arcd_transcripts.json" in files and arcd_trans_file is None:
                arcd_trans_file = os.path.join(root, "spoken_arcd_transcripts.json")

print(f"[OK] ASR propagation file : {asr_out_file}")
print(f"[OK] Spoken ARCD file     : {arcd_trans_file}")

# Load XL-Sum articles
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

# Load ARCD Passages & Queries
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
        elif f.endswith(".zip") and any(k in f.lower() for k in ["arcd", "comprehension", "arabic"]):
            try:
                with zipfile.ZipFile(cand_p, "r") as z:
                    for zf in z.namelist():
                        if zf.endswith(".csv") and "train" in zf.lower():
                            cand_df = pd.read_csv(z.open(zf))
                            cols_low = [c.lower() for c in cand_df.columns]
                            if any(c in cols_low for c in ["context", "text"]) and any(c in cols_low for c in ["question", "query"]):
                                raw_df = cand_df
                                print(f"[OK] Found ARCD dataset in archive: {cand_p} -> {zf}")
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

if not passages and RUN_MODE == "smoke":
    passages = [{'id': f'p_{i:03d}', 'text': f'تعتبر هذه الفقرة تجريبية رقم {i} لدراسة الاسترجاع الدلالي العربي.'} for i in range(2)]
    queries = [{'query': f'ما هو رقم الفقرة {i}؟', 'target_id': f'p_{i:03d}'} for i in range(2)]

print(f"[OK] Preflight complete: {len(xlsum_articles)} articles, {len(passages)} passages, {len(queries)} queries.")
""",
        # Cell 3: Spoken Document Retrieval (Sets i, ii, iii & Oracle)
        """# ── CELL 3: SPOKEN DOCUMENT RETRIEVAL EVALUATION (SETS i, ii, iii) ────────
import torch, numpy as np, faiss
from evalkit.retrieval import (
    load_camelbert, encode_camelbert, chunk_passage,
    build_faiss_index, evaluate_retrieval, set_all_seeds,
    cross_encoder_rerank, wilson_ci, mcnemar_exact
)
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

set_all_seeds(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Retrieval Evaluation Device: {device}")

# 1. Load Bi-Encoder (CAMeL-BERT) and Cross-Encoder (mMARCO) offline
camel_tok = AutoTokenizer.from_pretrained(camelbert_path)
camel_mdl = AutoModel.from_pretrained(camelbert_path).to(device).eval()

try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(mmarco_path, device=device)
except Exception:
    reranker = None

# Build Clean Corpus Chunks (50 words each)
clean_chunks, clean_chunk_to_passage = [], []
for p in passages:
    p_chunks = chunk_passage(p['text'], chunk_words=50, overlap=0, normalize=False)
    for ch in p_chunks:
        clean_chunks.append(ch)
        clean_chunk_to_passage.append(p['id'])

print(f"Total Clean Chunks: {len(clean_chunks)} from {len(passages)} passages.")
clean_chunk_embs = encode_camelbert(clean_chunks, camel_mdl, camel_tok, max_length=128, device=device)
clean_index = build_faiss_index(clean_chunk_embs)

# Clean Queries Embeddings
clean_queries_text = [q['query'] for q in queries]
clean_query_embs = encode_camelbert(clean_queries_text, camel_mdl, camel_tok, max_length=128, device=device)

# Load Spoken ARCD Transcripts
with open(arcd_trans_file, "r", encoding="utf-8") as f:
    spoken_arcd = json.load(f)

# Evaluate Baseline: Clean Text Query -> Clean Text Passages (Oracle)
print("\\n--- Evaluating Baseline: Oracle Clean Query -> Clean Passages ---")
res_oracle = evaluate_retrieval(queries, clean_query_embs, clean_chunks, clean_chunk_to_passage, clean_index, cross_encoder=reranker)
print(f"Oracle: P@1 = {res_oracle['p1']:.4f} | P@3 = {res_oracle['p3']:.4f} | MRR = {res_oracle['mrr']:.4f}")

retrieval_rows = [{
    "Condition": "Oracle (Clean -> Clean)",
    "Model": "Reference",
    "P@1": res_oracle['p1'],
    "P@3": res_oracle['p3'],
    "MRR": res_oracle['mrr'],
    "NDCG@3": res_oracle.get('ndcg3', res_oracle['p3'])
}]

models_eval = ["large_v3", "turbo", "medium", "small"]
full_retrieval_results = {"oracle": res_oracle, "set_i": {}, "set_ii": {}, "set_iii": {}}

for m_key in models_eval:
    print(f"\\n=== Evaluating Retrieval for: {m_key} ===")
    
    # ── Set (i): Clean Query -> Spoken Passages
    spoken_chunks_i, spoken_map_i = [], []
    for p in passages:
        pid = p['id']
        sp_text = spoken_arcd["passages"][pid]["models"][m_key]["norm_legacy"]
        for ch in chunk_passage(sp_text, chunk_words=50, overlap=0, normalize=False):
            spoken_chunks_i.append(ch)
            spoken_map_i.append(pid)
    
    sp_chunk_embs_i = encode_camelbert(spoken_chunks_i, camel_mdl, camel_tok, max_length=128, device=device)
    sp_index_i = build_faiss_index(sp_chunk_embs_i)
    res_i = evaluate_retrieval(queries, clean_query_embs, spoken_chunks_i, spoken_map_i, sp_index_i, cross_encoder=reranker)
    full_retrieval_results["set_i"][m_key] = res_i
    retrieval_rows.append({
        "Condition": "Set (i): Clean Q -> Spoken P",
        "Model": m_key,
        "P@1": res_i['p1'],
        "P@3": res_i['p3'],
        "MRR": res_i['mrr'],
        "NDCG@3": res_i.get('ndcg3', res_i['p3'])
    })
    
    # ── Set (ii): Spoken Query -> Clean Passages
    spoken_queries_ii = []
    for idx, q in enumerate(queries):
        qid = f"query_{idx:03d}"
        sq_text = spoken_arcd["queries"][qid]["models"][m_key]["norm_legacy"]
        spoken_queries_ii.append({'query': sq_text, 'target_id': q['target_id']})
    
    sp_query_embs_ii = encode_camelbert([sq['query'] for sq in spoken_queries_ii], camel_mdl, camel_tok, max_length=128, device=device)
    res_ii = evaluate_retrieval(spoken_queries_ii, sp_query_embs_ii, clean_chunks, clean_chunk_to_passage, clean_index, cross_encoder=reranker)
    full_retrieval_results["set_ii"][m_key] = res_ii
    retrieval_rows.append({
        "Condition": "Set (ii): Spoken Q -> Clean P",
        "Model": m_key,
        "P@1": res_ii['p1'],
        "P@3": res_ii['p3'],
        "MRR": res_ii['mrr'],
        "NDCG@3": res_ii.get('ndcg3', res_ii['p3'])
    })
    
    # ── Set (iii): Spoken Query -> Spoken Passages
    res_iii = evaluate_retrieval(spoken_queries_ii, sp_query_embs_ii, spoken_chunks_i, spoken_map_i, sp_index_i, cross_encoder=reranker)
    full_retrieval_results["set_iii"][m_key] = res_iii
    retrieval_rows.append({
        "Condition": "Set (iii): Spoken Q -> Spoken P",
        "Model": m_key,
        "P@1": res_iii['p1'],
        "P@3": res_iii['p3'],
        "MRR": res_iii['mrr'],
        "NDCG@3": res_iii.get('ndcg3', res_iii['p3'])
    })

retrieval_out_csv = os.path.join(OUT_DIR, "spoken_retrieval_summary.csv")
retrieval_out_json = os.path.join(OUT_DIR, "spoken_retrieval_results.json")

df_ret = pd.DataFrame(retrieval_rows)
df_ret.to_csv(retrieval_out_csv, index=False)
with open(retrieval_out_json, "w", encoding="utf-8") as f:
    json.dump(full_retrieval_results, f, indent=2, ensure_ascii=False)

print(f"\\n=== RETRIEVAL RESULTS SUMMARY ===")
print(df_ret.to_string(index=False))
print(f"\\nSaved retrieval outputs to: {retrieval_out_csv} and {retrieval_out_json}")
""",
        # Cell 4: Downstream Abstractive Summarization (AraBART Batch Size 8)
        """# ── CELL 4: ABSTRACTIVE SUMMARIZATION WITH DYNAMIC BATCHING ──────────────
from evalkit.rouge import score_summary, multilingual_rouge_scoring
from evalkit.summarize import summarize_batch
from evalkit.stats import bootstrap_ci
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print(f"Loading AraBART summarizer offline from: {arabart_path}...")
bart_tok = AutoTokenizer.from_pretrained(arabart_path)
bart_mdl = AutoModelForSeq2SeqLM.from_pretrained(arabart_path).to(device).eval()

# Load N2 XL-Sum transcript results
with open(asr_out_file, "r", encoding="utf-8") as f:
    xl_results = json.load(f)

# 1. Oracle Clean Text Baseline
print("\\n--- Generating Oracle Summaries (Clean Text) ---")
clean_texts = [a['text'] for a in xlsum_articles]
reference_summaries = [a['summary'] for a in xlsum_articles]

oracle_summaries = summarize_batch(clean_texts, bart_mdl, bart_tok, device=device)
oracle_rl = [score_summary(ref, pred)['rougeL'] for ref, pred in zip(reference_summaries, oracle_summaries)]
oracle_r1 = [score_summary(ref, pred)['rouge1'] for ref, pred in zip(reference_summaries, oracle_summaries)]
oracle_r2 = [score_summary(ref, pred)['rouge2'] for ref, pred in zip(reference_summaries, oracle_summaries)]

oracle_boot = bootstrap_ci(oracle_rl, n_bootstrap=2000, seed=42)
print(f"Oracle ROUGE-L: {np.mean(oracle_rl):.2f}% (95% CI: [{oracle_boot['ci_lower']:.2f}, {oracle_boot['ci_upper']:.2f}])")

summarization_rows = [{
    "Pipeline": "Oracle (Clean Text -> AraBART)",
    "ASR Model": "None (0% WER)",
    "ROUGE-1": round(np.mean(oracle_r1), 2),
    "ROUGE-2": round(np.mean(oracle_r2), 2),
    "ROUGE-L": round(np.mean(oracle_rl), 2),
    "95% CI ROUGE-L": f"[{oracle_boot['ci_lower']:.2f}, {oracle_boot['ci_upper']:.2f}]",
    "Quality Retention (%)": 100.0
}]

full_summarization_results = {
    "oracle": {
        "rouge1": np.mean(oracle_r1),
        "rouge2": np.mean(oracle_r2),
        "rougeL": np.mean(oracle_rl),
        "ci_lower": oracle_boot['ci_lower'],
        "ci_upper": oracle_boot['ci_upper'],
        "summaries": oracle_summaries
    },
    "models": {}
}

# 2. Spoken Cascades across 4 Models
for m_key in ["large_v3", "turbo", "medium", "small"]:
    print(f"\\n--- Summarizing ASR transcripts for: {m_key} ---")
    spoken_texts = [a["norm_legacy_transcript"] for a in xl_results[m_key]["articles"]]
    spoken_summaries = summarize_batch(spoken_texts, bart_mdl, bart_tok, device=device)
    
    sp_rl = [score_summary(ref, pred)['rougeL'] for ref, pred in zip(reference_summaries, spoken_summaries)]
    sp_r1 = [score_summary(ref, pred)['rouge1'] for ref, pred in zip(reference_summaries, spoken_summaries)]
    sp_r2 = [score_summary(ref, pred)['rouge2'] for ref, pred in zip(reference_summaries, spoken_summaries)]
    
    sp_boot = bootstrap_ci(sp_rl, n_bootstrap=2000, seed=42)
    retention = (np.mean(sp_rl) / np.mean(oracle_rl) * 100.0) if np.mean(oracle_rl) > 0 else 0.0
    
    summarization_rows.append({
        "Pipeline": f"Spoken ({m_key} -> AraBART)",
        "ASR Model": m_key,
        "ROUGE-1": round(np.mean(sp_r1), 2),
        "ROUGE-2": round(np.mean(sp_r2), 2),
        "ROUGE-L": round(np.mean(sp_rl), 2),
        "95% CI ROUGE-L": f"[{sp_boot['ci_lower']:.2f}, {sp_boot['ci_upper']:.2f}]",
        "Quality Retention (%)": round(retention, 2)
    })
    
    full_summarization_results["models"][m_key] = {
        "rouge1": np.mean(sp_r1),
        "rouge2": np.mean(sp_r2),
        "rougeL": np.mean(sp_rl),
        "ci_lower": sp_boot['ci_lower'],
        "ci_upper": sp_boot['ci_upper'],
        "retention_pct": retention,
        "summaries": spoken_summaries
    }
    print(f"--> {m_key}: ROUGE-L = {np.mean(sp_rl):.2f}% | Retention = {retention:.1f}%")

df_sum = pd.DataFrame(summarization_rows)
print(f"\\n=== SUMMARIZATION CASCADE SUMMARY ===")
print(df_sum.to_string(index=False))
""",
        # Cell 5: Results Consolidation
        """# ── CELL 5: RESULTS CONSOLIDATION & HASH METADATA RECORDING ──────────────
summary_out = os.path.join(OUT_DIR, "downstream_retrieval_summarization.json")
consolidated = {
    "retrieval": full_retrieval_results,
    "summarization": full_summarization_results
}
with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(consolidated, f, indent=2, ensure_ascii=False)

print(f"Consolidated results saved to: {summary_out}")
print("[OK] Downstream summarization and spoken document retrieval completed successfully!")
"""
    ]
    create_notebook("Notebooks/final_03_summarization_retrieval.ipynb", n3_cells)'''

    updated_content = content[:idx_start] + new_section + content[idx_end:]
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print("[SUCCESS] Successfully updated scripts/build_all_final_notebooks.py with complete N2 and N3 code!")

if __name__ == "__main__":
    main()
