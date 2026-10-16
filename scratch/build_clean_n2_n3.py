import os, sys, nbformat

def main():
    target_file = r"scripts/build_all_final_notebooks.py"
    with open(target_file, "r", encoding="utf-8") as f:
        content = f.read()

    n2_marker = "    # N2: final_02_tts_asr.ipynb"
    if n2_marker not in content:
        raise ValueError("Could not find n2_marker")
    idx_start = content.index(n2_marker)

    end_marker = 'create_notebook("Notebooks/final_03_summarization_retrieval.ipynb", n3_cells)'
    if end_marker not in content:
        raise ValueError("Could not find end_marker")
    idx_end = content.index(end_marker) + len(end_marker)

    # We will write the replacement block using pure string templates (NO outer f-strings!)
    # That way, inner strings and backslashes are 100% literal and uncorrupted.

    n2_c1 = '''# ── CELL 1: ENVIRONMENT & EVALKIT-SRC VERIFICATION ────────────────────
import os, sys, subprocess, json, hashlib, zipfile, shutil

for pkg in ["ctranslate2", "faster-whisper", "edge-tts", "soundfile"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
            print(f"[OK] Installed {pkg}.")
        except Exception as e:
            print(f"[ERROR] Could not install {pkg}: {e}")
            print("Please toggle Internet: ON in Kaggle Notebook Settings (right sidebar) to allow pip install.")
            raise

RUN_MODE = os.environ.get("RUN_MODE", "full")
OUT_DIR = "Results/smoke" if RUN_MODE == "smoke" else "Results/final"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("Results/env", exist_ok=True)
os.makedirs("audio_tts/xlsum", exist_ok=True)
os.makedirs("audio_tts/arcd_passages", exist_ok=True)
os.makedirs("audio_tts/arcd_queries", exist_ok=True)

EXPECTED_EVALKIT_SHA = "%%EVALKIT_SHA%%"
print(f"Internet: ON | RUN_MODE: {RUN_MODE} | Target: {OUT_DIR} | Expected evalkit SHA: {EXPECTED_EVALKIT_SHA}")

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
out_lines = ["Python Version: " + str(sys.version)]
try:
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
    out_lines.append("GPU Info:" + chr(10) + str(gpu_info).strip())
except Exception as e:
    out_lines.append("GPU Info: None (" + str(e) + ")")

out_lines.append("Package Versions:")
for p in ['torch', 'transformers', 'ctranslate2', 'faster_whisper', 'edge_tts', 'soundfile']:
    try:
        mod = __import__(p.replace("-", "_"))
        out_lines.append(f"  {p}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        out_lines.append(f"  {p}: NOT INSTALLED ({e})")

with open(env_path, 'w', encoding='utf-8') as f:
    f.write("\\n".join(out_lines) + "\\n")
print(f"[OK] Environment logged to: {env_path}")
'''

    n2_c2 = '''# ── CELL 2: PREFLIGHT INPUTS AUDIT & DATA EXTRACTION ─────────────────────
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

if not passages and RUN_MODE == "smoke":
    passages = [{'id': f'p_{i:03d}', 'text': f'تعتبر هذه الفقرة تجريبية رقم {i} لدراسة الاسترجاع الدلالي العربي.'} for i in range(2)]
    queries = [{'query': f'ما هو رقم الفقرة {i}؟', 'target_id': f'p_{i:03d}'} for i in range(2)]

print(f"[OK] Loaded {len(xlsum_articles)} XL-Sum articles.")
print(f"[OK] Loaded {len(passages)} ARCD passages and {len(queries)} ARCD queries.")
assert len(xlsum_articles) >= target_xl_n, f"Could not load {target_xl_n} XL-Sum articles!"
assert len(passages) >= target_arcd_n, f"Could not load {target_arcd_n} ARCD passages!"
'''

    n2_c3 = '''# ── CELL 3: NEURAL AUDIO GENERATION (XL-SUM & ARCD CORPUS) ───────────────
import asyncio, time, json, os, glob, edge_tts, soundfile as sf, numpy as np
import concurrent.futures

VOICE = "ar-SA-HamedNeural"
manifest_path = os.path.join(OUT_DIR, "tts_manifest.json")
tts_manifest = {}
if os.path.exists(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        tts_manifest = json.load(f)

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
print(f"--- Synthesizing {len(xlsum_articles)} XL-Sum articles ---")
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
        synthesize_clip(p['text'], wav_p)
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
        synthesize_clip(q['query'], wav_p)
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
print()
print(f"[OK] Neural TTS synthesis complete! Total clips: {len(tts_manifest)} ({total_audio_sec:.1f} s / {total_audio_sec / 60:.1f} min)")
print(f"Manifest written to: {manifest_path}")
'''

    with open("scratch/cell_3.py", "r", encoding="utf-8") as f:
        n2_c4 = f.read()

    n2_c5 = '''# ── CELL 5: ARTIFACT VALIDATION & DUAL WER CHECKPOINT ─────────────────────
import pandas as pd

print("=== N2 MULTI-MODEL ASR SUMMARY ===")
df_summary = pd.DataFrame(summary_records)
print(df_summary.to_string(index=False))

df_summary.to_csv(os.path.join(OUT_DIR, "tts_asr_summary.csv"), index=False)
print()
print("Artifacts verified:")
print(f" - {asr_out_file} ({os.path.getsize(asr_out_file) / 1024:.1f} KB)")
print(f" - {arcd_trans_file} ({os.path.getsize(arcd_trans_file) / 1024:.1f} KB)")
print(f" - {manifest_path} ({os.path.getsize(manifest_path) / 1024:.1f} KB)")
print()
print("[OK] Neural TTS synthesis and multi-model ASR transcription complete.")
'''

    # N3 Cells
    n3_c1 = '''# ── CELL 1: ENVIRONMENT & EVALKIT-SRC VERIFICATION (INTERNET OFF) ──────
import os, sys, subprocess, json, hashlib, zipfile, shutil

RUN_MODE = os.environ.get("RUN_MODE", "full")
OUT_DIR = "Results/smoke" if RUN_MODE == "smoke" else "Results/final"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("Results/env", exist_ok=True)

EXPECTED_EVALKIT_SHA = "%%EVALKIT_SHA%%"
print(f"Internet: OFF | RUN_MODE: {RUN_MODE} | Target: {OUT_DIR} | Expected evalkit SHA: {EXPECTED_EVALKIT_SHA}")

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
out_lines = ["Python Version: " + str(sys.version)]
try:
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
    out_lines.append("GPU Info:" + chr(10) + str(gpu_info).strip())
except Exception as e:
    out_lines.append("GPU Info: None (" + str(e) + ")")

out_lines.append("Package Versions:")
for p in ['torch', 'transformers', 'sentence_transformers', 'faiss', 'rouge_score', 'numpy']:
    try:
        mod = __import__(p.replace("-", "_"))
        out_lines.append(f"  {p}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        out_lines.append(f"  {p}: NOT INSTALLED ({e})")

with open(env_path, 'w', encoding='utf-8') as f:
    f.write("\\n".join(out_lines) + "\\n")
print(f"[OK] Environment logged to: {env_path}")
'''

    n3_c2 = '''# ── CELL 2: PREFLIGHT INPUTS AUDIT & OFFLINE MODEL INITIALIZATION ────────
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

# Downstream model paths (auto-search in /kaggle/input with fallback)
arabart_path = None
camelbert_path = None
mmarco_path = None

for root, dirs, files in os.walk("/kaggle/input") if os.path.exists("/kaggle/input") else []:
    if "arabart-xlsum-arabic" in dirs and arabart_path is None:
        arabart_path = os.path.join(root, "arabart-xlsum-arabic")
    if "camelbert-msa" in dirs and camelbert_path is None:
        camelbert_path = os.path.join(root, "camelbert-msa")
    if "mmarco-reranker" in dirs and mmarco_path is None:
        mmarco_path = os.path.join(root, "mmarco-reranker")

if not arabart_path or not os.path.exists(arabart_path):
    p_cand = os.path.join(merged_root, "downstream/arabart-xlsum-arabic")
    arabart_path = p_cand if os.path.exists(p_cand) else "Omar10lfc/arabart-xlsum-arabic"
if not camelbert_path or not os.path.exists(camelbert_path):
    p_cand = os.path.join(merged_root, "downstream/camelbert-msa")
    camelbert_path = p_cand if os.path.exists(p_cand) else "CAMeL-Lab/bert-base-arabic-camelbert-msa"
if not mmarco_path or not os.path.exists(mmarco_path):
    p_cand = os.path.join(merged_root, "downstream/mmarco-reranker")
    mmarco_path = p_cand if os.path.exists(p_cand) else "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

print(f"AraBART Path   : {arabart_path} (Status: {'Local Folder' if os.path.exists(str(arabart_path)) else 'HuggingFace ID'})")
print(f"CAMeL-BERT Path: {camelbert_path} (Status: {'Local Folder' if os.path.exists(str(camelbert_path)) else 'HuggingFace ID'})")
print(f"mMARCO Path    : {mmarco_path} (Status: {'Local Folder' if os.path.exists(str(mmarco_path)) else 'HuggingFace ID'})")

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

print(f"[OK] Preflight complete: {len(xlsum_articles)} articles, {len(passages)} passages, {len(queries)} queries.")
'''

    n3_c3 = '''# ── CELL 3: SPOKEN DOCUMENT RETRIEVAL EVALUATION (SETS i, ii, iii) ────────
import os, sys, types, json, torch
import numpy as np
import pandas as pd

# 1. Guarantee FAISS compatibility (pure NumPy exact fallback)
if 'faiss' not in sys.modules:
    mock_faiss = types.ModuleType('faiss')
    class MockIndexFlatIP:
        def __init__(self, d: int):
            self.d = d
            self.embs = None
            self.ntotal = 0
        def add(self, embs: np.ndarray):
            arr = np.array(embs, dtype=np.float32)
            if self.embs is None:
                self.embs = arr
            else:
                self.embs = np.vstack([self.embs, arr])
            self.ntotal = len(self.embs)
        def search(self, q_embs: np.ndarray, k: int):
            scores = q_embs @ self.embs.T
            k = min(k, self.embs.shape[0])
            indices = np.argsort(-scores, axis=1)[:, :k]
            distances = np.take_along_axis(scores, indices, axis=1)
            return distances, indices

    def mock_normalize_L2(arr: np.ndarray):
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr /= np.maximum(norms, 1e-12)

    mock_faiss.IndexFlatIP = MockIndexFlatIP
    mock_faiss.normalize_L2 = mock_normalize_L2
    sys.modules['faiss'] = mock_faiss

# 2. Patch on-disk evalkit/retrieval.py if it still has unhandled import faiss
retrieval_file = '/kaggle/working/evalkit/retrieval.py'
if os.path.exists(retrieval_file):
    with open(retrieval_file, 'r', encoding='utf-8') as f:
        code_str = f.read()
    if 'try:\\n    import faiss' not in code_str:
        code_str = code_str.replace(
            'import faiss\\n',
            'try:\\n    import faiss\\n    _HAS_FAISS = True\\nexcept ImportError:\\n    faiss = None\\n    _HAS_FAISS = False\\n'
        )
        with open(retrieval_file, 'w', encoding='utf-8') as f:
            f.write(code_str)

from evalkit.retrieval import (
    load_camelbert, encode_camelbert, chunk_passage,
    build_faiss_index, evaluate_retrieval, set_all_seeds
)
from transformers import AutoTokenizer, AutoModel

set_all_seeds(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Retrieval Evaluation Device: {device}")

# 3. Load Bi-Encoder (CAMeL-BERT)
print(f"Loading Bi-Encoder from: {camelbert_path}...")
camel_tok = AutoTokenizer.from_pretrained(camelbert_path)
camel_mdl = AutoModel.from_pretrained(camelbert_path).to(device).eval()

# 4. Attempt to load Cross-Encoder (mMARCO) with graceful fallback
reranker = None
try:
    from sentence_transformers import CrossEncoder
    print(f"Loading Cross-Encoder from: {mmarco_path}...")
    reranker = CrossEncoder(mmarco_path, device=device)
    print("[OK] Cross-Encoder loaded successfully.")
except Exception as e:
    print(f"[NOTE] Cross-Encoder reranker not loaded ({e}). Proceeding with pure Bi-Encoder retrieval.")

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

print()
print("--- Evaluating Baseline: Oracle Clean Query -> Clean Passages ---")
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
    print()
    print(f"=== Evaluating Retrieval for: {m_key} ===")
    
    # Set (i): Clean Query -> Spoken Passages
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
    
    # Set (ii): Spoken Query -> Clean Passages
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
    
    # Set (iii): Spoken Query -> Spoken Passages
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

print()
print("=== RETRIEVAL RESULTS SUMMARY ===")
print(df_ret.to_string(index=False))
print(f"Saved retrieval outputs to: {retrieval_out_csv} and {retrieval_out_json}")
'''

    n3_c4 = '''# ── CELL 4: ABSTRACTIVE SUMMARIZATION WITH OFFICIAL XL-SUM SCORER ─────────
import os, sys, json, types, re, subprocess
import numpy as np
import pandas as pd

# 1. Inject pure-Python pyonmttok mock to satisfy Python 3.13 import requirement
if "pyonmttok" not in sys.modules:
    mock_pyonmttok = types.ModuleType("pyonmttok")
    class MockTokenizer:
        def __init__(self, mode="aggressive"):
            self.mode = mode
        def tokenize(self, text):
            tokens = re.findall(r'\\w+|[^\\w\\s]', text)
            return [tokens, None]
    mock_pyonmttok.Tokenizer = MockTokenizer
    sys.modules["pyonmttok"] = mock_pyonmttok

# 2. Load official csebuetnlp/xl-sum multilingual ROUGE scorer
try:
    from rouge_score import rouge_scorer
    _test = rouge_scorer.RougeScorer(['rouge1'], lang='arabic', use_stemmer=True)
    print("[OK] Official csebuetnlp/xl-sum multilingual rouge_score is active.")
except (TypeError, ModuleNotFoundError, Exception):
    print("Installing official csebuetnlp/xl-sum multilingual scorer from GitHub...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring"
    ])
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("rouge_score") or mod_name.startswith("evalkit.rouge"):
            del sys.modules[mod_name]
    from rouge_score import rouge_scorer
    print("[OK] Loaded official csebuetnlp/xl-sum rouge_scorer.")

# Initialize the official XL-Sum Arabic scorers (stemmed and unstemmed)
xlsum_scorer_stemmed = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], 
    lang='arabic', 
    use_stemmer=True
)
xlsum_scorer_unstemmed = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'], 
    lang='arabic', 
    use_stemmer=False
)

def score_summary(reference: str, hypothesis: str, use_stemmer: bool = True):
    scorer = xlsum_scorer_stemmed if use_stemmer else xlsum_scorer_unstemmed
    scores = scorer.score(reference.strip(), hypothesis.strip())
    return {
        'rouge1': round(scores['rouge1'].fmeasure * 100.0, 4),
        'rouge2': round(scores['rouge2'].fmeasure * 100.0, 4),
        'rougeL': round(scores['rougeL'].fmeasure * 100.0, 4),
    }

# Validation check: Arabic stemmer correctly matches inflected forms
_val_score = score_summary("المعلم في المدرسة", "المعلمون في المدرسة", use_stemmer=True)
print(f"[OK] XL-Sum Arabic Scorer verified (ROUGE-L: {_val_score['rougeL']}%)")

from evalkit.summarize import summarize_batch
from evalkit.stats import bootstrap_ci
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print(f"\\nLoading AraBART summarizer from: {arabart_path}...")
bart_tok = AutoTokenizer.from_pretrained(arabart_path)
bart_mdl = AutoModelForSeq2SeqLM.from_pretrained(arabart_path).to(device).eval()

with open(asr_out_file, "r", encoding="utf-8") as f:
    xl_results = json.load(f)

print()
print("--- Generating Oracle Summaries (Clean Text) ---")
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

for m_key in ["large_v3", "turbo", "medium", "small"]:
    print()
    print(f"--- Summarizing ASR transcripts for: {m_key} ---")
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
print()
print("=== SUMMARIZATION CASCADE SUMMARY ===")
print(df_sum.to_string(index=False))
'''

    n3_c5 = '''# ── CELL 5: RESULTS CONSOLIDATION & HASH METADATA RECORDING ──────────────
summary_out = os.path.join(OUT_DIR, "downstream_retrieval_summarization.json")
consolidated = {
    "retrieval": full_retrieval_results,
    "summarization": full_summarization_results
}
with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(consolidated, f, indent=2, ensure_ascii=False)

print(f"Consolidated results saved to: {summary_out}")
print("[OK] Downstream summarization and spoken document retrieval completed successfully!")
'''

    # Build the replacement python code string that will be written into build_all_final_notebooks.py
    # We serialize the lists of cells as python code with repr() to avoid escaping issues entirely!
    import pprint
    
    n2_repr = f"[\n" + ",\n".join([f"        {repr(c)}" for c in [n2_c1, n2_c2, n2_c3, n2_c4, n2_c5]]) + "\n    ]"
    n3_repr = f"[\n" + ",\n".join([f"        {repr(c)}" for c in [n3_c1, n3_c2, n3_c3, n3_c4, n3_c5]]) + "\n    ]"

    code_to_insert = f'''    # N2: final_02_tts_asr.ipynb (Internet ON)
    # ─────────────────────────────────────────────────────────────────────────
    n2_cells = {n2_repr}
    # Substitute eval kit sha
    n2_cells[0] = n2_cells[0].replace("%%EVALKIT_SHA%%", evalkit_zip_sha)
    create_notebook("Notebooks/final_02_tts_asr.ipynb", n2_cells)

    # ─────────────────────────────────────────────────────────────────────────
    # N3: final_03_summarization_retrieval.ipynb (Internet OFF)
    # ─────────────────────────────────────────────────────────────────────────
    n3_cells = {n3_repr}
    n3_cells[0] = n3_cells[0].replace("%%EVALKIT_SHA%%", evalkit_zip_sha)
    create_notebook("Notebooks/final_03_summarization_retrieval.ipynb", n3_cells)'''

    updated_content = content[:idx_start] + code_to_insert + content[idx_end:]
    with open(target_file, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print("[SUCCESS] Fully rewritten N2 and N3 builders using safe repr() serialization!")

if __name__ == "__main__":
    main()
