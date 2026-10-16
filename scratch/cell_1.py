# ── CELL 2: PREFLIGHT INPUTS AUDIT & DATA EXTRACTION ─────────────────────
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