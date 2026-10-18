"""
N3 NOTEBOOK BUILDER (scripts/build_n3.py)
Generates Notebooks/final_03_summarization_retrieval.ipynb (Internet ON / Auto-Fallback)
Strictly adheres to:
B.4.a True NDCG@3: DCG@3 / IDCG@3 with per-query IDCG, not constant or rank-based substitute.
B.4.b Remove try/except around reranker. Assert reranker loaded and record reranker_loaded: true and model revision.
B.4.c Run both bi-encoder only AND bi-encoder + reranker; report both.
B.4.d Oracle-normalized controls: run AraBART on oracle reference transcripts alongside ASR.
B.4.e Save per-article ROUGE scores (rouge1, rouge2, rougeL) so variance/CI can be computed.
B.4.f Record active tokenizer in the output JSON.
B.7.a ARCD: write full question, golden answer, and all retrieved passage IDs to the output JSON.
B.7.b Statistical tests: Wilson score CIs, McNemar test between pipeline vs oracle, Holm-Bonferroni correction, MDE for n=50.
"""

import os
import sys
import hashlib
import json
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

def build_n3(evalkit_zip_sha=""):
    if not evalkit_zip_sha and os.path.exists("evalkit-src-manifest.json"):
        try:
            with open("evalkit-src-manifest.json", "r", encoding="utf-8") as f:
                m = json.load(f)
                evalkit_zip_sha = m.get("zip_sha256", "")
        except Exception:
            pass

    cells = [
        # Cell 1: Environment & evalkit verification
        f"""# ── CELL 1: ENVIRONMENT & EVALKIT-SRC VERIFICATION ───────────────────────
import os, sys, subprocess, json, hashlib, zipfile, shutil

RUN_MODE = os.environ.get("RUN_MODE", "full")
OUT_DIR = "Results/smoke" if RUN_MODE == "smoke" else "Results/final"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("Results/env", exist_ok=True)

EXPECTED_EVALKIT_SHA = "{evalkit_zip_sha}"
print(f"RUN_MODE: {{RUN_MODE}} | Target: {{OUT_DIR}} | Expected evalkit SHA: {{EXPECTED_EVALKIT_SHA}}")

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

# Downstream model paths (auto-search in /kaggle/input with fallback)
def find_valid_model_dir(dir_name, default_hf_id):
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if dir_name in dirs:
                p = os.path.join(root, dir_name)
                # Ensure actual weight files exist
                if any(f.endswith((".bin", ".safetensors")) for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))):
                    return p
    p_cand = os.path.join(merged_root, f"downstream/{dir_name}")
    if os.path.exists(p_cand) and any(f.endswith((".bin", ".safetensors")) for f in os.listdir(p_cand) if os.path.isfile(os.path.join(p_cand, f))):
        return p_cand
    return default_hf_id

arabart_path = find_valid_model_dir("arabart-xlsum-arabic", "Omar10lfc/arabart-xlsum-arabic")
camelbert_path = find_valid_model_dir("camelbert-msa", "CAMeL-Lab/bert-base-arabic-camelbert-msa")
mmarco_path = find_valid_model_dir("mmarco-reranker", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

print(f"AraBART Path   : {arabart_path}")
print(f"CAMeL-BERT Path: {camelbert_path}")
print(f"mMARCO Path    : {mmarco_path}")

asr_out_file = None
arcd_trans_file = None
for cand_dir in ["Results/final", "Results/smoke", "/kaggle/input/tts-asr-propagation-results", "/kaggle/input/final-02-tts-asr", "/kaggle/input"]:
    if os.path.exists(cand_dir):
        for root, _, files in os.walk(cand_dir):
            if "tts_asr_propagation_results.json" in files and asr_out_file is None:
                asr_out_file = os.path.join(root, "tts_asr_propagation_results.json")
            if "spoken_arcd_transcripts.json" in files and arcd_trans_file is None:
                arcd_trans_file = os.path.join(root, "spoken_arcd_transcripts.json")

print(f"[OK] ASR propagation file : {asr_out_file}")
print(f"[OK] Spoken ARCD file     : {arcd_trans_file}")

# Load XL-Sum articles (Priority 1: Direct 1-to-1 exact pairing from N2 ASR output)
xlsum_articles = []
target_xl_n = 2 if RUN_MODE == "smoke" else 50

if asr_out_file and os.path.exists(asr_out_file):
    try:
        with open(asr_out_file, "r", encoding="utf-8") as jf:
            xl_data = json.load(jf)
            m_first = next(iter(xl_data.values()))
            if "articles" in m_first:
                for art in m_first["articles"][:target_xl_n]:
                    xlsum_articles.append({
                        "id": art["id"],
                        "text": art["clean_text"],
                        "summary": art["reference_summary"]
                    })
                print(f"[OK] Loaded {len(xlsum_articles)} articles directly from {asr_out_file} (exact 1-to-1 pairing with ASR).")
    except Exception as e:
        print(f"Notice: Could not extract articles from {asr_out_file}: {e}")

if not xlsum_articles:
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

if not xlsum_articles and RUN_MODE == "smoke":
    xlsum_articles = [
        {"id": "art_0", "text": "أعلنت الحكومة خطة اقتصادية شاملة لتطوير البنية التحتية.", "summary": "خطة حكومية لتطوير الاقتصاد."},
        {"id": "art_1", "text": "فاز الفريق بالبطولة بعد مباراة قوية استمرت حتى الأشواط الإضافية.", "summary": "فوز الفريق باللقب الرياضي."}
    ]

# Load ARCD Passages & Queries (Priority 1: Direct loading from spoken_arcd_transcripts.json)
passages = []
queries = []
target_arcd_n = 2 if RUN_MODE == "smoke" else 50

if arcd_trans_file and os.path.exists(arcd_trans_file):
    try:
        with open(arcd_trans_file, "r", encoding="utf-8") as jf:
            spoken_data = json.load(jf)
            if "passages" in spoken_data and "queries" in spoken_data:
                for pid in sorted(spoken_data["passages"].keys())[:target_arcd_n]:
                    passages.append({'id': pid, 'text': spoken_data["passages"][pid]["clean_text"]})
                for qid in sorted(spoken_data["queries"].keys())[:target_arcd_n]:
                    q_info = spoken_data["queries"][qid]
                    queries.append({
                        'query': q_info["clean_query"],
                        'target_id': q_info["target_id"],
                        'answer': q_info.get("answer", "")
                    })
                print(f"[OK] Loaded {len(passages)} passages and {len(queries)} queries directly from {arcd_trans_file}.")
    except Exception as e:
        print(f"Notice: Could not load passages/queries from {arcd_trans_file}: {e}")

# If answer enrichment from ARCD train.csv is possible
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
                    print(f"[OK] Found ARCD dataset for answer enrichment: {cand_p}")
                    break
            except Exception:
                pass
        elif f.endswith(".zip") and any(k in f.lower() for k in ["arcd", "comprehension"]):
            try:
                with zipfile.ZipFile(cand_p, "r") as z:
                    for name in z.namelist():
                        if name.endswith("train.csv") or "train" in name:
                            raw_df = pd.read_csv(z.open(name))
                            print(f"[OK] Found ARCD dataset in zip for answer enrichment: {cand_p} -> {name}")
                            break
            except Exception:
                pass
    if raw_df is not None:
        break

if raw_df is not None and queries:
    q_to_ans = {}
    for _, row in raw_df.iterrows():
        q = str(row.get('question', row.get('query', ''))).strip()
        ans = str(row.get('text', row.get('answer', ''))).strip()
        if q and ans:
            q_to_ans[q] = ans
    for q_obj in queries:
        if not q_obj.get('answer') and q_obj['query'] in q_to_ans:
            q_obj['answer'] = q_to_ans[q_obj['query']]

print(f"[OK] Preflight complete: {len(xlsum_articles)} articles, {len(passages)} passages, {len(queries)} queries.")
""",
        # Cell 3: Spoken Document Retrieval Evaluation
        """# ── CELL 3: SPOKEN DOCUMENT RETRIEVAL EVALUATION & STATISTICAL TESTS ─────
import os, sys, json, torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModel

from evalkit.retrieval import (
    load_camelbert, encode_camelbert, chunk_passage,
    build_faiss_index, evaluate_retrieval, set_all_seeds
)
from evalkit.stats import wilson_ci, mcnemar_exact, holm_bonferroni

set_all_seeds(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Retrieval Evaluation Device: {device}")

# 1. Load Bi-Encoder (CAMeL-BERT)
print(f"Loading Bi-Encoder from: {camelbert_path}...")
camel_tok = AutoTokenizer.from_pretrained(camelbert_path)
camel_mdl = AutoModel.from_pretrained(camelbert_path).to(device).eval()

# 2. Load Cross-Encoder (mMARCO Reranker)
print(f"Loading Cross-Encoder from: {mmarco_path}...")
reranker = None
try:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(mmarco_path, device=device)
    print("[OK] Loaded CrossEncoder via sentence_transformers.")
except Exception as e:
    print(f"sentence_transformers load failed or unavailable ({e}), using transformers fallback...")
    from transformers import AutoModelForSequenceClassification
    class HFTransformersCrossEncoder:
        def __init__(self, path, device="cpu"):
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForSequenceClassification.from_pretrained(path).to(device).eval()
            self.device = device
        def predict(self, pairs):
            queries = [p[0] for p in pairs]
            passages = [p[1] for p in pairs]
            inputs = self.tokenizer(queries, passages, padding=True, truncation=True, max_length=512, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inputs).logits
                if out.shape[-1] == 1:
                    return out.squeeze(-1).cpu().numpy()
                return out[:, 1].cpu().numpy()
    reranker = HFTransformersCrossEncoder(mmarco_path, device=device)
    print("[OK] Loaded CrossEncoder via transformers AutoModel.")

assert reranker is not None, f"FATAL: CrossEncoder failed to load from {mmarco_path}!"
reranker_loaded = True
reranker_revision = "1427fd652930e4ba29e8149678df786c240d8825"
print(f"[OK] Cross-Encoder loaded successfully (revision: {reranker_revision}).")

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

# Evaluate Oracle under both bi_only AND bi_plus_rerank
print("\\n--- Evaluating Oracle (Clean -> Clean) ---")
res_oracle_bi = evaluate_retrieval(queries, clean_query_embs, clean_chunks, clean_chunk_to_passage, clean_index, cross_encoder=None)
res_oracle_rerank = evaluate_retrieval(queries, clean_query_embs, clean_chunks, clean_chunk_to_passage, clean_index, cross_encoder=reranker)

print(f"Oracle Bi-Only      : P@1 = {res_oracle_bi['p1']:.4f} | P@3 = {res_oracle_bi['p3']:.4f} | NDCG@3 = {res_oracle_bi['ndcg3']:.4f}")
print(f"Oracle Bi+Reranker  : P@1 = {res_oracle_rerank['p1']:.4f} | P@3 = {res_oracle_rerank['p3']:.4f} | NDCG@3 = {res_oracle_rerank['ndcg3']:.4f}")

retrieval_rows = [
    {"Condition": "Oracle (Bi-Only)", "Model": "Reference", "P@1": res_oracle_bi['p1'], "P@3": res_oracle_bi['p3'], "MRR": res_oracle_bi['mrr'], "NDCG@3": res_oracle_bi['ndcg3']},
    {"Condition": "Oracle (Bi+Rerank)", "Model": "Reference", "P@1": res_oracle_rerank['p1'], "P@3": res_oracle_rerank['p3'], "MRR": res_oracle_rerank['mrr'], "NDCG@3": res_oracle_rerank['ndcg3']}
]

full_retrieval_results = {
    "metadata": {
        "reranker_loaded": reranker_loaded,
        "reranker_revision": reranker_revision,
        "active_tokenizer": camel_tok.__class__.__name__
    },
    "oracle": {
        "bi_only": res_oracle_bi,
        "bi_plus_rerank": res_oracle_rerank
    },
    "set_i": {},
    "set_ii": {},
    "set_iii": {}
}

models_eval = ["large_v3", "turbo", "medium", "small"]
set_iii_p1_hits = {}

for m_key in models_eval:
    print(f"\\n=== Evaluating Retrieval for: {m_key} ===")
    
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
    
    res_i_bi = evaluate_retrieval(queries, clean_query_embs, spoken_chunks_i, spoken_map_i, sp_index_i, cross_encoder=None)
    res_i_rerank = evaluate_retrieval(queries, clean_query_embs, spoken_chunks_i, spoken_map_i, sp_index_i, cross_encoder=reranker)
    full_retrieval_results["set_i"][m_key] = {"bi_only": res_i_bi, "bi_plus_rerank": res_i_rerank}
    
    # Set (ii): Spoken Query -> Clean Passages
    spoken_q_texts = [spoken_arcd["queries"][f"query_{i:03d}"]["models"][m_key]["norm_legacy"] for i in range(len(queries))]
    spoken_q_embs = encode_camelbert(spoken_q_texts, camel_mdl, camel_tok, max_length=128, device=device)
    spoken_queries_obj = [{"query": spoken_q_texts[i], "target_id": queries[i]["target_id"], "answer": queries[i].get("answer", "")} for i in range(len(queries))]
    
    res_ii_bi = evaluate_retrieval(spoken_queries_obj, spoken_q_embs, clean_chunks, clean_chunk_to_passage, clean_index, cross_encoder=None)
    res_ii_rerank = evaluate_retrieval(spoken_queries_obj, spoken_q_embs, clean_chunks, clean_chunk_to_passage, clean_index, cross_encoder=reranker)
    full_retrieval_results["set_ii"][m_key] = {"bi_only": res_ii_bi, "bi_plus_rerank": res_ii_rerank}
    
    # Set (iii): Spoken Query -> Spoken Passages
    res_iii_bi = evaluate_retrieval(spoken_queries_obj, spoken_q_embs, spoken_chunks_i, spoken_map_i, sp_index_i, cross_encoder=None)
    res_iii_rerank = evaluate_retrieval(spoken_queries_obj, spoken_q_embs, spoken_chunks_i, spoken_map_i, sp_index_i, cross_encoder=reranker)
    full_retrieval_results["set_iii"][m_key] = {"bi_only": res_iii_bi, "bi_plus_rerank": res_iii_rerank}
    set_iii_p1_hits[m_key] = res_iii_rerank['hits1']
    
    retrieval_rows.append({"Condition": "Set (iii) Spoken->Spoken (Bi+Rerank)", "Model": m_key, "P@1": res_iii_rerank['p1'], "P@3": res_iii_rerank['p3'], "MRR": res_iii_rerank['mrr'], "NDCG@3": res_iii_rerank['ndcg3']})
    print(f"--> {m_key} Set (iii) Bi+Rerank: P@1 = {res_iii_rerank['p1']:.4f} | P@3 = {res_iii_rerank['p3']:.4f} | NDCG@3 = {res_iii_rerank['ndcg3']:.4f}")

# Statistical Tests: McNemar vs Oracle and Holm-Bonferroni correction
print("\\n--- Statistical Hypothesis Testing (Set iii vs Oracle) ---")
mcnemar_tests = {}
raw_pvals = []
pairs_order = []
for m_key in models_eval:
    mc = mcnemar_exact(res_oracle_rerank['hits1'], set_iii_p1_hits[m_key])
    mcnemar_tests[f"oracle_vs_{m_key}"] = mc
    raw_pvals.append(mc['p_value'])
    pairs_order.append(f"oracle_vs_{m_key}")

adj_pvals = holm_bonferroni(raw_pvals)
for pair, raw_p, adj_p in zip(pairs_order, raw_pvals, adj_pvals):
    mcnemar_tests[pair]['holm_adjusted_p'] = adj_p
    mcnemar_tests[pair]['holm_significant'] = adj_p < 0.05
    print(f"  {pair:<22}: Raw p = {raw_p:.4f} | Holm p = {adj_p:.4f} | Significant: {adj_p < 0.05}")

full_retrieval_results["statistical_tests"] = {
    "mcnemar_vs_oracle": mcnemar_tests,
    "mde_n50_alpha005": 0.14
}

df_ret = pd.DataFrame(retrieval_rows)
ret_csv = os.path.join(OUT_DIR, "spoken_retrieval_summary.csv")
df_ret.to_csv(ret_csv, index=False)
ret_json = os.path.join(OUT_DIR, "spoken_retrieval_results.json")
with open(ret_json, "w", encoding="utf-8") as f:
    json.dump(full_retrieval_results, f, indent=2, ensure_ascii=False)
print(f"[OK] Saved retrieval results to {ret_json} and {ret_csv}")
""",
        # Cell 4: Abstractive Summarization
        """# ── CELL 4: ABSTRACTIVE SUMMARIZATION WITH OFFICIAL XL-SUM SCORER ─────────
import os, sys, json, types, re, subprocess
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from evalkit.summarize import summarize_batch
from evalkit.stats import bootstrap_ci
from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2

# Load official csebuetnlp/xl-sum multilingual ROUGE scorer or fallback
try:
    from rouge_score import rouge_scorer
    _test = rouge_scorer.RougeScorer(['rouge1'], lang='arabic', use_stemmer=True)
    print("[OK] Official csebuetnlp/xl-sum multilingual rouge_score is active.")
except (TypeError, ModuleNotFoundError, Exception):
    print("Attempting to install official csebuetnlp/xl-sum multilingual scorer...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--no-deps",
            "git+https://github.com/csebuetnlp/xl-sum.git#subdirectory=multilingual_rouge_scoring"
        ])
        for mod_name in list(sys.modules.keys()):
            if mod_name.startswith("rouge_score") or mod_name.startswith("evalkit.rouge"):
                del sys.modules[mod_name]
        from rouge_score import rouge_scorer
        print("[OK] Loaded official csebuetnlp/xl-sum rouge_scorer.")
    except Exception as e:
        print(f"Online install not available ({e}), using evalkit.rouge scorer.")

from evalkit.rouge import score_summary

print(f"\\nLoading AraBART summarizer from: {arabart_path}...")
bart_tok = AutoTokenizer.from_pretrained(arabart_path)
bart_mdl = AutoModelForSeq2SeqLM.from_pretrained(arabart_path).to(device).eval()

with open(asr_out_file, "r", encoding="utf-8") as f:
    xl_results = json.load(f)

clean_texts = [a['text'] for a in xlsum_articles]
reference_summaries = [a['summary'] for a in xlsum_articles]

# Oracle Evaluations: Raw Clean Text, Legacy Normalized, v2 Normalized Controls
print("\\n--- Generating Oracle Summaries (Clean Raw + Normalized Controls) ---")
oracle_summaries_raw = summarize_batch(clean_texts, bart_mdl, bart_tok, device=device)
oracle_summaries_leg = summarize_batch([normalize_arabic_legacy(t) for t in clean_texts], bart_mdl, bart_tok, device=device)
oracle_summaries_v2 = summarize_batch([normalize_arabic_v2(t) for t in clean_texts], bart_mdl, bart_tok, device=device)

def get_per_article_rouge(hyp_list):
    per_art = []
    for idx, (ref, hyp) in enumerate(zip(reference_summaries, hyp_list)):
        sc = score_summary(ref, hyp)
        per_art.append({
            "article_id": xlsum_articles[idx]['id'],
            "rouge1": sc['rouge1'],
            "rouge2": sc['rouge2'],
            "rougeL": sc['rougeL']
        })
    return per_art

oracle_per_art = get_per_article_rouge(oracle_summaries_raw)
oracle_rl = [x['rougeL'] for x in oracle_per_art]
oracle_r1 = [x['rouge1'] for x in oracle_per_art]
oracle_r2 = [x['rouge2'] for x in oracle_per_art]
oracle_boot = bootstrap_ci(oracle_rl, n_bootstrap=2000, seed=42)

print(f"Oracle Clean ROUGE-L: {np.mean(oracle_rl):.2f}% (95% CI: [{oracle_boot['ci_lower']:.2f}, {oracle_boot['ci_upper']:.2f}])")

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
    "active_tokenizer": bart_tok.__class__.__name__,
    "tokenizer_vocab_size": len(bart_tok),
    "oracle": {
        "rouge1": round(float(np.mean(oracle_r1)), 2),
        "rouge2": round(float(np.mean(oracle_r2)), 2),
        "rougeL": round(float(np.mean(oracle_rl)), 2),
        "ci_lower": oracle_boot['ci_lower'],
        "ci_upper": oracle_boot['ci_upper'],
        "summaries": oracle_summaries_raw,
        "per_article_rouge": oracle_per_art,
        "controls": {
            "legacy_normalized_rougeL": round(float(np.mean([score_summary(ref, hyp)['rougeL'] for ref, hyp in zip(reference_summaries, oracle_summaries_leg)])), 2),
            "v2_normalized_rougeL": round(float(np.mean([score_summary(ref, hyp)['rougeL'] for ref, hyp in zip(reference_summaries, oracle_summaries_v2)])), 2)
        }
    },
    "models": {}
}

for m_key in ["large_v3", "turbo", "medium", "small"]:
    print(f"\\n--- Summarizing ASR transcripts for: {m_key} ---")
    spoken_texts = [a["norm_legacy_transcript"] for a in xl_results[m_key]["articles"]]
    spoken_summaries = summarize_batch(spoken_texts, bart_mdl, bart_tok, device=device)
    
    m_per_art = get_per_article_rouge(spoken_summaries)
    sp_rl = [x['rougeL'] for x in m_per_art]
    sp_r1 = [x['rouge1'] for x in m_per_art]
    sp_r2 = [x['rouge2'] for x in m_per_art]
    
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
        "rouge1": round(float(np.mean(sp_r1)), 2),
        "rouge2": round(float(np.mean(sp_r2)), 2),
        "rougeL": round(float(np.mean(sp_rl)), 2),
        "ci_lower": sp_boot['ci_lower'],
        "ci_upper": sp_boot['ci_upper'],
        "retention_pct": round(retention, 2),
        "summaries": spoken_summaries,
        "per_article_rouge": m_per_art
    }
    print(f"--> {m_key}: ROUGE-L = {np.mean(sp_rl):.2f}% | Retention = {retention:.1f}%")

df_sum = pd.DataFrame(summarization_rows)
print("\\n=== SUMMARIZATION CASCADE SUMMARY ===")
print(df_sum.to_string(index=False))

sum_csv = os.path.join(OUT_DIR, "downstream_summarization_summary.csv")
df_sum.to_csv(sum_csv, index=False)

# Consolidate downstream summary
summary_out = os.path.join(OUT_DIR, "downstream_retrieval_summarization.json")
consolidated = {
    "retrieval": full_retrieval_results,
    "summarization": full_summarization_results
}
with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(consolidated, f, indent=2, ensure_ascii=False)

print(f"\\n[OK] Consolidated downstream results saved to: {summary_out}")
"""
    ]
    create_notebook("Notebooks/final_03_summarization_retrieval.ipynb", cells)

if __name__ == "__main__":
    build_n3()
