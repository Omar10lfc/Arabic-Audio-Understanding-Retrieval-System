# ── CELL 3: SPOKEN DOCUMENT RETRIEVAL EVALUATION (SETS i, ii, iii) ────────
import torch, numpy as np
try:
    import faiss
except ImportError:
    faiss = None

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