import os, sys, time, zipfile, random, math
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder
import faiss
from scipy.stats import binomtest

sys.stdout.reconfigure(encoding='utf-8')

# 1. Deterministic Seeding Function
def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 2. Load ARCD
with zipfile.ZipFile('Data/ARCD (Arabic Language Comprehension)-Dataset.zip', 'r') as z:
    raw_df = pd.read_csv(z.open('train.csv'))

passages = []
queries = []
seen_contexts = {}

for _, row in raw_df.iterrows():
    ctx = str(row.get('context', row.get('text', ''))).strip()
    q = str(row.get('question', row.get('query', ''))).strip()
    if not ctx or not q or len(ctx) < 40 or len(q) < 5:
        continue
    if ctx not in seen_contexts:
        pid = f'p_{len(seen_contexts):03d}'
        seen_contexts[ctx] = pid
        if len(passages) < 50:
            passages.append({'id': pid, 'text': ctx})
    target_pid = seen_contexts[ctx]
    if any(p['id'] == target_pid for p in passages):
        if len(queries) < 50:
            queries.append({'query': q, 'target_id': target_pid})
    if len(passages) >= 50 and len(queries) >= 50:
        break

print(f'Loaded {len(passages)} passages and {len(queries)} queries.')

# 3. Chunking
def split_into_chunks(text, max_words=50):
    words = str(text).split()
    if not words:
        return ["فارغ"]
    chunks = [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    return [c for c in chunks if c.strip()]

clean_chunks = []
clean_chunk_to_passage = []
for p in passages:
    for ch in split_into_chunks(p['text'], 50):
        clean_chunks.append(ch)
        clean_chunk_to_passage.append(p['id'])

print(f'Total clean chunks: {len(clean_chunks)}')

# Models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Loading models on {device} (local_files_only=True)...')
embed_id = 'CAMeL-Lab/bert-base-arabic-camelbert-msa'
embed_tokenizer = AutoTokenizer.from_pretrained(embed_id, local_files_only=True)
embed_model = AutoModel.from_pretrained(embed_id, local_files_only=True).to(device).eval()

reranker_id = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
reranker = CrossEncoder(reranker_id, device=device, local_files_only=True)

def embed_texts(texts, batch_size=32):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        tokens = embed_tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        with torch.no_grad():
            out = embed_model(**tokens)
            mask = tokens['attention_mask'].unsqueeze(-1)
            mean_pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            normed = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        all_embs.append(normed.cpu().numpy())
    return np.vstack(all_embs)

def evaluate_retrieval(query_list, chunks, chunk_to_passage, faiss_idx, use_rerank=False, top_k_retrieve=10):
    q_embs = embed_texts([q['query'] for q in query_list])
    D, all_indices = faiss_idx.search(q_embs, min(len(chunks), top_k_retrieve))

    hits1, hits3 = [], []
    mrrs = []

    for q_idx, q in enumerate(query_list):
        retrieved_chunk_ids = [c for c in all_indices[q_idx].tolist() if c >= 0]
        target_pid = q['target_id']

        if not use_rerank:
            candidate_passage_ids = [chunk_to_passage[c] for c in retrieved_chunk_ids]
            if target_pid in candidate_passage_ids:
                rank = candidate_passage_ids.index(target_pid) + 1
                mrrs.append(1.0 / rank)
                hits1.append(1 if rank == 1 else 0)
                hits3.append(1 if rank <= 3 else 0)
            else:
                mrrs.append(0.0)
                hits1.append(0)
                hits3.append(0)
        else:
            candidate_texts = [chunks[c] for c in retrieved_chunk_ids]
            cross_inp = [[q['query'], t] for t in candidate_texts]

            if cross_inp:
                cross_scores = reranker.predict(cross_inp)
                candidate_passage_ids = [chunk_to_passage[c] for c in retrieved_chunk_ids]
                # Tie-breaking with index preserved
                paired = list(zip(cross_scores, candidate_passage_ids, range(len(candidate_passage_ids))))
                paired.sort(key=lambda x: (x[0], -x[2]), reverse=True)
                reranked_passage_ids = [p_id for _, p_id, _ in paired]

                if target_pid in reranked_passage_ids:
                    rank = reranked_passage_ids.index(target_pid) + 1
                    mrrs.append(1.0 / rank)
                    hits1.append(1 if rank == 1 else 0)
                    hits3.append(1 if rank <= 3 else 0)
                else:
                    mrrs.append(0.0)
                    hits1.append(0)
                    hits3.append(0)
            else:
                mrrs.append(0.0)
                hits1.append(0)
                hits3.append(0)

    return {
        'P@1': round(float(np.mean(hits1)), 4),
        'P@3': round(float(np.mean(hits3)), 4),
        'MRR': round(float(np.mean(mrrs)), 4),
        'hits1': hits1,
        'hits3': hits3,
        'mrrs': mrrs,
    }

def run_experiment(seed=42):
    set_all_seeds(seed)
    clean_embs = embed_texts(clean_chunks)
    clean_idx = faiss.IndexFlatIP(clean_embs.shape[1])
    clean_idx.add(clean_embs)
    
    bi_res = evaluate_retrieval(queries, clean_chunks, clean_chunk_to_passage, clean_idx, use_rerank=False)
    rr_res = evaluate_retrieval(queries, clean_chunks, clean_chunk_to_passage, clean_idx, use_rerank=True)
    return bi_res, rr_res

print("\n--- Running Test 1 (Seed 42, Run A) ---")
b1, r1 = run_experiment(42)
print(f"Run A: Bi-Encoder P@1={b1['P@1']:.4f}, P@3={b1['P@3']:.4f}, MRR={b1['MRR']:.4f}")
print(f"Run A: Re-Rank    P@1={r1['P@1']:.4f}, P@3={r1['P@3']:.4f}, MRR={r1['MRR']:.4f}")

print("\n--- Running Test 2 (Seed 42, Run B - Identical Seed) ---")
b2, r2 = run_experiment(42)
print(f"Run B: Bi-Encoder P@1={b2['P@1']:.4f}, P@3={b2['P@3']:.4f}, MRR={b2['MRR']:.4f}")
print(f"Run B: Re-Rank    P@1={r2['P@1']:.4f}, P@3={r2['P@3']:.4f}, MRR={r2['MRR']:.4f}")

print("\nIdentical Run A vs Run B:")
print("Bi-Encoder exact match:", b1['hits1'] == b2['hits1'] and b1['mrrs'] == b2['mrrs'])
print("Re-Rank exact match   :", r1['hits1'] == r2['hits1'] and r1['mrrs'] == r2['mrrs'])
