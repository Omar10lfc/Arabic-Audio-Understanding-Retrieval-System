import hashlib, zipfile, torch, numpy as np, pandas as pd
from transformers import AutoTokenizer, AutoModel
import faiss

with zipfile.ZipFile('Data/ARCD (Arabic Language Comprehension)-Dataset.zip', 'r') as z:
    raw_df_orig = pd.read_csv(z.open('train.csv'))

embed_id = 'CAMeL-Lab/bert-base-arabic-camelbert-msa'
embed_tokenizer = AutoTokenizer.from_pretrained(embed_id)
embed_model = AutoModel.from_pretrained(embed_id).eval()

def embed_texts(texts, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        tokens = embed_tokenizer(texts[i:i+batch_size], padding=True, truncation=True, max_length=128, return_tensors='pt')
        with torch.no_grad():
            out = embed_model(**tokens)
            mask = tokens['attention_mask'].unsqueeze(-1)
            mean_pooled = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1)
            normed = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
        all_embs.append(normed.cpu().numpy())
    return np.vstack(all_embs)

def split_into_chunks(text, max_words=50):
    words = str(text).split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)] if words else ['فارغ']

def evaluate_df(df_input, label):
    passages, queries, seen = [], [], {}
    for _, row in df_input.iterrows():
        ctx = str(row.get('context', row.get('text', ''))).strip()
        q = str(row.get('question', row.get('query', ''))).strip()
        if not ctx or not q or len(ctx) < 40 or len(q) < 5:
            continue
        if ctx not in seen:
            pid = f'p_{len(seen):03d}'
            seen[ctx] = pid
            if len(passages) < 50:
                passages.append({'id': pid, 'text': ctx})
        target_pid = seen[ctx]
        if any(p['id'] == target_pid for p in passages):
            if len(queries) < 50:
                queries.append({'query': q, 'target_id': target_pid})
        if len(passages) >= 50 and len(queries) >= 50:
            break
    
    clean_chunks, clean_c2p = [], []
    for p in passages:
        for ch in split_into_chunks(p['text'], 50):
            clean_chunks.append(ch)
            clean_c2p.append(p['id'])
            
    chunk_concat = "".join(clean_chunks)
    chunk_md5 = hashlib.md5(chunk_concat.encode('utf-8')).hexdigest()
    query_concat = "".join([q['query'] for q in queries])
    query_md5 = hashlib.md5(query_concat.encode('utf-8')).hexdigest()
    
    q_embs = embed_texts([q['query'] for q in queries])
    c_embs = embed_texts(clean_chunks)
    idx = faiss.IndexFlatIP(c_embs.shape[1])
    idx.add(c_embs)
    D, indices = idx.search(q_embs, 10)
    
    hits1 = []
    for i, q in enumerate(queries):
        target = q['target_id']
        retrieved_c = [c for c in indices[i] if c >= 0]
        p_ids = [clean_c2p[c] for c in retrieved_c]
        hits1.append(1 if (p_ids and p_ids[0] == target) else 0)
        
    p1 = np.mean(hits1)
    print(f"=== {label} ===")
    print(f"Passages: {len(passages)}, Queries: {len(queries)}, Chunks: {len(clean_chunks)}")
    print(f"Chunks MD5 : {chunk_md5}")
    print(f"Queries MD5: {query_md5}")
    print(f"Clean Bi-Encoder P@1: {p1:.4f}")
    return p1, chunk_md5, query_md5, passages, queries

print("Running Comparison...")
p1_orig, ch_md5_orig, q_md5_orig, p_orig, q_orig = evaluate_df(raw_df_orig, "Unsorted Canonical Row Order")

raw_df_sorted = raw_df_orig.sort_values(by='id').reset_index(drop=True)
p1_sort, ch_md5_sort, q_md5_sort, p_sort, q_sort = evaluate_df(raw_df_sorted, "Sorted by 'id' (from task-3885)")
