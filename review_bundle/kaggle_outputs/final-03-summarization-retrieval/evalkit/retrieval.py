import os
import random
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    faiss = None
    _HAS_FAISS = False

class SimpleFlatIPIndex:
    """
    Pure NumPy exact equivalent to faiss.IndexFlatIP for environments without FAISS.
    Guarantees bitwise/numerical parity on cosine inner-product search.
    """
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

def normalize_L2(arr: np.ndarray):
    if _HAS_FAISS and hasattr(faiss, 'normalize_L2'):
        faiss.normalize_L2(arr)
    else:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr /= np.maximum(norms, 1e-12)


def set_all_seeds(seed: int = 42):
    """
    Enforces deterministic seed across python, numpy, torch, and CUDA.
    Restores deterministic cuDNN execution.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def load_camelbert(
    model_id: str = "CAMeL-Lab/bert-base-arabic-camelbert-msa",
    revision: str = "277069fd3645fedb22b746caf38d111aadee0241",
    device: Optional[str] = None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    mdl = AutoModel.from_pretrained(model_id, revision=revision).to(device).eval()
    return mdl, tok, device

def encode_camelbert(
    texts: List[str],
    model,
    tokenizer,
    max_length: int = 128,
    batch_size: int = 32,
    device: Optional[str] = None
) -> np.ndarray:
    if not texts:
        return np.empty((0, 768), dtype=np.float32)
    if device is None:
        device = model.device if hasattr(model, 'device') else 'cpu'
    
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inp = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model(**inp)
        mask = inp["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        sum_emb = torch.sum(out.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        emb = (sum_emb / sum_mask).cpu().numpy().astype("float32")
        all_embs.append(emb)
    return np.vstack(all_embs)

def chunk_passage(
    text: str, 
    chunk_words: int = 50, 
    overlap: int = 0,
    normalize: bool = False
) -> List[str]:
    """
    Splits passage text into chunks of specified word length.
    Default parameters (chunk_words=50, overlap=0, normalize=False) reproduce the canonical 121-chunk ARCD corpus.
    """
    if normalize:
        from .normalize import normalize_arabic
        text = normalize_arabic(text)
    words = str(text).split()
    if not words:
        return ["فارغ"] if not normalize else []
    if len(words) <= chunk_words:
        return [" ".join(words)]

    chunks = []
    step = max(1, chunk_words - overlap)
    for i in range(0, len(words), step):
        chunk = words[i:i + chunk_words]
        ch_str = " ".join(chunk).strip()
        if ch_str:
            chunks.append(ch_str)
        if i + chunk_words >= len(words):
            break
    return chunks

def build_faiss_index(embeddings: np.ndarray):
    """
    L2-normalizes embeddings and builds a FAISS IndexFlatIP (or pure-NumPy equivalent) inner product index.
    """
    embs = np.array(embeddings, dtype=np.float32)
    normalize_L2(embs)
    d = embs.shape[1]
    if _HAS_FAISS and hasattr(faiss, 'IndexFlatIP'):
        index = faiss.IndexFlatIP(d)
    else:
        index = SimpleFlatIPIndex(d)
    index.add(embs)
    return index

def evaluate_retrieval(
    queries: List[Dict[str, Any]],
    query_embeddings: np.ndarray,
    chunks: List[str],
    chunk_to_passage: List[str],
    index,
    cross_encoder=None,
    top_k_bi: int = 10,
    top_k_final: int = 3
) -> Dict[str, Any]:
    """
    Evaluates retrieval pipeline with FAISS IndexFlatIP (or SimpleFlatIPIndex) and optional CrossEncoder reranker.
    Enforces stable tie-breaking for cross-encoder reranking.
    Returns:
      - Chunk-level hits1, hits3, mrrs (legacy, primary)
      - Passage-level hits3_passage, mrrs_passage (dedup) as extra fields
      - P@1 is identical in both.
    """
    q_embs = np.array(query_embeddings, dtype=np.float32)
    normalize_L2(q_embs)

    distances, indices = index.search(q_embs, min(top_k_bi, len(chunks)))

    hits1 = []
    hits3 = []          # Chunk-level (legacy, primary)
    mrrs = []           # Chunk-level (legacy, primary)
    ndcgs = []          # True NDCG@3 per query
    hits3_passage = []  # Passage-level (dedup)
    mrrs_passage = []   # Passage-level (dedup)
    per_query_details = []

    # Count relevant targets in corpus for per-query IDCG
    target_counts = {}
    for pid in chunk_to_passage:
        target_counts[str(pid)] = target_counts.get(str(pid), 0) + 1

    for q_idx, query in enumerate(queries):
        target_pid = str(query.get('target_passage_id', query.get('target_id', '')))
        q_text = str(query.get('question', query.get('query', '')))
        golden_answer = str(query.get('answer', query.get('golden_answer', '')))

        retrieved_chunk_indices = indices[q_idx]

        if cross_encoder is not None:
            # Prepare pairs with chunk index for stable tie-breaking
            pairs = [(q_text, chunks[c_idx]) for c_idx in retrieved_chunk_indices]
            raw_scores = cross_encoder.predict(pairs)
            
            # Stable sort: primary key -score (descending), secondary key chunk index (ascending)
            ranked = sorted(
                zip(retrieved_chunk_indices, raw_scores),
                key=lambda x: (-float(x[1]), int(x[0]))
            )
            candidate_chunk_indices = [c_idx for c_idx, _ in ranked]
        else:
            candidate_chunk_indices = list(retrieved_chunk_indices)

        # Chunk-level retrieved passage ids
        retrieved_pids = [str(chunk_to_passage[c_idx]) for c_idx in candidate_chunk_indices]

        # 1. P@1 (identical for both chunk and passage level)
        h1 = 1 if (len(retrieved_pids) > 0 and retrieved_pids[0] == target_pid) else 0
        hits1.append(h1)

        # 2. Chunk-level metrics (legacy, primary)
        h3 = 1 if target_pid in retrieved_pids[:top_k_final] else 0
        recip = (1.0 / (retrieved_pids.index(target_pid) + 1)) if target_pid in retrieved_pids else 0.0
        hits3.append(h3)
        mrrs.append(recip)

        # 3. True NDCG@3: DCG@3 / IDCG@3 with per-query IDCG
        # Binary relevance: 1 if chunk passage equals target_pid, else 0
        dcg3 = 0.0
        for rank_idx, pid in enumerate(retrieved_pids[:top_k_final]):
            rel = 1.0 if pid == target_pid else 0.0
            if rel > 0:
                dcg3 += rel / np.log2(rank_idx + 2.0)

        # Compute ideal DCG@3 based on availability of target in corpus
        n_avail = min(top_k_final, target_counts.get(target_pid, 0))
        idcg3 = sum(1.0 / np.log2(r + 2.0) for r in range(n_avail)) if n_avail > 0 else 1.0
        q_ndcg = (dcg3 / idcg3) if idcg3 > 0.0 else 0.0
        ndcgs.append(q_ndcg)

        # 4. Passage-level metrics (deduplicated passages)
        seen_passages = []
        for pid in retrieved_pids:
            if pid not in seen_passages:
                seen_passages.append(pid)
        
        h3_p = 1 if target_pid in seen_passages[:top_k_final] else 0
        recip_p = (1.0 / (seen_passages.index(target_pid) + 1)) if target_pid in seen_passages else 0.0
        hits3_passage.append(h3_p)
        mrrs_passage.append(recip_p)

        per_query_details.append({
            'query_index': q_idx,
            'question': q_text,
            'golden_target_id': target_pid,
            'golden_answer': golden_answer,
            'retrieved_passage_ids': retrieved_pids[:top_k_final],
            'hit1': h1,
            'hit3': h3,
            'mrr': round(recip, 4),
            'ndcg3': round(q_ndcg, 4)
        })

    n = len(queries)
    from .stats import wilson_ci
    return {
        'n_queries': n,
        'p1': round(sum(hits1) / n, 4) if n > 0 else 0.0,
        'p3': round(sum(hits3) / n, 4) if n > 0 else 0.0,
        'mrr': round(sum(mrrs) / n, 4) if n > 0 else 0.0,
        'ndcg3': round(float(np.mean(ndcgs)), 4) if n > 0 else 0.0,
        'ci_p1': wilson_ci(sum(hits1), n) if n > 0 else (0.0, 0.0),
        'ci_p3': wilson_ci(sum(hits3), n) if n > 0 else (0.0, 0.0),
        'hits1': hits1,
        'hits3': hits3,
        'mrrs': mrrs,
        'ndcgs': ndcgs,
        'p3_passage': round(sum(hits3_passage) / n, 4) if n > 0 else 0.0,
        'mrr_passage': round(sum(mrrs_passage) / n, 4) if n > 0 else 0.0,
        'ci_p3_passage': wilson_ci(sum(hits3_passage), n) if n > 0 else (0.0, 0.0),
        'hits3_passage': hits3_passage,
        'mrrs_passage': mrrs_passage,
        'queries_detail': per_query_details
    }
