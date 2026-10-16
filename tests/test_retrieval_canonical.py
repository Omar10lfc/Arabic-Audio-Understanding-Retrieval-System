import os
import zipfile
import hashlib
import pytest
import pandas as pd
import numpy as np
import faiss
from evalkit.retrieval import (
    chunk_passage, build_faiss_index, evaluate_retrieval,
    load_camelbert, encode_camelbert, set_all_seeds
)

@pytest.mark.slow
def test_canonical_arcd_corpus_and_biencoder_p1():
    set_all_seeds(42)
    arcd_zip = 'Data/ARCD (Arabic Language Comprehension)-Dataset.zip'
    assert os.path.exists(arcd_zip), f"Missing ARCD dataset at {arcd_zip}"
    
    with zipfile.ZipFile(arcd_zip, 'r') as z:
        df = pd.read_csv(z.open('train.csv'))

    seen, passages, queries = {}, [], []
    for _, row in df.iterrows():
        ctx = str(row.get('context', '')).strip()
        q = str(row.get('question', '')).strip()
        if not ctx or not q or len(ctx) < 40 or len(q) < 5:
            continue
        if ctx not in seen:
            pid = f'p_{len(seen):03d}'
            seen[ctx] = pid
            if len(passages) < 50:
                passages.append({'id': pid, 'text': ctx})
        t_id = seen[ctx]
        if any(p['id'] == t_id for p in passages):
            if len(queries) < 50:
                queries.append({'query': q, 'target_id': t_id})
        if len(passages) >= 50 and len(queries) >= 50:
            break

    assert len(passages) == 50, f"Expected 50 passages, got {len(passages)}"
    assert len(queries) == 50, f"Expected 50 queries, got {len(queries)}"

    # Generate chunks using default chunk_passage (chunk_words=50, overlap=0, normalize=False)
    chunks, chunk_to_passage = [], []
    for p in passages:
        p_chunks = chunk_passage(p['text'], chunk_words=50, overlap=0, normalize=False)
        for ch in p_chunks:
            chunks.append(ch)
            chunk_to_passage.append(p['id'])

    assert len(chunks) == 121, f"Expected 121 chunks, got {len(chunks)}"

    chunks_md5 = hashlib.md5("".join(chunks).encode('utf-8')).hexdigest()
    queries_md5 = hashlib.md5("".join([q['query'] for q in queries]).encode('utf-8')).hexdigest()

    assert chunks_md5 == "af460da3c4ef72770409f7c538067196", f"Corpus MD5 mismatch: {chunks_md5}"
    assert queries_md5 == "6af4c515785a4a6b7e80ef1e45f9cf4c", f"Queries MD5 mismatch: {queries_md5}"

    # Verify Bi-Encoder P@1 = 0.5000 on CPU
    mdl, tok, dev = load_camelbert(device='cpu')
    chunk_embs = encode_camelbert(chunks, mdl, tok, max_length=128, device='cpu')
    query_embs = encode_camelbert([q['query'] for q in queries], mdl, tok, max_length=128, device='cpu')

    index = build_faiss_index(chunk_embs)
    res = evaluate_retrieval(queries, query_embs, chunks, chunk_to_passage, index)

    assert res['p1'] == 0.5000, f"Expected clean Bi-Encoder P@1 == 0.5000, got {res['p1']}"
    assert res['p3'] == 0.6600, f"Expected chunk-level P@3 == 0.6600, got {res['p3']}"
    assert res['p3_passage'] == 0.6600, f"Expected passage-level P@3 == 0.6600, got {res['p3_passage']}"
