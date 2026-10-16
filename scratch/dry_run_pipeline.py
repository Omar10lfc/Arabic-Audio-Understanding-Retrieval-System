import json, os, sys
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath('.'))

from evalkit import (
    normalize_arabic, clean_for_summary,
    compute_wer, wer_breakdown,
    score_summary, multilingual_rouge_scoring,
    chunk_passage, build_faiss_index, evaluate_retrieval, set_all_seeds,
    wilson_ci, mcnemar_exact, bootstrap_ci
)

print("=== DRY RUN: 2-ITEM PIPELINE ON CPU ===")

# 1. Normalization & WER
ref = "ذهب الطالب الى المدرسة"
hyp = "ذهب الطالب للمدرسة"
s, d, i, h, n, w = wer_breakdown(ref, hyp)
print(f"WER dry run: ref='{ref}', hyp='{hyp}' -> WER={w:.4f} (S={s}, D={d}, I={i}, H={h}, N={n})")

# 2. ROUGE Scoring
ref_sum = "أعلن الوزير تعافي الاقتصاد الوطني خلال العام الجاري."
pred_sum = "قال الوزير أن الاقتصاد الوطني يتعافى في هذا العام."
rouge_res = score_summary(ref_sum, pred_sum)
print(f"ROUGE dry run: {rouge_res}")

# 3. Retrieval 2-item index
passages = [
    "تعتبر مدينة القاهرة عاصمة جمهورية مصر العربية وأكبر مدنها من حيث عدد السكان.",
    "تعد بغداد عاصمة جمهورية العراق والمركز التجاري والتعليمي الرئيسي في البلاد."
]
queries = [
    {'question': 'ما هي عاصمة مصر؟', 'target_passage_id': '0'},
    {'question': 'ما هي عاصمة العراق؟', 'target_passage_id': '1'}
]

chunks = []
chunk_to_p = []
for pid, p in enumerate(passages):
    p_chunks = chunk_passage(p, chunk_words=10, overlap=2)
    for c in p_chunks:
        chunks.append(c)
        chunk_to_p.append(str(pid))

# Dummy embeddings for dry run
dim = 8
np.random.seed(42)
c_embs = np.random.randn(len(chunks), dim).astype(np.float32)
q_embs = np.random.randn(len(queries), dim).astype(np.float32)

idx = build_faiss_index(c_embs)
ret_res = evaluate_retrieval(queries, q_embs, chunks, chunk_to_p, idx)
print(f"Retrieval dry run: P@1={ret_res['p1']}, P@3={ret_res['p3']}, MRR={ret_res['mrr']}")
print("[OK] Dry run completed successfully with zero errors.")
