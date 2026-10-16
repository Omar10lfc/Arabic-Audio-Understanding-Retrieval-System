import pytest
import numpy as np
import pandas as pd
import json
import os
import faiss

from evalkit.normalize import normalize_arabic, normalize_arabic_legacy, normalize_arabic_v2, clean_for_summary
from evalkit.wer import compute_wer, wer_breakdown, compute_dual_wer_record
from evalkit.rouge import score_summary, multilingual_rouge_scoring
from evalkit.retrieval import chunk_passage, build_faiss_index, evaluate_retrieval, set_all_seeds
from evalkit.stats import (
    wilson_ci, mcnemar_exact, bootstrap_ci, paired_bootstrap_ci,
    cluster_bootstrap_ci, holm_bonferroni, save_result_file
)

class TestEvalkitNormalize:
    def test_normalize_arabic_core(self):
        assert normalize_arabic("أحمد") == "احمد"
        assert normalize_arabic("إيمان") == "ايمان"
        assert normalize_arabic("آية") == "ايه"
        assert normalize_arabic("على") == "علي"
        assert normalize_arabic("مدرسة") == "مدرسه"
        assert normalize_arabic("سـلام") == "سلام"
        assert normalize_arabic("مَكْتَبَة") == "مكتبه"

    def test_clean_for_summary(self):
        assert clean_for_summary("أحمد") == "أحمد"
        assert clean_for_summary("مدرسة") == "مدرسة"
        assert clean_for_summary("سـلام") == "سلام"
        assert clean_for_summary("مَكْتَبَة") == "مكتبة"

class TestEvalkitWER:
    def test_exact_jiwer_equivalence(self):
        ref = "ذهب الطالب الى المدرسة صباحا"
        hyp = "ذهب طالب الى مدرسة صباحا باكرا"
        s, d, i, h, n, wer = wer_breakdown(ref, hyp)
        assert n == h + s + d
        assert abs(wer - (s + d + i) / n) < 1e-12
        assert compute_wer(ref, hyp) == wer

    def test_identity_wer(self):
        ref = "هذا نص تجريبي"
        hyp = "هذا نص تجريبي"
        s, d, i, h, n, wer = wer_breakdown(ref, hyp)
        assert wer == 0.0
        assert s == 0 and d == 0 and i == 0 and h == 3

    def test_compute_dual_wer_record(self):
        raw_ref = "قال: «السلام عليكم»!"
        raw_hyp = "قال السلام عليكم"
        rec = compute_dual_wer_record(raw_ref, raw_hyp)
        assert 'raw_ref' in rec and 'raw_hyp' in rec
        assert 'legacy_wer' in rec and 'v2_wer' in rec
        assert 'legacy_s' in rec and 'v2_s' in rec

class TestEvalkitRouge:
    def test_stemmer_differentiation(self):
        s_stem = score_summary("المعلم", "المعلمون", use_stemmer=True)
        s_nostem = score_summary("المعلم", "المعلمون", use_stemmer=False)
        assert s_stem['rougeL'] == 100.0
        assert s_nostem['rougeL'] == 0.0

    def test_multilingual_scoring_parity(self):
        preds = ["قال الوزير ان الاقتصاد يتعافى", "فاز الفريق بالبطولة"]
        refs = ["اعلن الوزير تعافي الاقتصاد", "توج الفريق بلقب البطولة"]
        scores = multilingual_rouge_scoring(preds, refs)
        assert 'rouge1' in scores
        assert 'rouge2' in scores
        assert 'rougeL' in scores
        assert scores['rougeL'] > 0.0

        # Parity check: single pair in multilingual_rouge_scoring vs score_summary
        single_multi = multilingual_rouge_scoring([preds[0]], [refs[0]])
        single_direct = score_summary(refs[0], preds[0])
        assert abs(single_multi['rougeL'] - single_direct['rougeL']) < 1e-5

class TestEvalkitRetrieval:
    def test_chunking_default(self):
        words = ["كلمة"] * 120
        text = " ".join(words)
        chunks = chunk_passage(text, chunk_words=50, overlap=0, normalize=False)
        assert len(chunks) == 3
        assert len(chunks[0].split()) == 50
        assert len(chunks[1].split()) == 50
        assert len(chunks[2].split()) == 20

    def test_faiss_and_eval(self):
        set_all_seeds(42)
        dim = 16
        n_chunks = 10
        chunks = [f"chunk {i}" for i in range(n_chunks)]
        chunk_to_p = [f"p_{i//2}" for i in range(n_chunks)]

        chunk_embs = np.random.randn(n_chunks, dim).astype(np.float32)
        index = build_faiss_index(chunk_embs)
        assert index.ntotal == n_chunks

        q_embs = chunk_embs[:1]
        queries = [{'question': 'q0', 'target_passage_id': 'p_0'}]

        res = evaluate_retrieval(queries, q_embs, chunks, chunk_to_p, index)
        assert res['p1'] == 1.0
        assert res['hits1'] == [1]
        assert 'p3_passage' in res
        assert 'mrrs_passage' in res

class TestEvalkitStats:
    def test_wilson_ci(self):
        ci = wilson_ci(50, 100)
        assert 0.0 <= ci[0] <= 0.5 <= ci[1] <= 1.0

    def test_mcnemar_exact(self):
        y_a = [1, 1, 1, 0, 0]
        y_b = [1, 0, 0, 1, 0]
        res = mcnemar_exact(y_a, y_b)
        assert res['b'] == 2
        assert res['c'] == 1
        assert res['n_discordant'] == 3
        assert 0.0 <= res['p_value'] <= 1.0

    def test_bootstrap_ci(self):
        data = [10.0, 12.0, 14.0, 16.0, 18.0]
        res = bootstrap_ci(data, n_bootstrap=500, seed=42)
        assert 10.0 <= res['mean'] <= 18.0
        assert res['se'] > 0.0

    def test_paired_bootstrap_ci_known_answers(self):
        # Known answer 1: constant difference of 5.0
        a = [10.0, 20.0, 30.0, 40.0]
        b = [5.0, 15.0, 25.0, 35.0]
        res_diff = paired_bootstrap_ci(a, b, stat="mean_diff", n_bootstrap=500, seed=42)
        assert abs(res_diff['mean'] - 5.0) < 1e-4
        assert abs(res_diff['se'] - 0.0) < 1e-4
        assert abs(res_diff['ci_lower'] - 5.0) < 1e-4
        assert abs(res_diff['ci_upper'] - 5.0) < 1e-4

        # Known answer 2: constant ratio of 2.0
        a_ratio = [20.0, 40.0, 60.0]
        b_ratio = [10.0, 20.0, 30.0]
        res_ratio = paired_bootstrap_ci(a_ratio, b_ratio, stat="ratio_of_means", n_bootstrap=500, seed=42)
        assert abs(res_ratio['mean'] - 2.0) < 1e-4
        assert abs(res_ratio['se'] - 0.0) < 1e-4
        assert abs(res_ratio['ci_lower'] - 2.0) < 1e-4
        assert abs(res_ratio['ci_upper'] - 2.0) < 1e-4

    def test_cluster_bootstrap(self):
        df = pd.DataFrame({
            'client_id': ['c1', 'c1', 'c2', 'c2', 'c3'],
            'err': [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        res = cluster_bootstrap_ci(df, 'client_id', lambda d: d['err'].mean(), n_bootstrap=100, seed=42)
        assert res['se'] > 0.0

    def test_holm_bonferroni(self):
        p_vals = [0.01, 0.04, 0.03]
        adj = holm_bonferroni(p_vals)
        assert len(adj) == 3
        assert adj[0] == 0.03
        assert adj[2] == 0.06
        assert adj[1] == 0.06

    def test_save_result_file(self, tmp_path):
        fpath = str(tmp_path / "test_res.json")
        save_result_file(fpath, {'score': 42.0}, config={'opt': True}, device='cpu')
        assert os.path.exists(fpath)
        with open(fpath, 'r', encoding='utf-8') as f:
            d = json.load(f)
        assert d['results']['score'] == 42.0
        assert d['metadata']['device'] == 'cpu'
        assert d['metadata']['config_copy']['opt'] is True
        assert 'git_commit' in d['metadata']
        assert 'evalkit_sha256' in d['metadata']
        assert isinstance(d['metadata']['evalkit_sha256'], dict)
        assert 'normalize.py' in d['metadata']['evalkit_sha256']
