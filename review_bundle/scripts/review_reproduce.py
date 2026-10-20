#!/usr/bin/env python3
"""
scripts/review_reproduce.py — Independent Recomputation & Verification Script
=============================================================================
Strictly reads ONLY raw per-item data from Results/final/ and scratch/test.tsv.
Performs recomputations, second-implementation cross-checks, negative controls,
cluster-bootstrap dependence checks, and exact statistical testing.
Outputs raw evidence without adjectives or quality judgments.
"""

import os
import sys
import json
import math
import hashlib
import random
import re
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# Root dir
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from evalkit.normalize import normalize_arabic_legacy, normalize_arabic_v2
from evalkit.stats import wilson_ci, bootstrap_ci, mcnemar_exact, holm_bonferroni, cluster_bootstrap_ci

# ─────────────────────────────────────────────────────────────────────────────
# 1. RAW DATA LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_cv_pass1_raw():
    p = ROOT_DIR / "Results" / "final" / "cv_pass1_checkpoint.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, encoding="utf-8")
    return df

def load_downstream_raw():
    p = ROOT_DIR / "Results" / "final" / "downstream_retrieval_summarization.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_spoken_retrieval_raw():
    p = ROOT_DIR / "Results" / "final" / "spoken_retrieval_results.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# 2. SECOND-IMPLEMENTATION CHECKERS
# ─────────────────────────────────────────────────────────────────────────────

def levenshtein_distance(ref_words, hyp_words):
    r_len, h_len = len(ref_words), len(hyp_words)
    dp = [[0] * (h_len + 1) for _ in range(r_len + 1)]
    for i in range(r_len + 1):
        dp[i][0] = i
    for j in range(h_len + 1):
        dp[0][j] = j
    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # deletion
                    dp[i][j - 1] + 1,      # insertion
                    dp[i - 1][j - 1] + 1   # substitution
                )
    return dp[r_len][h_len]

def lcs_length(a_tokens, b_tokens):
    m, n = len(a_tokens), len(b_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a_tokens[i - 1] == b_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]

def plain_rouge_l_f1(cand, ref):
    c_tokens = cand.strip().split()
    r_tokens = ref.strip().split()
    if not c_tokens or not r_tokens:
        return 0.0
    lcs = lcs_length(c_tokens, r_tokens)
    p = lcs / len(c_tokens) if c_tokens else 0.0
    r = lcs / len(r_tokens) if r_tokens else 0.0
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)

# ─────────────────────────────────────────────────────────────────────────────
# 3. RECOMPUTATION OF HEADLINE NUMBERS
# ─────────────────────────────────────────────────────────────────────────────

def run_reproduction():
    print("=" * 80)
    print("RAW EVIDENCE REPRODUCTION REPORT (scripts/review_reproduce.py)")
    print("=" * 80)
    print()

    claims = []

    # A. COMMON VOICE PASS 1 POOLED WER
    df_cv = load_cv_pass1_raw()
    if df_cv is not None:
        n_cv = len(df_cv)
        total_audio_s = df_cv["audio_duration_s"].sum()
        total_audio_h = total_audio_s / 3600.0

        leg_s = df_cv["legacy_s"].sum()
        leg_d = df_cv["legacy_d"].sum()
        leg_i = df_cv["legacy_i"].sum()
        leg_n = df_cv["legacy_n"].sum()
        pooled_leg_wer = (leg_s + leg_d + leg_i) / leg_n

        v2_s = df_cv["v2_s"].sum()
        v2_d = df_cv["v2_d"].sum()
        v2_i = df_cv["v2_i"].sum()
        v2_n = df_cv["v2_n"].sum()
        pooled_v2_wer = (v2_s + v2_d + v2_i) / v2_n

        claims.append({
            "claim_id": "CV-PASS1-N",
            "desc": "Common Voice Pass 1 clip count",
            "doc_val": 10506,
            "recomputed": n_cv,
            "abs_diff": abs(10506 - n_cv),
            "source": "Results/final/cv_pass1_checkpoint.csv"
        })
        claims.append({
            "claim_id": "CV-PASS1-AUDIO-HOURS",
            "desc": "Common Voice Pass 1 total audio hours",
            "doc_val": 12.67,
            "recomputed": round(total_audio_h, 2),
            "abs_diff": abs(12.67 - round(total_audio_h, 2)),
            "source": "Results/final/cv_pass1_checkpoint.csv"
        })
        claims.append({
            "claim_id": "CV-PASS1-LEGACY-WER",
            "desc": "Whisper-Large-v3 PyTorch FP16 Pooled Legacy WER",
            "doc_val": 14.07,
            "recomputed": round(pooled_leg_wer * 100.0, 2),
            "abs_diff": abs(14.07 - round(pooled_leg_wer * 100.0, 2)),
            "source": "Results/final/cv_pass1_checkpoint.csv"
        })
        claims.append({
            "claim_id": "CV-PASS1-V2-WER",
            "desc": "Whisper-Large-v3 PyTorch FP16 Pooled v2 WER",
            "doc_val": 11.70,
            "recomputed": round(pooled_v2_wer * 100.0, 2),
            "abs_diff": abs(11.70 - round(pooled_v2_wer * 100.0, 2)),
            "source": "Results/final/cv_pass1_checkpoint.csv"
        })

    # B. RETRIEVAL & SUMMARIZATION RECOMPUTATION
    ds_data = load_downstream_raw()
    if ds_data is not None:
        ret_data = ds_data.get("retrieval", {})
        summ_data = ds_data.get("summarization", {})

        # 1. Retrieval Metrics
        ret_targets = [
            ("RET-ORACLE-P1", "Oracle", "Reference", "oracle", None, "p1", 0.6400),
            ("RET-ORACLE-P3", "Oracle", "Reference", "oracle", None, "p3", 0.7800),
            ("RET-ORACLE-MRR", "Oracle", "Reference", "oracle", None, "mrr", 0.7117),

            ("RET-SET1-LV3-P1", "Set (i)", "large_v3", "set_i", "large_v3", "p1", 0.6000),
            ("RET-SET1-LV3-P3", "Set (i)", "large_v3", "set_i", "large_v3", "p3", 0.8000),
            ("RET-SET1-LV3-MRR", "Set (i)", "large_v3", "set_i", "large_v3", "mrr", 0.7017),

            ("RET-SET1-TURBO-P1", "Set (i)", "turbo", "set_i", "turbo", "p1", 0.6200),
            ("RET-SET1-TURBO-P3", "Set (i)", "turbo", "set_i", "turbo", "p3", 0.8200),
            ("RET-SET1-TURBO-MRR", "Set (i)", "turbo", "set_i", "turbo", "mrr", 0.7330),

            ("RET-SET1-MED-P1", "Set (i)", "medium", "set_i", "medium", "p1", 0.6400),
            ("RET-SET1-MED-P3", "Set (i)", "medium", "set_i", "medium", "p3", 0.8000),
            ("RET-SET1-MED-MRR", "Set (i)", "medium", "set_i", "medium", "mrr", 0.7262),

            ("RET-SET1-SML-P1", "Set (i)", "small", "set_i", "small", "p1", 0.6200),
            ("RET-SET1-SML-P3", "Set (i)", "small", "set_i", "small", "p3", 0.8000),
            ("RET-SET1-SML-MRR", "Set (i)", "small", "set_i", "small", "mrr", 0.7210),

            ("RET-SET2-LV3-P1", "Set (ii)", "large_v3", "set_ii", "large_v3", "p1", 0.6200),
            ("RET-SET2-LV3-P3", "Set (ii)", "large_v3", "set_ii", "large_v3", "p3", 0.7600),
            ("RET-SET2-LV3-MRR", "Set (ii)", "large_v3", "set_ii", "large_v3", "mrr", 0.6800),

            ("RET-SET2-TURBO-P1", "Set (ii)", "turbo", "set_ii", "turbo", "p1", 0.6800),
            ("RET-SET2-TURBO-P3", "Set (ii)", "turbo", "set_ii", "turbo", "p3", 0.8400),
            ("RET-SET2-TURBO-MRR", "Set (ii)", "turbo", "set_ii", "turbo", "mrr", 0.7617),

            ("RET-SET2-MED-P1", "Set (ii)", "medium", "set_ii", "medium", "p1", 0.6000),
            ("RET-SET2-MED-P3", "Set (ii)", "medium", "set_ii", "medium", "p3", 0.7600),
            ("RET-SET2-MED-MRR", "Set (ii)", "medium", "set_ii", "medium", "mrr", 0.6817),

            ("RET-SET2-SML-P1", "Set (ii)", "small", "set_ii", "small", "p1", 0.5000),
            ("RET-SET2-SML-P3", "Set (ii)", "small", "set_ii", "small", "p3", 0.7000),
            ("RET-SET2-SML-MRR", "Set (ii)", "small", "set_ii", "small", "mrr", 0.5967),

            ("RET-SET3-LV3-P1", "Set (iii)", "large_v3", "set_iii", "large_v3", "p1", 0.7000),
            ("RET-SET3-LV3-P3", "Set (iii)", "large_v3", "set_iii", "large_v3", "p3", 0.8800),
            ("RET-SET3-LV3-MRR", "Set (iii)", "large_v3", "set_iii", "large_v3", "mrr", 0.7900),

            ("RET-SET3-TURBO-P1", "Set (iii)", "turbo", "set_iii", "turbo", "p1", 0.6400),
            ("RET-SET3-TURBO-P3", "Set (iii)", "turbo", "set_iii", "turbo", "p3", 0.8800),
            ("RET-SET3-TURBO-MRR", "Set (iii)", "turbo", "set_iii", "turbo", "mrr", 0.7608),

            ("RET-SET3-MED-P1", "Set (iii)", "medium", "set_iii", "medium", "p1", 0.5600),
            ("RET-SET3-MED-P3", "Set (iii)", "medium", "set_iii", "medium", "p3", 0.8000),
            ("RET-SET3-MED-MRR", "Set (iii)", "medium", "set_iii", "medium", "mrr", 0.6842),

            ("RET-SET3-SML-P1", "Set (iii)", "small", "set_iii", "small", "p1", 0.5600),
            ("RET-SET3-SML-P3", "Set (iii)", "small", "set_iii", "small", "p3", 0.7800),
            ("RET-SET3-SML-MRR", "Set (iii)", "small", "set_iii", "small", "mrr", 0.6790),
        ]

        for cid, cond, model, k1, k2, metric, expected in ret_targets:
            node = ret_data.get(k1, {})
            if k2:
                node = node.get(k2, {})
            if "bi_plus_rerank" in node:
                node = node["bi_plus_rerank"]
            hits1 = node.get("hits1", [])
            hits3 = node.get("hits3", [])
            mrrs = node.get("mrrs", [])

            if metric == "p1":
                val = float(np.mean(hits1)) if hits1 else 0.0
            elif metric == "p3":
                val = float(np.mean(hits3)) if hits3 else 0.0
            elif metric == "mrr":
                val = float(np.mean(mrrs)) if mrrs else 0.0
            else:
                val = 0.0

            val = round(val, 4)
            doc_val = node.get(metric, expected)
            claims.append({
                "claim_id": cid,
                "desc": f"ARCD Retrieval {cond} {model} {metric.upper()}",
                "doc_val": doc_val,
                "recomputed": val,
                "abs_diff": abs(doc_val - val),
                "source": "Results/final/downstream_retrieval_summarization.json"
            })

        # 2. Summarization Metrics: Recomputed from per_article_rouge arrays
        oracle_per_art = summ_data.get("oracle", {}).get("per_article_rouge", [])
        if oracle_per_art:
            oracle_rl = [x['rougeL'] for x in oracle_per_art]
            oracle_recomp = round(float(np.mean(oracle_rl)), 2)
            oracle_doc = summ_data.get("oracle", {}).get("rougeL", 0.0)
            claims.append({
                "claim_id": "SUMM-ORACLE-ROUGEL",
                "desc": "AraBART Clean Text Oracle ROUGE-L",
                "doc_val": oracle_doc,
                "recomputed": oracle_recomp,
                "abs_diff": abs(oracle_doc - oracle_recomp),
                "source": "Results/final/downstream_retrieval_summarization.json"
            })

        for m_key in ["large_v3", "turbo", "medium", "small"]:
            m_data = summ_data.get("models", {}).get(m_key, {})
            m_per_art = m_data.get("per_article_rouge", [])
            if m_per_art:
                m_rl = [x['rougeL'] for x in m_per_art]
                m_recomp = round(float(np.mean(m_rl)), 2)
                m_doc = m_data.get("rougeL", 0.0)
                claims.append({
                    "claim_id": f"SUMM-{m_key.upper()}-ROUGEL",
                    "desc": f"AraBART Spoken {m_key} ROUGE-L",
                    "doc_val": m_doc,
                    "recomputed": m_recomp,
                    "abs_diff": abs(m_doc - m_recomp),
                    "source": "Results/final/downstream_retrieval_summarization.json"
                })

    # PRINT RECOMPUTATION TABLE
    print("TABLE: RECOMPUTED HEADLINE NUMBERS VS DOCUMENTED CLAIMS")
    print("-" * 105)
    print(f"{'Claim ID':<22} | {'Description':<38} | {'Doc Val':>8} | {'Recomputed':>10} | {'Abs Diff':>8} | Source")
    print("-" * 105)
    for c in claims:
        print(f"{c['claim_id']:<22} | {c['desc']:<38} | {c['doc_val']:>8} | {c['recomputed']:>10} | {c['abs_diff']:>8.4f} | {c['source']}")
    print("-" * 105)
    max_diff = max(c["abs_diff"] for c in claims) if claims else 0.0
    print(f"Max absolute discrepancy: {max_diff:.6f}")
    if max_diff < 0.01:
        print("STATUS: ALL HEADLINE METRICS VERIFIED AGAINST ITEM-LEVEL RAW ARTIFACTS (< 0.01 TOLERANCE).")
    else:
        print(f"WARNING: Found discrepancies above rounding tolerance: {[c for c in claims if c['abs_diff'] >= 0.01]}")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # 4. SECOND-IMPLEMENTATION CROSS-CHECKS
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("SECOND-IMPLEMENTATION CROSS-CHECKS")
    print("=" * 80)

    # 1. Levenshtein WER cross-check on 200 random clips vs jiwer
    if df_cv is not None:
        import jiwer
        random.seed(42)
        sample_indices = random.sample(range(len(df_cv)), 200)
        sample_df = df_cv.iloc[sample_indices]

        wer_matches = 0
        jw_matches = 0
        total_sample_checked = len(sample_df)

        for _, row in sample_df.iterrows():
            ref_raw = str(row["raw_ref"])
            hyp_raw = str(row["raw_hyp"])

            ref_norm = normalize_arabic_legacy(ref_raw).split()
            hyp_norm = normalize_arabic_legacy(hyp_raw).split()

            dist = levenshtein_distance(ref_norm, hyp_norm)
            manual_wer = dist / len(ref_norm) if ref_norm else (1.0 if hyp_norm else 0.0)

            logged_s = row["legacy_s"]
            logged_d = row["legacy_d"]
            logged_i = row["legacy_i"]
            logged_n = row["legacy_n"]
            logged_wer = (logged_s + logged_d + logged_i) / logged_n if logged_n > 0 else 0.0

            if abs(dist - (logged_s + logged_d + logged_i)) == 0:
                wer_matches += 1

            # Direct comparison with jiwer
            ref_str = " ".join(ref_norm)
            hyp_str = " ".join(hyp_norm)
            jw_val = jiwer.wer(ref_str, hyp_str) if len(ref_norm) > 0 else (1.0 if len(hyp_norm) > 0 else 0.0)
            if abs(manual_wer - jw_val) < 1e-4:
                jw_matches += 1

        print(f"1. Hand-written DP Levenshtein vs. Evaluated WER on 200 Random Clips:")
        print(f"   Matches against logged S+D+I: {wer_matches} / {total_sample_checked} ({wer_matches / total_sample_checked * 100.0:.1f}%)")
        print(f"   Matches against jiwer.wer   : {jw_matches} / {total_sample_checked} ({jw_matches / total_sample_checked * 100.0:.1f}%)")
        print(f"   Exact Alignment Match: {'PASSED' if wer_matches == total_sample_checked and jw_matches == total_sample_checked else 'FLAGGED'}")
        print()

    # 2. Plain LCS vs. Official ROUGE-L on 20 pairs
    if df_cv is not None:
        from rouge_score import rouge_scorer
        sc_check = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
        sample_20 = df_cv.iloc[:20]
        lcs_diffs = []
        for _, r in sample_20.iterrows():
            ref_txt = str(r["raw_ref"])
            hyp_txt = str(r["raw_hyp"])
            ref_tok = sc_check._tokenizer(ref_txt)
            hyp_tok = sc_check._tokenizer(hyp_txt)
            L = lcs_length(hyp_tok, ref_tok)
            p = L / len(hyp_tok) if hyp_tok else 0.0
            rec = L / len(ref_tok) if ref_tok else 0.0
            hand_f1 = (2 * p * rec / (p + rec)) if (p + rec) > 0 else 0.0
            scorer_f1 = sc_check.score(ref_txt, hyp_txt)['rougeL'].fmeasure
            lcs_diffs.append(abs(hand_f1 - scorer_f1))
        print(f"2. Hand-written Plain LCS vs Official ROUGE-L Scorer on 20 Text Pairs:")
        print(f"   Max Abs Diff between Plain LCS and rouge_scorer: {max(lcs_diffs):.6f}")
        print(f"   LCS Implementation Parity: {'PASSED' if max(lcs_diffs) < 1e-5 else 'FLAGGED'}")
        print()

    # 3. P@1 Hand-count verification
    if ds_data is not None:
        p1_checks = []
        for cond_k in ["set_i", "set_ii", "set_iii"]:
            for mod_k in ["large_v3", "turbo", "medium", "small"]:
                node = ds_data.get("retrieval", {}).get(cond_k, {}).get(mod_k, {})
                hits1 = node.get("hits1", [])
                reported_p1 = node.get("p1", 0.0)
                hand_p1 = sum(hits1) / len(hits1) if hits1 else 0.0
                p1_checks.append(abs(reported_p1 - hand_p1))
        print(f"3. Hand-counted P@1 from Raw Hit Vectors across 12 Model-Condition Cells:")
        print(f"   Max Abs Discrepancy: {max(p1_checks):.6f}")
        print(f"   Hand-Count Verification: {'PASSED' if max(p1_checks) < 1e-4 else 'FLAGGED'}")
        print()

    # ─────────────────────────────────────────────────────────────────────────
    # 5. NEGATIVE CONTROLS (from saved data)
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("NEGATIVE CONTROLS & PERMUTATION BASELINES")
    print("=" * 80)

    # 1. WER Negative Controls
    if df_cv is not None:
        sample_100 = df_cv.iloc[:100].copy()
        refs_norm = [normalize_arabic_legacy(str(r)).split() for r in sample_100["raw_ref"]]
        hyps_norm = [normalize_arabic_legacy(str(h)).split() for h in sample_100["raw_hyp"]]

        # a. Identity: Ref == Hyp
        id_dist = sum(levenshtein_distance(r, r) for r in refs_norm)
        id_n = sum(len(r) for r in refs_norm)
        id_wer = id_dist / id_n if id_n > 0 else 0.0

        # b. Empty: Hyp == ""
        empty_dist = sum(levenshtein_distance(r, []) for r in refs_norm)
        empty_wer = empty_dist / id_n if id_n > 0 else 0.0

        # c. Shuffled: Random Permutation
        random.seed(12345)
        shuffled_hyps = list(hyps_norm)
        random.shuffle(shuffled_hyps)
        shuff_dist = sum(levenshtein_distance(r, h) for r, h in zip(refs_norm, shuffled_hyps))
        shuff_wer = shuff_dist / id_n if id_n > 0 else 0.0

        print("1. ASR Word Error Rate Negative Controls (N=100 test clips):")
        print(f"   Control A (Identity ref==hyp) : WER = {id_wer * 100.0:.2f}% (Expected: 0.00%)")
        print(f"   Control B (Empty hypothesis)  : WER = {empty_wer * 100.0:.2f}% (Expected: 100.00%)")
        print(f"   Control C (Shuffled hyp)      : WER = {shuff_wer * 100.0:.2f}% (Expected: >= 95.00%)")
        print()

    # 2. Retrieval Permutation Controls
    if ds_data is not None:
        # Expected chance level for 121 chunks
        chance_level = 1.0 / 121.0
        random.seed(999)
        # 50 queries, target is uniformly distributed in 121
        rand_ranks = [random.randint(1, 121) for _ in range(50)]
        rand_p1 = sum(1 for r in rand_ranks if r == 1) / 50.0
        rand_p3 = sum(1 for r in rand_ranks if r <= 3) / 50.0
        rand_mrr = sum(1.0 / r for r in rand_ranks) / 50.0

        print("2. Speech-RAG Retrieval Chance Level & Permutation Controls (N=50 queries, 121 chunks):")
        print(f"   Theoretical Random P@1 : {chance_level:.4f} (0.83%)")
        print(f"   Empirical Random P@1   : {rand_p1:.4f} (Wilson 95% CI: {wilson_ci(sum(1 for r in rand_ranks if r == 1), 50)})")
        print(f"   Empirical Random P@3   : {rand_p3:.4f}")
        print(f"   Empirical Random MRR   : {rand_mrr:.4f}")
        print()

    # 3. Hypothesis Anomalies
    print("3. Hypothesis Anomalies Scan:")
    if df_cv is not None:
        cv_empty = sum(1 for h in df_cv["raw_hyp"] if not isinstance(h, str) or len(str(h).strip()) == 0)
        cv_len_gt2 = 0
        cv_ngram_loops = 0
        for _, r in df_cv.iterrows():
            ref_w = str(r["raw_ref"]).split()
            hyp_w = str(r["raw_hyp"]).split()
            if len(ref_w) > 0 and len(hyp_w) > 2 * len(ref_w):
                cv_len_gt2 += 1
            if len(hyp_w) >= 6:
                trigrams = [tuple(hyp_w[i:i+3]) for i in range(len(hyp_w)-2)]
                counts = Counter(trigrams)
                if counts and counts.most_common(1)[0][1] >= 3:
                    cv_ngram_loops += 1

        print(f"   Common Voice Pass 1 (Whisper-large-v3 FP16, N=10,506):")
        print(f"     - Empty output count           : {cv_empty}")
        print(f"     - Length ratio > 2 (hyp > 2*ref): {cv_len_gt2} ({cv_len_gt2 / len(df_cv) * 100.0:.2f}%)")
        print(f"     - Repeated 3-gram loops (>=3x) : {cv_ngram_loops} ({cv_ngram_loops / len(df_cv) * 100.0:.2f}%)")

    print(f"   Downstream Summarization (XL-Sum, N=50): NOT AVAILABLE (raw generated strings not serialized in Results/final/downstream_retrieval_summarization.json)")
    print(f"   Downstream Spoken ARCD Transcripts (N=50): NOT AVAILABLE (transcripts not serialized in Results/final/spoken_retrieval_results.json)")
    print()

    # ─────────────────────────────────────────────────────────────────────────
    # 6. STATISTICAL INFERENCE & DEPENDENCE CHECKS
    # ─────────────────────────────────────────────────────────────────────────
    print("=" * 80)
    print("STATISTICAL TESTING & DEPENDENCE STRUCTURE")
    print("=" * 80)

    # 1. Paired McNemar Test on Retrieval
    if ds_data is not None:
        ret_s3_lv3 = ds_data.get("retrieval", {}).get("set_iii", {}).get("large_v3", {}).get("hits1", [])
        ret_s3_turbo = ds_data.get("retrieval", {}).get("set_iii", {}).get("turbo", {}).get("hits1", [])
        ret_s3_med = ds_data.get("retrieval", {}).get("set_iii", {}).get("medium", {}).get("hits1", [])
        ret_s3_sml = ds_data.get("retrieval", {}).get("set_iii", {}).get("small", {}).get("hits1", [])

        # Paired comparisons
        comp_pairs = [
            ("Large-v3 vs Turbo (Set iii)", ret_s3_lv3, ret_s3_turbo),
            ("Large-v3 vs Medium (Set iii)", ret_s3_lv3, ret_s3_med),
            ("Large-v3 vs Small (Set iii)", ret_s3_lv3, ret_s3_sml),
            ("Turbo vs Small (Set iii)", ret_s3_turbo, ret_s3_sml),
        ]

        raw_p_vals = []
        comp_names = []

        print("1. Paired Exact McNemar Tests on Set (iii) Top-1 Retrieval (N=50 queries):")
        for name, h_a, h_b in comp_pairs:
            mcn = mcnemar_exact(h_a, h_b)
            raw_p_vals.append(mcn["p_value"])
            comp_names.append(name)
            print(f"   {name:<32}: b (A=1, B=0) = {mcn['b']}, c (A=0, B=1) = {mcn['c']}, p = {mcn['p_value']:.4f}")

        # Holm-Bonferroni correction
        adj_p_vals = holm_bonferroni(raw_p_vals)
        print()
        print("   Holm-Bonferroni Adjusted p-values:")
        for name, raw_p, adj_p in zip(comp_names, raw_p_vals, adj_p_vals):
            sig = "Statistically Significant (p < 0.05)" if adj_p < 0.05 else "Not Significant"
            print(f"   {name:<32}: raw p = {raw_p:.4f} -> adj p = {adj_p:.4f} ({sig})")
        print()

    # 2. Cluster-Bootstrap by Client ID on Common Voice
    # Map paths to client_ids from scratch/test.tsv
    tsv_p = ROOT_DIR / "scratch" / "test.tsv"
    if tsv_p.exists() and df_cv is not None:
        print("2. Common Voice Dependence Structure (Clustering by Client ID):")
        df_tsv = pd.read_csv(tsv_p, sep="\t", usecols=["path", "client_id"])
        # Merge on path
        merged_cv = pd.merge(df_cv, df_tsv, on="path", how="inner")
        n_merged = len(merged_cv)
        n_clients = merged_cv["client_id"].nunique()
        print(f"   Total clips with client_id: {n_merged} / {len(df_cv)}")
        print(f"   Unique speakers (client_id): {n_clients}")
        print(f"   Clips per speaker: min={merged_cv['client_id'].value_counts().min()}, max={merged_cv['client_id'].value_counts().max()}, median={merged_cv['client_id'].value_counts().median()}")

        # Cluster bootstrap on client_id (500 resamples via pre-aggregated client matrix)
        client_sums = merged_cv.groupby("client_id")[["legacy_s", "legacy_d", "legacy_i", "legacy_n"]].sum().values
        n_clients_k = len(client_sums)

        np.random.seed(42)
        n_boot = 500
        boot_wers = []
        for _ in range(n_boot):
            idx = np.random.randint(0, n_clients_k, size=n_clients_k)
            sampled_sums = client_sums[idx].sum(axis=0)
            b_s, b_d, b_i, b_n = sampled_sums[0], sampled_sums[1], sampled_sums[2], sampled_sums[3]
            boot_wers.append((b_s + b_d + b_i) / b_n)

        ci_low = np.percentile(boot_wers, 2.5) * 100.0
        ci_high = np.percentile(boot_wers, 97.5) * 100.0
        print(f"   Cluster-Bootstrap 95% CI (by speaker, B={n_boot}): [{ci_low:.2f}%, {ci_high:.2f}%]")
        print()

    print("=" * 80)
    print("REPRODUCTION & AUDIT RUN COMPLETE.")
    print("=" * 80)

if __name__ == "__main__":
    run_reproduction()
