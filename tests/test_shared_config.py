"""
tests/test_shared_config.py — Acceptance test for Task Q3
==========================================================
Verifies that all protocol keys in configs/final_protocol.json
equal the pipeline constants in pipeline.py and evalkit modules.
"""

import json
from pathlib import Path
import pytest

import pipeline
import configs.shared_config as sc
import evalkit.summarize as es
import evalkit.retrieval as er

def test_protocol_keys_equal_pipeline_constants():
    protocol_path = Path(__file__).resolve().parent.parent / "configs" / "final_protocol.json"
    assert protocol_path.exists(), "configs/final_protocol.json must exist"
    
    with open(protocol_path, "r", encoding="utf-8") as f:
        protocol = json.load(f)
        
    summ = protocol["summarization"]
    ret = protocol["retrieval"]
    asr = protocol["asr"]
    
    # 1. Summarization alignment
    assert pipeline.SUMM_MAX_IN == summ["max_input_length"], "SUMM_MAX_IN mismatch"
    assert pipeline.SUMM_MAX_OUT == summ["max_summary_length"], "SUMM_MAX_OUT mismatch (must be 100)"
    assert pipeline.SUMM_NO_REPEAT_NGRAM == summ["no_repeat_ngram_size"], "SUMM_NO_REPEAT_NGRAM mismatch (must be 3)"
    assert pipeline.SUMM_LENGTH_PENALTY == summ["length_penalty"], "SUMM_LENGTH_PENALTY mismatch (must be 1.0)"
    assert pipeline.SUMM_EARLY_STOPPING == summ["early_stopping"], "SUMM_EARLY_STOPPING mismatch"
    assert pipeline.SUMM_MIN_LENGTH == summ["min_length"], "SUMM_MIN_LENGTH mismatch"
    assert pipeline.SUMM_NUM_BEAMS == summ["num_beams"], "SUMM_NUM_BEAMS mismatch"
    
    # 2. Retrieval alignment
    assert pipeline.ENCODE_MAX_LENGTH == ret["bi_encoder_max_length"], "ENCODE_MAX_LENGTH mismatch (must be 128)"
    assert pipeline.CHUNK_WORDS == ret["chunk_words"], "CHUNK_WORDS mismatch"
    assert pipeline.TOP_K_FAISS == ret["top_k_bi"], "TOP_K_FAISS mismatch"
    assert pipeline.EMBEDDER_ID == ret["bi_encoder_id"], "EMBEDDER_ID mismatch"
    assert pipeline.CROSS_ENCODER_ID == ret["cross_encoder_id"], "CROSS_ENCODER_ID mismatch"
    
    # 3. ASR decoding alignment
    assert pipeline.ASR_GREEDY == asr["primary_greedy"], "ASR_GREEDY mismatch"
    assert pipeline.ASR_LONGFORM == asr["longform_decoding"], "ASR_LONGFORM mismatch"
    
    # 4. evalkit alignment
    assert es.DEFAULT_SUMMARIZATION_CONFIG["max_summary_length"] == 100
    assert es.DEFAULT_SUMMARIZATION_CONFIG["no_repeat_ngram_size"] == 3
    assert er.ENCODE_MAX_LENGTH == 128
    
    # 5. News framing filter test
    assert pipeline.filter_news_framing("في هذا المقال: يتناول الكاتب الذكاء الاصطناعي") == "يتناول الكاتب الذكاء الاصطناعي"
    assert pipeline.filter_news_framing("ذكرت بي بي سي أن المؤتمر بدأ اليوم") == "أن المؤتمر بدأ اليوم"
    assert pipeline.filter_news_framing("شرح مفصل عن المحاضرة") == "شرح مفصل عن المحاضرة"

