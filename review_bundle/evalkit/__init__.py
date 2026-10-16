"""
evalkit: Standardized Evaluation Toolkit for Arabic Audio-Text Pipeline.
Unified protocols for ASR (WER), Cascading Summarization (ROUGE),
Spoken Document Retrieval (Speech-RAG), and Statistical Rigor.
"""

from .normalize import (
    normalize_arabic,
    normalize_arabic_legacy,
    normalize_arabic_v2,
    clean_for_summary
)
from .wer import compute_wer, wer_breakdown, compute_dual_wer_record
from .stats import (
    wilson_ci, mcnemar_exact, bootstrap_ci, paired_bootstrap_ci,
    cluster_bootstrap_ci, holm_bonferroni, save_result_file
)

# Graceful optional imports for heavy / downstream packages
try:
    from .rouge import score_summary, multilingual_rouge_scoring
except Exception:
    score_summary = None
    multilingual_rouge_scoring = None

try:
    from .retrieval import (
        chunk_passage, build_faiss_index, evaluate_retrieval, set_all_seeds,
        load_camelbert, encode_camelbert
    )
except Exception:
    chunk_passage = None
    build_faiss_index = None
    evaluate_retrieval = None
    set_all_seeds = None
    load_camelbert = None
    encode_camelbert = None

try:
    from .summarize import summarize_batch, load_summarizer
except Exception:
    summarize_batch = None
    load_summarizer = None

__all__ = [
    "normalize_arabic",
    "normalize_arabic_legacy",
    "normalize_arabic_v2",
    "clean_for_summary",
    "compute_wer",
    "wer_breakdown",
    "compute_dual_wer_record",
    "score_summary",
    "multilingual_rouge_scoring",
    "chunk_passage",
    "build_faiss_index",
    "evaluate_retrieval",
    "set_all_seeds",
    "load_camelbert",
    "encode_camelbert",
    "wilson_ci",
    "mcnemar_exact",
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "cluster_bootstrap_ci",
    "holm_bonferroni",
    "save_result_file",
    "summarize_batch",
    "load_summarizer",
]
