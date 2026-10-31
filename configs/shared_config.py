"""
configs/shared_config.py — Shared System & Protocol Configuration
==================================================================
Single source of truth imported by both serving (`pipeline.py`) and evaluation (`evalkit`).
Loads `configs/final_protocol.json` and exposes validated constants.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

CONFIGS_DIR = Path(__file__).resolve().parent
PROTOCOL_JSON = CONFIGS_DIR / "final_protocol.json"

if not PROTOCOL_JSON.exists():
    raise FileNotFoundError(f"Missing protocol configuration at {PROTOCOL_JSON}")

with open(PROTOCOL_JSON, "r", encoding="utf-8") as f:
    PROTOCOL: Dict[str, Any] = json.load(f)

# Protocol metadata
PROTOCOL_VERSION: str = PROTOCOL.get("protocol_version", "1.0")

# ── Summarization (AraBART) ──────────────────────────────────────────────────
SUMM_CFG: Dict[str, Any] = PROTOCOL.get("summarization", {})
SUMM_MODEL_ID: str       = SUMM_CFG.get("model_id", "Omar10lfc/arabart-xlsum-arabic")
SUMM_MAX_IN: int         = SUMM_CFG.get("max_input_length", 512)
SUMM_MAX_OUT: int        = SUMM_CFG.get("max_summary_length", 100)
SUMM_MIN_LENGTH: int     = SUMM_CFG.get("min_length", 10)
SUMM_NUM_BEAMS: int      = SUMM_CFG.get("num_beams", 4)
SUMM_NO_REPEAT_NGRAM: int = SUMM_CFG.get("no_repeat_ngram_size", 3)
SUMM_LENGTH_PENALTY: float = float(SUMM_CFG.get("length_penalty", 1.0))
SUMM_EARLY_STOPPING: bool = bool(SUMM_CFG.get("early_stopping", True))
SUMM_MODEL_REVISION: str = SUMM_CFG.get("model_revision", "c43dad77a8f6ac6319536f296bd0c260118a9efd")

# ── Retrieval (CAMeL-BERT / FAISS / Cross-Encoder) ───────────────────────────
RET_CFG: Dict[str, Any]          = PROTOCOL.get("retrieval", {})
EMBEDDER_ID: str                 = RET_CFG.get("bi_encoder_id", "CAMeL-Lab/bert-base-arabic-camelbert-msa")
EMBEDDER_REVISION: str           = RET_CFG.get("bi_encoder_revision", "277069fd3645fedb22b746caf38d111aadee0241")
ENCODE_MAX_LENGTH: int           = RET_CFG.get("bi_encoder_max_length", 128)

CROSS_ENCODER_ID: str            = RET_CFG.get("cross_encoder_id", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
CROSS_ENCODER_REVISION: str      = RET_CFG.get("cross_encoder_revision", "1427fd652930e4ba29e8149678df786c240d8825")
CROSS_ENCODER_MAX_LENGTH: int    = 512

CHUNK_WORDS: int                 = RET_CFG.get("chunk_words", 50)
CHUNK_OVERLAP: int               = RET_CFG.get("chunk_overlap", 0)
TOP_K_FAISS: int                 = RET_CFG.get("top_k_bi", 10)
TOP_K_FINAL: int                 = RET_CFG.get("top_k_final", 3)
RET_SEED: int                    = RET_CFG.get("seed", 42)

# ── Speech-to-Text (Whisper Decoding) ─────────────────────────────────────────
ASR_CFG: Dict[str, Any]          = PROTOCOL.get("asr", {})
ASR_GREEDY: Dict[str, Any]       = ASR_CFG.get("primary_greedy", {
    "language": "ar",
    "task": "transcribe",
    "beam_size": 1,
    "best_of": 1,
    "temperature": 0.0,
    "vad_filter": False,
    "condition_on_previous_text": False
})
ASR_LONGFORM: Dict[str, Any]     = ASR_CFG.get("longform_decoding", {
    "language": "ar",
    "task": "transcribe",
    "beam_size": 1,
    "best_of": 1,
    "condition_on_previous_text": False,
    "temperature": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "compression_ratio_threshold": 2.4,
    "no_repeat_ngram_size": 3,
    "vad_filter": True
})

def get_protocol_dict() -> Dict[str, Any]:
    return PROTOCOL

