"""
pipeline.py — Smart Lecture Assistant backend
==============================================

All model loading and pipeline logic lives here. `app.py` (Gradio) and `api.py`
(FastAPI) both import from this module. Tests can also import the pure helpers
without triggering any model load — model loading is lazy.

Pipeline:
    audio  ── Whisper (ASR) ──>  transcript
                                     │
                                     ├─ chunked + CAMeL-BERT embedded → FAISS
                                     ├─ AraBART summarizer → cheat sheet (PDF)
                                     └─ AraBART per-segment → 3-5 takeaways
                                                                  │
                                          click a takeaway → Cross-Encoder rerank
                                                          → exact transcript chunk
"""

from __future__ import annotations

import os
import re
import textwrap
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# CONFIG
# =============================================================================
PROJECT_DIR        = Path(__file__).resolve().parent

# Models are loaded from the HF Hub by default (works on HF Spaces and any
# clean machine). To use a local checkpoint instead, point these at a folder
# path — `from_pretrained` accepts either form.
WHISPER_DIR        = os.environ.get(
    "WHISPER_MODEL_ID", "Omar10lfc/whisper-small-arabic")
SUMMARIZER_DIR     = os.environ.get(
    "SUMMARIZER_MODEL_ID", "Omar10lfc/arabart-xlsum-arabic")

# Embedder + reranker default to HF IDs. Replace with `PROJECT_DIR / "<folder>"`
# if you've downloaded local copies.
EMBEDDER_ID        = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
CROSS_ENCODER_ID   = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"  # multilingual

FONT_DIR  = PROJECT_DIR / "fonts"
FONT_PATH = FONT_DIR / "Amiri-Regular.ttf"
FONT_URL  = ("https://github.com/aliftype/amiri/raw/master/fonts/ttf/"
             "Amiri-Regular.ttf")

CHUNK_WORDS   = 50
TOP_K_FAISS   = 10
N_TAKEAWAYS   = 5
SUMM_MAX_IN   = 512
SUMM_MAX_OUT  = 84

# Hidden queries that drive the cheat sheet — each becomes one section.
CHEAT_QUERIES: List[Tuple[str, str]] = [
    ("النقاط الرئيسية",       "ما هي النقاط الرئيسية والمفاهيم الأساسية؟"),
    ("التعريفات والمصطلحات", "ما هي التعريفات والمصطلحات المهمة؟"),
    ("الأمثلة التوضيحية",     "ما هي الأمثلة التوضيحية المذكورة؟"),
    ("القواعد والصيغ",        "ما هي القواعد أو الصيغ أو المعادلات؟"),
    ("الخلاصة",                "ما هي الخلاصة والاستنتاجات النهائية؟"),
]


# =============================================================================
# PURE HELPERS — no model imports, safe for unit tests
# =============================================================================
_AR_ALEF   = re.compile(r"[إأآٱ]")
_AR_DIAC   = re.compile(r"[ً-ٰٟ]")
_TATWEEL   = re.compile(r"ـ")
_WS        = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Full Arabic normalization that matches `embedding-eval.ipynb` exactly:
       alef/ya/ta-marbuta unification + diacritics + tatweel + whitespace.

       Used for embeddings (CAMeL-BERT) and rerank (Cross-Encoder) inputs so the
       demo's retrieval space is identical to the one that produced P@1 = 0.86
       in the chunking/reranking eval."""
    if not text:
        return ""
    text = _AR_ALEF.sub("ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = _AR_DIAC.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text


def clean_for_summary(text: str) -> str:
    """Lighter cleanup for AraBART input (preserves alef forms / ta-marbuta /
       dotless-ya so tokenization matches what the model saw at training time)."""
    if not text:
        return ""
    text = _AR_DIAC.sub("", text)
    text = _TATWEEL.sub("", text)
    text = _WS.sub(" ", text).strip()
    return text


def chunk_text(text: str, chunk_words: int = CHUNK_WORDS) -> List[str]:
    """Split into fixed-size word windows. The 50-word default matches the
       configuration that won the chunking/reranking eval (P@1 0.86 + rerank)."""
    words = normalize_arabic(text).split()
    chunks = []
    for i in range(0, len(words), chunk_words):
        c = " ".join(words[i:i + chunk_words])
        if len(c.split()) >= 5:        # drop tiny trailing fragments
            chunks.append(c)
    return chunks


def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard for cheap dedup of near-identical takeaways."""
    A, B = set(a.split()), set(b.split())
    return len(A & B) / max(1, len(A | B))


# =============================================================================
# LAZY MODEL CONTAINER — models load on first use, never at import time
# =============================================================================
@dataclass
class _Models:
    whisper_proc = None
    whisper_mdl  = None
    asr_pipeline = None
    embed_tok    = None
    embed_mdl    = None
    summ_tok     = None
    summ_mdl     = None
    reranker     = None
    device       = None


_M = _Models()


def _device() -> str:
    if _M.device is None:
        import torch
        _M.device = "cuda" if torch.cuda.is_available() else "cpu"
    return _M.device


# `low_cpu_mem_usage=True` loads weights tensor-by-tensor instead of building
# the full state dict in RAM first. On Windows this is the difference between
# fitting in a typical page-file budget and getting OSError 1455.
# `torch_dtype=fp16` on GPU halves both VRAM and CPU staging memory.
def _load_kwargs():
    import torch
    kw = {"low_cpu_mem_usage": True}
    if _device() == "cuda":
        kw["torch_dtype"] = torch.float16
    return kw


def _load_whisper():
    if _M.asr_pipeline is not None:
        return
    from transformers import (
        WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
        WhisperForConditionalGeneration,
        pipeline as hf_pipeline,
    )
    print(f"[pipeline] loading Whisper from {WHISPER_DIR} ...")
    # Load the tokenizer/feature-extractor from the base model, not the
    # fine-tuned repo. Fine-tuning didn't change the tokenizer, and the
    # tokenizer.json saved alongside our fine-tuned weights was serialized
    # with a newer `tokenizers` (>=0.20) than transformers 4.44.2 can parse.
    # WhisperProcessor requires the slow WhisperTokenizer (not Fast), and the
    # base repo ships vocab.json + merges.txt that the slow tokenizer needs.
    _BASE = "openai/whisper-small"
    _feat = WhisperFeatureExtractor.from_pretrained(_BASE)
    _tok  = WhisperTokenizer.from_pretrained(_BASE)
    _M.whisper_proc = WhisperProcessor(feature_extractor=_feat, tokenizer=_tok)
    _M.whisper_mdl  = (WhisperForConditionalGeneration
                       .from_pretrained(str(WHISPER_DIR), **_load_kwargs())
                       .to(_device()).eval())
    _M.whisper_mdl.generation_config.language = "arabic"
    _M.whisper_mdl.generation_config.task     = "transcribe"
    _M.asr_pipeline = hf_pipeline(
        "automatic-speech-recognition",
        model=_M.whisper_mdl,
        tokenizer=_M.whisper_proc.tokenizer,
        feature_extractor=_M.whisper_proc.feature_extractor,
        chunk_length_s=30,
        stride_length_s=5,
        device=0 if _device() == "cuda" else -1,
    )


def _load_embedder():
    if _M.embed_mdl is not None:
        return
    from transformers import AutoTokenizer, AutoModel
    print(f"[pipeline] loading embedder {EMBEDDER_ID} ...")
    _M.embed_tok = AutoTokenizer.from_pretrained(EMBEDDER_ID)
    _M.embed_mdl = (AutoModel.from_pretrained(EMBEDDER_ID, **_load_kwargs())
                    .to(_device()).eval())


def _load_summarizer():
    if _M.summ_mdl is not None:
        return
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    print(f"[pipeline] loading summarizer from {SUMMARIZER_DIR} ...")
    _M.summ_tok = AutoTokenizer.from_pretrained(str(SUMMARIZER_DIR))
    _M.summ_mdl = (AutoModelForSeq2SeqLM
                   .from_pretrained(str(SUMMARIZER_DIR), **_load_kwargs())
                   .to(_device()).eval())


def _load_reranker():
    if _M.reranker is not None:
        return
    from sentence_transformers import CrossEncoder
    print(f"[pipeline] loading reranker {CROSS_ENCODER_ID} ...")
    _M.reranker = CrossEncoder(CROSS_ENCODER_ID, device=_device(), max_length=512)


def warm_up() -> None:
    """Eagerly load every model. Call once at process start (e.g., from FastAPI
       lifespan) to front-load the slow initialization instead of paying for it
       on the first user request."""
    _load_whisper()
    _load_embedder()
    _load_summarizer()
    _load_reranker()


# =============================================================================
# EMBEDDING — mean-pool + L2 normalize (matches embedding-eval.ipynb)
# =============================================================================
def _mean_pool(token_emb, attn_mask):
    import torch
    mask = attn_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


def encode_arabic(texts: List[str], batch_size: int = 32) -> np.ndarray:
    import torch
    _load_embedder()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inp = _M.embed_tok(batch, padding=True, truncation=True,
                               max_length=512, return_tensors="pt"
                              ).to(_device())
            h   = _M.embed_mdl(**inp).last_hidden_state
            emb = _mean_pool(h, inp["attention_mask"]).cpu().numpy().astype("float32")
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.where(norms == 0, 1, norms)
            out.append(emb)
    return np.vstack(out)


# =============================================================================
# 1. SPEECH-TO-TEXT
# =============================================================================
def download_audio_from_url(url: str) -> str:
    """Download audio from a YouTube / Vimeo / Twitter / podcast URL to a temp
       WAV file and return the local path. Powered by yt-dlp, so anything on
       https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md works.

       Requires ffmpeg on PATH (yt-dlp uses it to demux+encode the audio
       stream). On Windows: `winget install ffmpeg` once, then restart the shell.

       Raises RuntimeError with a friendly message if the URL is invalid,
       the video is private/age-restricted, or ffmpeg is missing."""
    import yt_dlp

    if not url or not url.strip():
        raise ValueError("Empty URL")

    # One temp dir per call so concurrent requests don't collide on the same id.
    out_dir = Path(tempfile.mkdtemp(prefix="lecture_audio_"))
    out_template = str(out_dir / "%(id)s.%(ext)s")

    opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,           # if a playlist URL is passed, take 1st item only
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",  # librosa+soundfile can decode WAV without ffmpeg
            "preferredquality": "192",
        }],
    }

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url.strip(), download=True)
    except Exception as e:
        raise RuntimeError(f"yt-dlp failed: {e}") from e

    # The FFmpegExtractAudio postprocessor renames to .wav. Fall back to
    # whatever yt-dlp actually wrote if the rename didn't happen (e.g. ffmpeg
    # was missing — yt-dlp will then have left the original m4a/webm in place).
    wav_path = out_dir / f"{info['id']}.wav"
    if wav_path.exists():
        return str(wav_path)
    fallback = list(out_dir.glob(f"{info['id']}.*"))
    if not fallback:
        raise RuntimeError(f"yt-dlp downloaded but no file appeared in {out_dir}")
    return str(fallback[0])


def run_whisper(audio_path: str) -> str:
    """Transcribe an Arabic audio file. Long-form audio is handled by HF's
       pipeline via 30s chunks + 5s stride — no manual windowing needed.

       We decode the file with librosa first (soundfile backend) and hand the
       resulting numpy array to the pipeline. This avoids the pipeline's
       default ffmpeg-shell-out, which is missing on most Windows installs and
       blows up with `WinError 2` when transcribing uploads."""
    if not audio_path:
        return ""
    import librosa
    _load_whisper()
    speech, _ = librosa.load(audio_path, sr=16000, mono=True)
    result = _M.asr_pipeline(
        speech,
        generate_kwargs={"language": "arabic", "task": "transcribe"},
    )
    return result["text"].strip()


# =============================================================================
# 2. INDEX
# =============================================================================
def build_index(transcript: str):
    """Chunk → embed → FAISS inner-product index. Returns (chunks, index).
       For lecture-sized inputs an exact `IndexFlatIP` is the right choice."""
    import faiss
    chunks = chunk_text(transcript)
    if not chunks:
        return [], None
    emb = encode_arabic(chunks)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return chunks, index


def _retrieve(query: str, chunks, index, k: int = TOP_K_FAISS):
    """Return [(chunk_idx, similarity)] ranked by FAISS."""
    if not chunks or index is None:
        return []
    qv = encode_arabic([normalize_arabic(query)])
    sims, idx = index.search(qv, min(k, len(chunks)))
    return [(int(idx[0][i]), float(sims[0][i])) for i in range(len(idx[0]))]


# =============================================================================
# 3. SUMMARIZATION (AraBART)
# =============================================================================
def _summarize(texts, num_beams: int = 4, max_target: int = SUMM_MAX_OUT,
               min_length: int = 10) -> List[str]:
    """Batched AraBART decoding using the XL-Sum-paper generation parameters."""
    import torch
    _load_summarizer()
    if isinstance(texts, str):
        texts = [texts]
    texts = [clean_for_summary(t) for t in texts]
    with torch.no_grad():
        enc = _M.summ_tok(texts, max_length=SUMM_MAX_IN, truncation=True,
                          padding="max_length", return_tensors="pt"
                         ).to(_device())
        out = _M.summ_mdl.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_length=max_target,
            min_length=min_length,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            length_penalty=0.6,
            early_stopping=True,
        )
    return _M.summ_tok.batch_decode(out, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=False)


# =============================================================================
# 4. CHEAT SHEET
# =============================================================================
def generate_cheat_sheet(chunks, index) -> Tuple[str, Optional[str]]:
    """Returns (markdown, pdf_path)."""
    if not chunks or index is None:
        return "## لا يوجد محتوى لتحليله.", None

    sections: List[Tuple[str, str]] = []
    for title, query in CHEAT_QUERIES:
        hits = _retrieve(query, chunks, index, k=3)
        if not hits:
            continue
        retrieved = " ".join(chunks[h[0]] for h in hits)
        summary   = _summarize(retrieved)[0]
        sections.append((title, summary))

    md = ["# 📘 ملخص المحاضرة — دليل المراجعة\n"]
    for title, body in sections:
        md.append(f"## {title}\n")
        md.append(f"- {body}\n")
    md_text = "\n".join(md)

    pdf_path = _write_arabic_pdf(sections)
    return md_text, pdf_path


# =============================================================================
# 5. TAKEAWAYS
# =============================================================================
def generate_takeaways(transcript: str, n: int = N_TAKEAWAYS) -> List[str]:
    """Split transcript into ~equal segments, summarize each → N bullet points
       that together cover the entire lecture (not just the first 512 tokens)."""
    words = clean_for_summary(transcript).split()
    if len(words) < 30:
        return [transcript.strip()] if transcript.strip() else []

    seg_size = max(60, len(words) // n)
    segments = [" ".join(words[i:i + seg_size])
                for i in range(0, len(words), seg_size)]
    if len(segments) > n:
        segments = segments[:n - 1] + [" ".join(segments[n - 1:])]

    bullets = _summarize(segments)

    uniq: List[str] = []
    for b in bullets:
        b = b.strip()
        if b and not any(jaccard(b, u) > 0.7 for u in uniq):
            uniq.append(b)
    return uniq[:n]


# =============================================================================
# 6. DRILL-DOWN
# =============================================================================
def drill_down(selected_takeaway: str, chunks, index) -> str:
    """FAISS top-K → cross-encoder rerank → return the single best chunk."""
    if not selected_takeaway or not chunks or index is None:
        return ""
    _load_reranker()
    hits = _retrieve(selected_takeaway, chunks, index, k=TOP_K_FAISS)
    if not hits:
        return ""
    # Cross-encoder reads the takeaway against the *normalized* form of each
    # candidate, mirroring the embedding-eval setup so scores are comparable.
    pairs     = [(normalize_arabic(selected_takeaway),
                  normalize_arabic(chunks[i])) for i, _ in hits]
    ce_scores = _M.reranker.predict(pairs)
    best_local = int(np.argmax(ce_scores))
    return chunks[hits[best_local][0]]


# =============================================================================
# 7. PDF EXPORT (Arabic-shaped + bidi, rendered with the Amiri font)
# =============================================================================
def _ensure_font():
    FONT_DIR.mkdir(exist_ok=True)
    if FONT_PATH.exists():
        return FONT_PATH
    try:
        print(f"[pdf] downloading Arabic font → {FONT_PATH}")
        urllib.request.urlretrieve(FONT_URL, FONT_PATH)
        return FONT_PATH
    except Exception as e:
        print(f"[pdf] font download failed: {e}; PDF export will be skipped.")
        return None


def _shape_ar(text: str) -> str:
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


def _write_arabic_pdf(sections: List[Tuple[str, str]]) -> Optional[str]:
    from fpdf import FPDF
    font_path = _ensure_font()
    if not font_path:
        return None
    pdf = FPDF()
    pdf.add_page()
    pdf.add_font("Amiri", style="", fname=str(font_path))

    pdf.set_font("Amiri", size=18)
    pdf.cell(0, 12, _shape_ar("ملخص المحاضرة - دليل المراجعة"),
             align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(6)

    for title, body in sections:
        pdf.set_font("Amiri", size=14)
        pdf.cell(0, 10, _shape_ar(title),
                 align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Amiri", size=12)
        for line in textwrap.wrap(body, width=70):
            pdf.cell(0, 8, _shape_ar(line),
                     align="R", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)

    out_path = Path(tempfile.gettempdir()) / "cheat_sheet.pdf"
    pdf.output(str(out_path))
    return str(out_path)
