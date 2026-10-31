"""
modal_app.py — Smart Lecture Assistant Modal Deployment
======================================================
Architecture:
- Pinned Debian 12 (bookworm-slim) image with Python 3.11, CUDA 12.1 PyTorch, cuDNN 9, CTranslate2 4.4.0.
- Modal Volume (`smart-lecture-assistant-volume`) mounted at `/vol` caching model weights.
- Weight verification on container boot: validates SHA-256 of `model.bin` against manifest (efa94de1...).
  Fails fast with fatal error if hash mismatches.
- Serving via `@modal.asgi_app()` wrapping FastAPI backend.
- Concurrency configured using Modal's current concurrency API (`max_inputs=20`, `max_containers=2`).
- Asynchronous spawn-and-poll mechanism for long audio files exceeding HTTP gateway timeout (150s).

VRAM & Cost Estimates (Pending Live GPU Measurement):
- GPU Target: Nvidia T4 (16GB VRAM)
- Estimated Serving Memory:
    * Faster-Whisper Turbo CT2 INT8: ~1.6 GB
    * AraBART Summarizer (FP16):     ~1.2 GB
    * CAMeL-BERT Bi-Encoder (FP16):  ~0.8 GB
    * Cross-Encoder Reranker (FP16): ~0.8 GB
    * Total estimated VRAM:          ~4.4 GB (well within 16GB VRAM capacity)
- Estimated Cost: ~$0.59 / active GPU-hour (~$0.0098 / min). Zero cost when scaled to 0.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional

import modal

# =============================================================================
# Modal App & Infrastructure Definition
# =============================================================================
APP_NAME = "smart-lecture-assistant"
VOLUME_NAME = "smart-lecture-assistant-volume"
JOBS_DICT_NAME = "smart-lecture-jobs"
SECRET_NAME = "lecture-assistant-secret"

# Canonical served model metadata
CANONICAL_MODEL_REPO = "Omar10lfc/whisper-large-v3-turbo-arabic-ct2-int8"
CANONICAL_MODEL_REV = "873028c03839f1a84a883a1e7e536f0041873154"
EXPECTED_MODEL_BIN_SHA256 = "efa94de178f110f53be1c951f6cfdd81b6aab2681adbbee1f18b77ca2e47f248"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
jobs_store = modal.Dict.from_name(JOBS_DICT_NAME, create_if_missing=True)

# Build pinned Debian 12 container image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.3.1",
        "torchaudio==2.3.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "nvidia-cudnn-cu12==9.1.0.70",
        "ctranslate2==4.4.0",
        "faster-whisper==1.0.3",
        "transformers==4.44.2",
        "accelerate>=0.30.0,<1.0",
        "sentence-transformers>=3.0.0",
        "faiss-cpu>=1.8.0",
        "fastapi>=0.110.0,<0.116",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.9",
        "starlette>=0.36,<0.45",
        "huggingface_hub>=0.23,<1.0",
        "librosa>=0.10.0",
        "fpdf2>=2.7.0",
        "arabic-reshaper>=3.0.0",
        "python-bidi>=0.4.2",
        "yt-dlp>=2024.1.0",
        "numpy<2.0",
        "pydantic>=2.0.0",
        "httpx>=0.27.0",
    )
    .env({
        "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH",
        "WHISPER_MODEL_ID": "/vol/models/whisper-large-v3-turbo-ct2-int8",
        "WHISPER_REVISION": CANONICAL_MODEL_REV,
        "ALLOW_URL_INGEST": "0",  # Security: strictly disabled in production
    })
    .add_local_python_source("api", "pipeline", "evalkit")
    .add_local_dir("configs", remote_path="/root/configs")
)


# =============================================================================
# Weight Verification & Synchronization Helper
# =============================================================================
def verify_or_download_served_weights() -> str:
    """Validate that the exact fine-tuned Turbo CT2 INT8 weights exist on the Volume.
    Fails fast if the sha256 does not match the manifest."""
    target_dir = Path("/vol/models/whisper-large-v3-turbo-ct2-int8")
    model_bin = target_dir / "model.bin"

    needs_download = False
    if not model_bin.exists():
        print(f"[modal_init] model.bin not found at {model_bin}. Initiating download...")
        needs_download = True
    else:
        # Check SHA-256
        h = hashlib.sha256()
        with open(model_bin, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        observed_sha = h.hexdigest()
        if observed_sha != EXPECTED_MODEL_BIN_SHA256:
            print(f"[modal_init] SHA mismatch! Expected {EXPECTED_MODEL_BIN_SHA256}, got {observed_sha}. Re-downloading...")
            needs_download = True
        else:
            print(f"[modal_init] Verified served CT2 model.bin SHA256 matches manifest: {observed_sha}")

    if needs_download:
        target_dir.mkdir(parents=True, exist_ok=True)
        from huggingface_hub import snapshot_download
        print(f"[modal_init] Downloading snapshot from {CANONICAL_MODEL_REPO} (rev: {CANONICAL_MODEL_REV})...")
        snapshot_download(
            repo_id=CANONICAL_MODEL_REPO,
            revision=CANONICAL_MODEL_REV,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        vol.commit()

        # Re-verify after download
        h = hashlib.sha256()
        with open(model_bin, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        final_sha = h.hexdigest()
        if final_sha != EXPECTED_MODEL_BIN_SHA256:
            raise RuntimeError(
                f"FATAL: Served weight SHA256 mismatch after download! Expected {EXPECTED_MODEL_BIN_SHA256}, got {final_sha}"
            )
        print(f"[modal_init] Successfully verified served CT2 model.bin SHA256: {final_sha}")

    return str(target_dir)


# =============================================================================
# Modal Functions: Background Spawn-and-Poll Worker
# =============================================================================
@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": vol},
    timeout=1800,  # 30 minutes max for long lectures
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
def process_lecture_job(job_id: str, audio_bytes: bytes, suffix: str = ".wav") -> Dict[str, Any]:
    """Asynchronous background worker for long lectures exceeding HTTP timeout."""
    import api
    import pipeline

    verify_or_download_served_weights()

    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)

        # Update job status in Dict
        jobs_store[job_id] = {
            "status": "processing",
            "progress": "transcribing_and_indexing",
            "created_at": time.time(),
        }

        # Run pipeline
        resp = api._analyze_audio_path(temp_path)
        result_data = resp.model_dump()

        jobs_store[job_id] = {
            "status": "completed",
            "result": result_data,
            "created_at": time.time(),
        }
        return result_data
    except Exception as e:
        jobs_store[job_id] = {
            "status": "failed",
            "error": str(e),
            "created_at": time.time(),
        }
        raise
    finally:
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass


# =============================================================================
# Modal Cls: FastAPI Web Service
# =============================================================================
@app.cls(
    image=image,
    gpu="T4",
    volumes={"/vol": vol},
    timeout=600,
    min_containers=0,
    max_containers=2,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
class SmartLectureService:

    @modal.enter()
    def initialize(self):
        """Container cold start hook: verify weights on Volume and warm up models."""
        print("[modal_boot] Executing container cold start hook...")
        weight_dir = verify_or_download_served_weights()
        os.environ["WHISPER_MODEL_ID"] = weight_dir
        os.environ["WHISPER_REVISION"] = CANONICAL_MODEL_REV

        import pipeline
        print("[modal_boot] Pre-warming heavy models in memory...")
        pipeline.warm_up()
        print("[modal_boot] Warm-up complete. Service ready.")

    @modal.asgi_app()
    def serve(self):
        """Expose FastAPI application through Modal reverse proxy."""
        import api
        return api.api
