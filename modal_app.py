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
# User-created secret on Modal
SECRET_NAME = os.environ.get("MODAL_SECRET_NAME", "audio")

# Canonical served model metadata
CANONICAL_MODEL_REPO = "Omar10lfc/whisper-large-v3-turbo-arabic-ct2-int8"
CANONICAL_MODEL_REV = "873028c03839f1a84a883a1e7e536f0041873154"
EXPECTED_MODEL_BIN_SHA256 = "efa94de178f110f53be1c951f6cfdd81b6aab2681adbbee1f18b77ca2e47f248"

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
jobs_store = modal.Dict.from_name(JOBS_DICT_NAME, create_if_missing=True)

# Compute target: T4 GPU by default, or CPU if MODAL_USE_CPU=1
MODAL_GPU_SPEC: Optional[str] = None if os.environ.get("MODAL_USE_CPU") == "1" else "T4"

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
        "tokenizers>=0.20.3",
        "faster-whisper>=1.0.3",
        "transformers>=4.45.0",
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

    # Ensure preprocessor_config.json exists with feature_size=128 for Turbo model
    preprocessor_json = target_dir / "preprocessor_config.json"
    processor_json = target_dir / "processor_config.json"
    if not preprocessor_json.exists():
        if processor_json.exists():
            import json
            with open(processor_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            feat_cfg = data.get("feature_extractor", data)
            with open(preprocessor_json, "w", encoding="utf-8") as f:
                json.dump(feat_cfg, f, indent=2)
            print(f"[modal_init] Extracted preprocessor_config.json with feature_size={feat_cfg.get('feature_size')}")
        else:
            import json
            with open(preprocessor_json, "w", encoding="utf-8") as f:
                json.dump({"feature_size": 128, "sampling_rate": 16000, "hop_length": 160, "chunk_length": 30, "n_fft": 400}, f, indent=2)
            print("[modal_init] Created default 128-mel preprocessor_config.json for Turbo")
        vol.commit()

    return str(target_dir)


# =============================================================================
# Modal Functions: Background Spawn-and-Poll Worker
# =============================================================================
@app.function(
    image=image,
    gpu=MODAL_GPU_SPEC,
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
# SESSIONS & STATE ARCHITECTURE NOTE:
# In the current dev deployment, session FAISS indexes (_RUNTIME) and asynchronous
# job states (_JOBS) are maintained in-process within container memory.
# Therefore, max_containers is pinned strictly to 1 so all requests route to the
# same stateful container.
# TASK B2.1 (Pre-prod hardening): Persist session FAISS indexes to modal.Volume
# and job states to modal.Dict so max_containers can autoscale horizontally.
@app.cls(
    image=image,
    gpu=MODAL_GPU_SPEC,
    volumes={"/vol": vol},
    timeout=600,
    min_containers=0,
    max_containers=1,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
class SmartLectureService:

    @modal.enter()
    def initialize(self):
        """Container cold start hook: enforce auth fail-closed, verify weights on Volume, warm up."""
        print("[modal_boot] Executing container cold start hook...")
        token = (os.environ.get("audiotoken") or os.environ.get("API_BEARER_TOKEN", "")).strip()
        if token:
            os.environ["API_BEARER_TOKEN"] = token
        dev_no_auth = os.environ.get("DEV_NO_AUTH") == "1"
        if not token and not dev_no_auth:
            raise RuntimeError(
                f"FATAL: API_BEARER_TOKEN / audiotoken is required on Modal. Refusing to boot because "
                f"secret '{SECRET_NAME}' is missing/empty and DEV_NO_AUTH!=1."
            )

        weight_dir = verify_or_download_served_weights()
        os.environ["WHISPER_MODEL_ID"] = weight_dir
        os.environ["WHISPER_REVISION"] = CANONICAL_MODEL_REV
        os.environ["WHISPER_CANONICAL_ID"] = CANONICAL_MODEL_REPO

        import pipeline
        print("[modal_boot] Pre-warming heavy models in GPU memory...")
        pipeline.warm_up()
        print("[modal_boot] Warm-up complete. Service ready.")

    @modal.asgi_app()
    def serve(self):
        """Expose FastAPI application through Modal reverse proxy."""
        import api
        return api.api


# =============================================================================
# GPU & cuDNN Pinning Verification Probe
# =============================================================================
@app.function(
    image=image,
    gpu=MODAL_GPU_SPEC,
    volumes={"/vol": vol},
    timeout=600,
    secrets=[modal.Secret.from_name(SECRET_NAME)],
)
def verify_gpu_transcription() -> Dict[str, Any]:
    """Test real transcription to verify PyTorch and CTranslate2 linking."""
    import torch
    import faster_whisper
    import numpy as np

    has_cuda = torch.cuda.is_available()
    print(f"[compute_test] PyTorch Version: {torch.__version__}")
    print(f"[compute_test] CUDA Available: {has_cuda}")

    if has_cuda:
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    else:
        device = "cpu"
        device_name = "CPU"
    print(f"[compute_test] Compute Device: {device_name}")

    weight_dir = verify_or_download_served_weights()
    print(f"[compute_test] Loading Faster-Whisper on {device_name} (compute_type=int8)...")
    t0 = time.time()
    model = faster_whisper.WhisperModel(weight_dir, device=device, compute_type="int8")
    load_time = time.time() - t0
    print(f"[compute_test] Model loaded in {load_time:.2f}s")

    # Generate 3 seconds of 16kHz audio for live transcription test
    sample_rate = 16000
    dummy_audio = np.zeros(sample_rate * 3, dtype=np.float32)

    t1 = time.time()
    segments, info = model.transcribe(dummy_audio, language="ar", beam_size=1)
    seg_list = list(segments)
    infer_time = time.time() - t1
    print(f"[compute_test] Real inference completed in {infer_time:.2f}s. Detected lang: {info.language}")

    vram_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024) if has_cuda else 0.0
    vram_res_mb = torch.cuda.memory_reserved() / (1024 * 1024) if has_cuda else 0.0

    return {
        "status": "success",
        "device": device,
        "device_name": device_name,
        "load_time_sec": round(load_time, 2),
        "infer_time_sec": round(infer_time, 2),
        "vram_allocated_mb": round(vram_alloc_mb, 2),
        "vram_reserved_mb": round(vram_res_mb, 2),
    }
