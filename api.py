"""
api.py — Smart Lecture Assistant (FastAPI backend)
===================================================

REST API exposing the same pipeline as the Gradio app. Useful for integrating
the assistant into other clients (mobile app, web frontend, batch tooling).

Endpoints:
    GET  /health                       liveness + model-load status
    POST /transcribe                   audio file → transcript only
    POST /analyze                      audio file → full bundle (transcript +
                                       cheat sheet markdown + takeaways +
                                       session_id for drill-down)
    POST /drill-down                   {session_id, takeaway} → matching chunk
    GET  /cheat-sheet/{sid}.pdf        download the PDF cheat sheet for a session

Run:
    uvicorn api:api --host 0.0.0.0 --port 8000

Notes on session storage:
    Sessions are kept in an in-process dict so drill-down can reuse the FAISS
    index built by `/analyze` without rebuilding from the transcript. For a
    multi-worker deployment, swap `_SESSIONS` for Redis or a similar store —
    FAISS indexes can be serialized via `faiss.write_index`.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
import time
import socket
import ipaddress
import hashlib
import hmac
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import pipeline


# =============================================================================
# In-process session store
# =============================================================================
SESSION_TTL_SECONDS = int(os.environ.get("SESSION_TTL_SECONDS", "14400"))  # 4 hours
MAX_SESSIONS = int(os.environ.get("MAX_SESSIONS", "1000"))


class SessionData(BaseModel):
    """Per-`/analyze` session — held server-side so drill-down doesn't have to
       re-transcribe and re-index. Lives in `_SESSIONS` until TTL expiry or capacity eviction."""
    transcript: str
    takeaways:  List[str]
    cheat_md:   str
    pdf_path:   Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    # Non-serializable fields kept out of the model — see `_RUNTIME` below.

    model_config = {"arbitrary_types_allowed": True}


# Pydantic can't validate FAISS indexes (they're C++ objects), so we keep the
# `chunks` + FAISS `index` in a parallel dict keyed by the same session id.
_SESSIONS: Dict[str, SessionData] = {}
_RUNTIME:  Dict[str, Dict[str, Any]] = {}


def evict_expired_sessions() -> int:
    """Evict sessions older than SESSION_TTL_SECONDS. Returns count of evicted sessions."""
    now = time.time()
    expired = []
    for sid, sdata in list(_SESSIONS.items()):
        created = getattr(sdata, "created_at", None)
        if created is None or (now - created > SESSION_TTL_SECONDS):
            expired.append(sid)

    for sid, rdata in list(_RUNTIME.items()):
        if sid not in _SESSIONS and sid not in expired:
            expired.append(sid)

    for sid in expired:
        sdata = _SESSIONS.pop(sid, None)
        _RUNTIME.pop(sid, None)
        if sdata and sdata.pdf_path and os.path.exists(sdata.pdf_path):
            try:
                os.unlink(sdata.pdf_path)
            except OSError:
                pass
    return len(expired)


def enforce_session_capacity() -> None:
    """Enforce MAX_SESSIONS limit by evicting expired or oldest sessions (FIFO)."""
    evict_expired_sessions()
    while len(_SESSIONS) >= MAX_SESSIONS:
        oldest_sid = min(_SESSIONS.keys(), key=lambda s: getattr(_SESSIONS[s], "created_at", 0))
        sdata = _SESSIONS.pop(oldest_sid, None)
        _RUNTIME.pop(oldest_sid, None)
        if sdata and sdata.pdf_path and os.path.exists(sdata.pdf_path):
            try:
                os.unlink(sdata.pdf_path)
            except OSError:
                pass


# Asynchronous jobs store (for long lectures exceeding standard HTTP timeouts)
_JOBS: Dict[str, Dict[str, Any]] = {}


def evict_expired_jobs() -> int:
    """Evict asynchronous job records older than SESSION_TTL_SECONDS."""
    now = time.time()
    expired = [jid for jid, jdata in list(_JOBS.items()) if now - jdata.get("created_at", 0) > SESSION_TTL_SECONDS]
    for jid in expired:
        _JOBS.pop(jid, None)
    return len(expired)
    for jid in expired:
        _JOBS.pop(jid, None)
    return len(expired)

# Rate Limiter: max requests per IP window
# NOTE: The in-process rate limiter is per container / per worker process.
# In a serverless/multi-worker deployment (e.g., Modal), each container manages
# its own sliding window. For production, URL ingest remains strictly disabled
# (ALLOW_URL_INGEST=0) to prevent SSRF, DNS rebinding, and redirect attacks.
_RATE_LIMIT_STORE: Dict[str, List[float]] = defaultdict(list)
_RATE_LIMIT_MAX_REQ = int(os.environ.get("RATE_LIMIT_MAX_REQ", "60"))
_RATE_LIMIT_WINDOW = 60.0


def validate_ingest_url(url: str) -> None:
    """Validate external URL against SSRF, loopback, private IPs and non-http schemes.
       NOTE: In production, ALLOW_URL_INGEST defaults to '0' (disabled) because third-party
       download tools (yt-dlp) follow HTTP redirects and are subject to DNS rebinding.
       Keep URL ingest disabled in production environments."""
    if not url or not url.strip():
        raise HTTPException(status_code=400, detail="Empty URL provided.")

    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid URL scheme '{scheme}'. Only http and https are allowed."
        )

    hostname = parsed.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="Missing hostname in URL.")

    # Resolve IP and check against forbidden ranges
    try:
        addr_info = socket.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        raise HTTPException(status_code=400, detail=f"Cannot resolve hostname '{hostname}': {e}")

    for family, _, _, _, sockaddr in addr_info:
        ip_str = sockaddr[0]
        try:
            ip = ipaddress.ip_address(ip_str)
        except ValueError:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip_str == "169.254.169.254"
        ):
            raise HTTPException(
                status_code=400,
                detail=f"Access to private/internal IP address '{ip_str}' is forbidden."
            )


def _new_session() -> str:
    return uuid.uuid4().hex


MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 25 * 1024 * 1024))  # 25 MB cap
ALLOWED_AUDIO_MIMES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/x-m4a",
    "audio/m4a",
    "audio/webm",
}


def _save_upload(upload: UploadFile) -> str:
    """Validate MIME type and file size, persisting an uploaded file to a temp path."""
    raw_ct = upload.content_type or ""
    mime = raw_ct.split(";")[0].strip().lower()
    if mime not in ALLOWED_AUDIO_MIMES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{mime or 'unknown'}'. Allowed formats: {sorted(ALLOWED_AUDIO_MIMES)}",
        )

    suffix = Path(upload.filename or "audio").suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)

    total_bytes = 0
    chunk_size = 1024 * 1024  # 1MB
    try:
        with open(path, "wb") as f:
            while True:
                chunk = upload.file.read(chunk_size)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum permitted size of {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.",
                    )
                f.write(chunk)
    except Exception:
        if os.path.exists(path):
            try:
                os.unlink(path)
            except OSError:
                pass
        raise

    return path


# =============================================================================
# FastAPI app + lifespan (eager model warm-up)
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm models once at boot so first user request isn't slow. Skip during
    # tests by setting SKIP_WARMUP=1 in the env.
    if os.environ.get("SKIP_WARMUP") != "1":
        try:
            pipeline.warm_up()
        except Exception as e:                      # missing weights, etc.
            print(f"[api] warm-up failed: {e!r} — models will load lazily")
    yield


api = FastAPI(
    title="Smart Lecture Assistant API",
    version="1.0.0",
    description="Arabic ASR + summarization + semantic-search REST backend.",
    lifespan=lifespan,
)


@api.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)

    # 1. Key by API key if provided
    api_key = request.headers.get("x-api-key")
    if not api_key:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            api_key = auth_header[7:].strip()

    if api_key:
        rate_key = f"key:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
    else:
        # 2. Key by trusted client IP behind proxy.
        # NOTE: In X-Forwarded-For: <client>, <proxy1>, the first entry is client-controlled
        # and trivially spoofable. Trusted reverse proxies (Modal, Cloudflare) append the
        # verified client IP. We take the last entry added by the trusted ingress.
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            hops = [h.strip() for h in forwarded.split(",") if h.strip()]
            client_ip = hops[-1] if hops else "127.0.0.1"
        elif request.client:
            client_ip = request.client.host
        else:
            client_ip = "127.0.0.1"
        rate_key = f"ip:{client_ip}"

    now = time.time()
    timestamps = _RATE_LIMIT_STORE[rate_key]

    valid_timestamps = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
    if len(valid_timestamps) >= _RATE_LIMIT_MAX_REQ:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please slow down."},
            headers={"Retry-After": str(int(_RATE_LIMIT_WINDOW))}
        )

    valid_timestamps.append(now)
    _RATE_LIMIT_STORE[rate_key] = valid_timestamps
    return await call_next(request)


@api.middleware("http")
async def auth_middleware(request: Request, call_next):
    token = os.environ.get("API_BEARER_TOKEN", "").strip()
    # Bypass auth if token is unset (dev mode) or for health check and docs
    if token and request.url.path not in ("/health", "/docs", "/openapi.json", "/redoc"):
        api_key = request.headers.get("x-api-key")
        if not api_key:
            auth_header = request.headers.get("authorization", "")
            if auth_header.lower().startswith("bearer "):
                api_key = auth_header[7:].strip()

        if not api_key or not hmac.compare_digest(api_key, token):
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized: invalid or missing API key."},
            )
    return await call_next(request)


# =============================================================================
# Request / response schemas
# =============================================================================
class TranscribeResponse(BaseModel):
    transcript: str


class AnalyzeResponse(BaseModel):
    session_id:   str = Field(..., description="Use with /drill-down or /cheat-sheet/{sid}.pdf")
    transcript:   str
    cheat_sheet:  str = Field(..., description="Markdown-formatted study guide")
    takeaways:    List[str]
    pdf_url:      Optional[str] = None


class DrillDownRequest(BaseModel):
    session_id: str
    takeaway:   str


class DrillDownResponse(BaseModel):
    context: str = Field(..., description="The exact transcript chunk best matching the takeaway")


class HealthResponse(BaseModel):
    status:        str
    models_loaded: Dict[str, bool]
    asr_model:     Dict[str, Any]
    device:        str


class AsyncJobResponse(BaseModel):
    job_id:   str
    status:   str
    poll_url: str


class JobStatusResponse(BaseModel):
    job_id:     str
    status:     str
    created_at: float
    result:     Optional[AnalyzeResponse] = None
    error:      Optional[str] = None


# =============================================================================
# Endpoints
# =============================================================================
@api.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe + which heavy models are already resident + ASR model metadata."""
    return HealthResponse(
        status="ok",
        models_loaded={
            "whisper":    (pipeline._M.asr_pipeline is not None or pipeline._M.whisper_mdl is not None),
            "embedder":   pipeline._M.embed_mdl   is not None,
            "summarizer": pipeline._M.summ_mdl    is not None,
            "reranker":   pipeline._M.reranker    is not None,
        },
        asr_model=pipeline.get_asr_model_info(),
        device=pipeline._device(),
    )


@api.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(audio: UploadFile = File(...)):
    """ASR-only endpoint — useful when the caller doesn't need summarization."""
    path = _save_upload(audio)
    try:
        text = pipeline.run_whisper(path)
    except Exception as e:
        raise HTTPException(500, f"ASR failed: {e!r}")
    finally:
        os.unlink(path)
    if not text:
        raise HTTPException(422, "Empty transcript — check audio quality / format")
    return TranscribeResponse(transcript=text)


def _analyze_audio_path(audio_path: str) -> AnalyzeResponse:
    """Shared body of /analyze and /analyze-url. Caller is responsible for
       deleting the source file if it's a temp upload — yt-dlp downloads in
       /analyze-url have their own temp dir handled inside `pipeline`."""
    transcript = pipeline.run_whisper(audio_path)
    if not transcript:
        raise HTTPException(422, "Empty transcript — check audio quality / format")

    chunks, index = pipeline.build_index(transcript)
    if not chunks:
        raise HTTPException(422, "Transcript too short to index")

    cheat_md, pdf_path = pipeline.generate_cheat_sheet(chunks, index)
    takeaways          = pipeline.generate_takeaways(transcript)

    enforce_session_capacity()
    sid = _new_session()
    now = time.time()
    _SESSIONS[sid] = SessionData(
        transcript=transcript, takeaways=takeaways,
        cheat_md=cheat_md, pdf_path=pdf_path,
        created_at=now,
    )
    _RUNTIME[sid] = {"chunks": chunks, "index": index, "created_at": now}

    return AnalyzeResponse(
        session_id=sid,
        transcript=transcript,
        cheat_sheet=cheat_md,
        takeaways=takeaways,
        pdf_url=f"/cheat-sheet/{sid}.pdf" if pdf_path else None,
    )


@api.post("/analyze", response_model=AnalyzeResponse)
async def analyze(audio: UploadFile = File(...)):
    """Full pipeline from an uploaded audio file."""
    path = _save_upload(audio)
    try:
        return _analyze_audio_path(path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {e!r}")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@api.post("/analyze-async", response_model=AsyncJobResponse, status_code=202)
async def analyze_async(audio: UploadFile = File(...)):
    """Asynchronous analysis endpoint for long lectures exceeding HTTP timeout."""
    path = _save_upload(audio)
    job_id = uuid.uuid4().hex
    evict_expired_jobs()
    now = time.time()
    _JOBS[job_id] = {
        "status": "processing",
        "created_at": now,
        "result": None,
        "error": None,
    }

    # If running on Modal, spawn function call
    if os.environ.get("MODAL_ENVIRONMENT") and "modal_app" in sys.modules:
        try:
            with open(path, "rb") as f:
                content = f.read()
            import modal_app
            modal_app.process_lecture_job.spawn(job_id, content, Path(path).suffix)
        except Exception as e:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["error"] = str(e)
        finally:
            if os.path.exists(path):
                os.unlink(path)
    else:
        # Local / test mode: execute pipeline and record output
        try:
            resp = _analyze_audio_path(path)
            _JOBS[job_id]["status"] = "completed"
            _JOBS[job_id]["result"] = resp
        except Exception as e:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["error"] = str(e)
        finally:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass

    return AsyncJobResponse(
        job_id=job_id,
        status=_JOBS[job_id]["status"],
        poll_url=f"/jobs/{job_id}",
    )


@api.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll the status and result of an asynchronous processing job."""
    evict_expired_jobs()
    if job_id not in _JOBS:
        # Check Modal dict if running in cloud
        if os.environ.get("MODAL_ENVIRONMENT") and "modal_app" in sys.modules:
            import modal_app
            if job_id in modal_app.jobs_store:
                data = modal_app.jobs_store[job_id]
                return JobStatusResponse(
                    job_id=job_id,
                    status=data.get("status", "processing"),
                    created_at=data.get("created_at", time.time()),
                    result=data.get("result"),
                    error=data.get("error"),
                )
        raise HTTPException(404, "Unknown or expired job_id")

    job = _JOBS[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        created_at=job["created_at"],
        result=job.get("result"),
        error=job.get("error"),
    )


class AnalyzeUrlRequest(BaseModel):
    url: str = Field(..., description="YouTube / Vimeo / podcast URL — anything yt-dlp supports")


@api.post("/analyze-url", response_model=AnalyzeResponse)
async def analyze_url(req: AnalyzeUrlRequest):
    """Full pipeline from a video/audio URL. Downloads the audio with yt-dlp
       and feeds it through the same pipeline as `/analyze`. Requires ALLOW_URL_INGEST=1.
       Protected against SSRF and internal address access."""
    if os.environ.get("ALLOW_URL_INGEST") != "1":
        raise HTTPException(
            status_code=403,
            detail="URL ingestion is disabled on this server. Set ALLOW_URL_INGEST=1 to enable."
        )

    validate_ingest_url(req.url)

    try:
        audio_path = pipeline.download_audio_from_url(req.url)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        # yt-dlp errors (private video, geo-block, missing ffmpeg, …)
        raise HTTPException(400, f"Failed to download from URL: {e}")
    except Exception as e:
        raise HTTPException(500, f"Unexpected download error: {e!r}")

    try:
        return _analyze_audio_path(audio_path)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {e!r}")
    finally:
        try:
            os.unlink(audio_path)
        except OSError:
            pass


@api.post("/drill-down", response_model=DrillDownResponse)
async def drill_down(req: DrillDownRequest):
    """Reuse the session's FAISS index to find the exact chunk behind a takeaway."""
    evict_expired_sessions()
    if req.session_id not in _RUNTIME:
        raise HTTPException(404, "Unknown or expired session_id (run /analyze first)")
    runtime = _RUNTIME[req.session_id]
    context = pipeline.drill_down(req.takeaway, runtime["chunks"], runtime["index"])
    if not context:
        raise HTTPException(404, "No matching chunk found for the given takeaway")
    return DrillDownResponse(context=context)


@api.get("/cheat-sheet/{sid}.pdf")
async def cheat_sheet_pdf(sid: str):
    """Download the PDF generated by the most recent /analyze for `sid`."""
    evict_expired_sessions()
    if sid not in _SESSIONS:
        raise HTTPException(404, "Unknown or expired session_id")
    pdf_path = _SESSIONS[sid].pdf_path
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(404, "PDF not available (font may have failed to load)")
    return FileResponse(pdf_path, media_type="application/pdf",
                        filename=f"cheat_sheet_{sid[:8]}.pdf")


@api.delete("/sessions/{sid}")
async def delete_session(sid: str):
    """Free server memory once a client is done with a session."""
    evict_expired_sessions()
    if sid not in _SESSIONS:
        raise HTTPException(404, "Unknown or expired session_id")
    sdata = _SESSIONS.pop(sid, None)
    _RUNTIME.pop(sid, None)
    if sdata and sdata.pdf_path and os.path.exists(sdata.pdf_path):
        try:
            os.unlink(sdata.pdf_path)
        except OSError:
            pass
    return {"deleted": sid}


# Convenience: `python api.py` runs the server.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:api", host="0.0.0.0", port=8000, reload=False)
