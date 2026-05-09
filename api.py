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
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

import pipeline


# =============================================================================
# In-process session store
# =============================================================================
class SessionData(BaseModel):
    """Per-`/analyze` session — held server-side so drill-down doesn't have to
       re-transcribe and re-index. Lives in `_SESSIONS` until the process exits."""
    transcript: str
    takeaways:  List[str]
    cheat_md:   str
    pdf_path:   Optional[str] = None
    # Non-serializable fields kept out of the model — see `_RUNTIME` below.

    model_config = {"arbitrary_types_allowed": True}


# Pydantic can't validate FAISS indexes (they're C++ objects), so we keep the
# `chunks` + FAISS `index` in a parallel dict keyed by the same session id.
_SESSIONS: Dict[str, SessionData] = {}
_RUNTIME:  Dict[str, Dict[str, Any]] = {}


def _new_session() -> str:
    return uuid.uuid4().hex


def _save_upload(upload: UploadFile) -> str:
    """Persist an uploaded file to a temp path that librosa can open."""
    suffix = Path(upload.filename or "audio").suffix or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    with open(path, "wb") as f:
        shutil.copyfileobj(upload.file, f)
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
    device:        str


# =============================================================================
# Endpoints
# =============================================================================
@api.get("/health", response_model=HealthResponse)
def health():
    """Liveness probe + which heavy models are already resident."""
    return HealthResponse(
        status="ok",
        models_loaded={
            "whisper":    pipeline._M.asr_pipeline is not None,
            "embedder":   pipeline._M.embed_mdl   is not None,
            "summarizer": pipeline._M.summ_mdl    is not None,
            "reranker":   pipeline._M.reranker    is not None,
        },
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

    sid = _new_session()
    _SESSIONS[sid] = SessionData(
        transcript=transcript, takeaways=takeaways,
        cheat_md=cheat_md, pdf_path=pdf_path,
    )
    _RUNTIME[sid] = {"chunks": chunks, "index": index}

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


class AnalyzeUrlRequest(BaseModel):
    url: str = Field(..., description="YouTube / Vimeo / podcast URL — anything yt-dlp supports")


@api.post("/analyze-url", response_model=AnalyzeResponse)
async def analyze_url(req: AnalyzeUrlRequest):
    """Full pipeline from a video/audio URL. Downloads the audio with yt-dlp
       and feeds it through the same pipeline as `/analyze`. Requires ffmpeg
       on the server (yt-dlp uses it to extract the audio track)."""
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
    if req.session_id not in _RUNTIME:
        raise HTTPException(404, "Unknown session_id (run /analyze first)")
    runtime = _RUNTIME[req.session_id]
    context = pipeline.drill_down(req.takeaway, runtime["chunks"], runtime["index"])
    if not context:
        raise HTTPException(404, "No matching chunk found for the given takeaway")
    return DrillDownResponse(context=context)


@api.get("/cheat-sheet/{sid}.pdf")
async def cheat_sheet_pdf(sid: str):
    """Download the PDF generated by the most recent /analyze for `sid`."""
    if sid not in _SESSIONS:
        raise HTTPException(404, "Unknown session_id")
    pdf_path = _SESSIONS[sid].pdf_path
    if not pdf_path or not Path(pdf_path).exists():
        raise HTTPException(404, "PDF not available (font may have failed to load)")
    return FileResponse(pdf_path, media_type="application/pdf",
                        filename=f"cheat_sheet_{sid[:8]}.pdf")


@api.delete("/sessions/{sid}")
async def delete_session(sid: str):
    """Free server memory once a client is done with a session."""
    if sid not in _SESSIONS:
        raise HTTPException(404, "Unknown session_id")
    _SESSIONS.pop(sid, None)
    _RUNTIME.pop(sid, None)
    return {"deleted": sid}


# Convenience: `python api.py` runs the server.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:api", host="0.0.0.0", port=8000, reload=False)
