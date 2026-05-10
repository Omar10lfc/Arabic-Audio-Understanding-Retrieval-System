"""
Shared pytest fixtures.

The API tests run without ever loading a real model — `monkeypatch_pipeline`
replaces the heavy functions on the `pipeline` module with deterministic stubs.
That keeps the test suite fast (<1s) and runnable on a machine without the
model checkpoints downloaded.
"""

import os
from typing import Any, List, Tuple

import pytest

# Skip the FastAPI lifespan warm-up so model loading isn't triggered on import.
os.environ.setdefault("SKIP_WARMUP", "1")


@pytest.fixture
def fake_index():
    """A minimal stand-in for a FAISS index. The pipeline only ever passes it
       around as an opaque object — the API doesn't introspect its contents,
       so any object will do."""
    return object()


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace every heavy pipeline call with a deterministic stub. Returns the
       module so individual tests can override specific stubs if needed."""
    import pipeline

    monkeypatch.setattr(pipeline, "run_whisper",
                        lambda path: "هذا نص تجريبي للمحاضرة. " * 10)

    def fake_build_index(transcript):
        chunks = [transcript[:50], transcript[50:100], transcript[100:150]]
        return chunks, object()

    monkeypatch.setattr(pipeline, "build_index", fake_build_index)
    monkeypatch.setattr(pipeline, "generate_cheat_sheet",
                        lambda chunks, idx: ("# دليل المراجعة\n- نقطة 1", None))
    monkeypatch.setattr(pipeline, "generate_takeaways",
                        lambda transcript, n=5: [f"فكرة {i + 1}" for i in range(3)])
    monkeypatch.setattr(pipeline, "drill_down",
                        lambda tk, chunks, idx: chunks[0] if chunks else "")

    # /analyze-url goes through download_audio_from_url. Stub returns a real
    # temp file path so the endpoint can call os.unlink(...) on it without
    # NotFoundError, but the file is empty — run_whisper above is also stubbed.
    import tempfile
    def fake_download(url: str) -> str:
        if not url or "://" not in url:
            raise ValueError("Empty URL")
        if "fail" in url.lower():
            raise RuntimeError("Simulated yt-dlp failure")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        return path
    monkeypatch.setattr(pipeline, "download_audio_from_url", fake_download)

    # Health endpoint reads the device helper directly — give it a safe default.
    monkeypatch.setattr(pipeline, "_device", lambda: "cpu")

    return pipeline


@pytest.fixture
def client(stub_pipeline):
    """FastAPI TestClient with the heavy pipeline already stubbed."""
    from fastapi.testclient import TestClient
    from api import api, _SESSIONS, _RUNTIME
    _SESSIONS.clear()
    _RUNTIME.clear()
    with TestClient(api) as c:
        yield c
