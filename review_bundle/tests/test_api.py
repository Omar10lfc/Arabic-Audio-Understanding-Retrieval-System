"""
FastAPI integration tests using TestClient + monkeypatched pipeline. No real
models are loaded; the heavy functions are replaced with deterministic stubs in
`conftest.py::stub_pipeline`. Whole suite runs in ~1 second.
"""

import io

import pytest


# A tiny WAV byte payload — header only. The pipeline call is stubbed, so the
# file just needs to exist and be uploadable; actual audio decoding never runs.
_FAKE_WAV = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00" \
            b"@\x1f\x00\x00\x80>\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"


def _upload(client, audio_bytes=_FAKE_WAV, filename="lecture.wav"):
    return client.post(
        "/analyze",
        files={"audio": (filename, io.BytesIO(audio_bytes), "audio/wav")},
    )


# =============================================================================
# /health
# =============================================================================
class TestHealth:

    def test_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "models_loaded" in body
        assert set(body["models_loaded"]) == {"whisper", "embedder",
                                              "summarizer", "reranker"}
        assert body["device"] in {"cpu", "cuda"}


# =============================================================================
# /transcribe
# =============================================================================
class TestTranscribe:

    def test_returns_transcript(self, client):
        r = client.post(
            "/transcribe",
            files={"audio": ("a.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
        )
        assert r.status_code == 200
        body = r.json()
        assert "transcript" in body
        assert body["transcript"]                  # non-empty stub output

    def test_empty_transcript_is_422(self, client, monkeypatch):
        import pipeline
        monkeypatch.setattr(pipeline, "run_whisper", lambda p: "")
        r = client.post(
            "/transcribe",
            files={"audio": ("a.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
        )
        assert r.status_code == 422

    def test_asr_failure_is_500(self, client, monkeypatch):
        import pipeline
        def boom(_):
            raise RuntimeError("CUDA OOM")
        monkeypatch.setattr(pipeline, "run_whisper", boom)
        r = client.post(
            "/transcribe",
            files={"audio": ("a.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
        )
        assert r.status_code == 500
        assert "ASR failed" in r.json()["detail"]


# =============================================================================
# /analyze
# =============================================================================
class TestAnalyze:

    def test_returns_full_bundle(self, client):
        r = _upload(client)
        assert r.status_code == 200
        body = r.json()
        assert set(body) >= {"session_id", "transcript", "cheat_sheet",
                             "takeaways", "pdf_url"}
        assert isinstance(body["takeaways"], list)
        assert body["takeaways"]                   # 3 fake takeaways from stub
        assert body["cheat_sheet"].startswith("#")

    def test_session_persists(self, client):
        body = _upload(client).json()
        sid = body["session_id"]
        # Session must be reusable for drill-down.
        r2 = client.post("/drill-down",
                         json={"session_id": sid, "takeaway": "فكرة 1"})
        assert r2.status_code == 200

    def test_each_call_creates_new_session(self, client):
        sid1 = _upload(client).json()["session_id"]
        sid2 = _upload(client).json()["session_id"]
        assert sid1 != sid2

    def test_empty_transcript_is_422(self, client, monkeypatch):
        import pipeline
        monkeypatch.setattr(pipeline, "run_whisper", lambda p: "")
        r = _upload(client)
        assert r.status_code == 422

    def test_short_transcript_is_422(self, client, monkeypatch):
        import pipeline
        monkeypatch.setattr(pipeline, "run_whisper", lambda p: "نص قصير")
        # Make build_index return no chunks — simulating a too-short transcript.
        monkeypatch.setattr(pipeline, "build_index", lambda t: ([], None))
        r = _upload(client)
        assert r.status_code == 422


# =============================================================================
# /drill-down
# =============================================================================
class TestDrillDown:

    def test_returns_chunk_for_known_session(self, client):
        sid = _upload(client).json()["session_id"]
        r = client.post("/drill-down",
                        json={"session_id": sid, "takeaway": "فكرة 1"})
        assert r.status_code == 200
        assert r.json()["context"]                 # stub returns chunks[0]

    def test_unknown_session_is_404(self, client):
        r = client.post("/drill-down",
                        json={"session_id": "no-such-session", "takeaway": "x"})
        assert r.status_code == 404

    def test_no_match_is_404(self, client, monkeypatch):
        sid = _upload(client).json()["session_id"]
        # Force drill_down to return empty — simulates no relevant chunk.
        import pipeline
        monkeypatch.setattr(pipeline, "drill_down", lambda *a, **k: "")
        r = client.post("/drill-down",
                        json={"session_id": sid, "takeaway": "فكرة"})
        assert r.status_code == 404

    def test_validates_request_body(self, client):
        # Missing 'takeaway' → 422 from pydantic.
        r = client.post("/drill-down", json={"session_id": "abc"})
        assert r.status_code == 422


# =============================================================================
# /analyze-url
# =============================================================================
class TestAnalyzeUrl:

    def test_returns_full_bundle(self, client):
        r = client.post("/analyze-url",
                        json={"url": "https://youtube.com/watch?v=fake"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) >= {"session_id", "transcript", "cheat_sheet",
                             "takeaways", "pdf_url"}

    def test_session_persists_for_drill_down(self, client):
        r = client.post("/analyze-url",
                        json={"url": "https://youtube.com/watch?v=fake"})
        sid = r.json()["session_id"]
        r2 = client.post("/drill-down",
                         json={"session_id": sid, "takeaway": "فكرة 1"})
        assert r2.status_code == 200

    def test_empty_url_is_400(self, client):
        r = client.post("/analyze-url", json={"url": ""})
        assert r.status_code == 400

    def test_yt_dlp_failure_is_400(self, client):
        # Stub raises RuntimeError when "fail" is in the URL — simulates
        # private video / geo-block / missing ffmpeg / 404.
        r = client.post("/analyze-url",
                        json={"url": "https://youtube.com/watch?v=fail-video"})
        assert r.status_code == 400
        assert "Failed to download" in r.json()["detail"]

    def test_validates_request_body(self, client):
        # Missing 'url' field → 422 from pydantic.
        r = client.post("/analyze-url", json={})
        assert r.status_code == 422


# =============================================================================
# /cheat-sheet/{sid}.pdf
# =============================================================================
class TestPdfDownload:

    def test_unknown_session_is_404(self, client):
        r = client.get("/cheat-sheet/no-such-session.pdf")
        assert r.status_code == 404

    def test_no_pdf_for_session_is_404(self, client):
        # Stub returns pdf=None, so the analyze response will too — calling
        # the download endpoint should 404.
        sid = _upload(client).json()["session_id"]
        r = client.get(f"/cheat-sheet/{sid}.pdf")
        assert r.status_code == 404

    def test_returns_pdf_when_present(self, client, tmp_path, monkeypatch):
        # Patch generate_cheat_sheet to point at a real (tiny) PDF on disk.
        fake_pdf = tmp_path / "fake.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n%fake\n")
        import pipeline
        monkeypatch.setattr(pipeline, "generate_cheat_sheet",
                            lambda c, i: ("# md", str(fake_pdf)))
        sid = _upload(client).json()["session_id"]
        r = client.get(f"/cheat-sheet/{sid}.pdf")
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/pdf"
        assert r.content.startswith(b"%PDF")


# =============================================================================
# /sessions/{sid} (DELETE)
# =============================================================================
class TestSessionLifecycle:

    def test_delete_removes_session(self, client):
        sid = _upload(client).json()["session_id"]
        r = client.delete(f"/sessions/{sid}")
        assert r.status_code == 200
        # Subsequent drill-down should now 404.
        r2 = client.post("/drill-down",
                         json={"session_id": sid, "takeaway": "x"})
        assert r2.status_code == 404

    def test_delete_unknown_is_404(self, client):
        assert client.delete("/sessions/missing").status_code == 404
