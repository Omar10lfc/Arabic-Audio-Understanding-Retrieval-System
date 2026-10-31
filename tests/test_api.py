"""
FastAPI integration tests using TestClient + monkeypatched pipeline. No real
models are loaded; the heavy functions are replaced with deterministic stubs in
`conftest.py::stub_pipeline`. Whole suite runs in ~1 second.
"""

import io
import time

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
        assert "asr_model" in body
        assert "model_id" in body["asr_model"]
        assert "revision" in body["asr_model"]
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

    def test_disabled_by_default(self, client, monkeypatch):
        monkeypatch.delenv("ALLOW_URL_INGEST", raising=False)
        r = client.post("/analyze-url", json={"url": "https://youtube.com/watch?v=fake"})
        assert r.status_code == 403
        assert "disabled" in r.json()["detail"].lower()

    def test_returns_full_bundle(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        r = client.post("/analyze-url",
                        json={"url": "https://youtube.com/watch?v=fake"})
        assert r.status_code == 200
        body = r.json()
        assert set(body) >= {"session_id", "transcript", "cheat_sheet",
                             "takeaways", "pdf_url"}

    def test_session_persists_for_drill_down(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        r = client.post("/analyze-url",
                        json={"url": "https://youtube.com/watch?v=fake"})
        sid = r.json()["session_id"]
        r2 = client.post("/drill-down",
                         json={"session_id": sid, "takeaway": "فكرة 1"})
        assert r2.status_code == 200

    def test_empty_url_is_400(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        r = client.post("/analyze-url", json={"url": ""})
        assert r.status_code == 400

    def test_yt_dlp_failure_is_400(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        # Stub raises RuntimeError when "fail" is in the URL — simulates
        # private video / geo-block / missing ffmpeg / 404.
        r = client.post("/analyze-url",
                        json={"url": "https://youtube.com/watch?v=fail-video"})
        assert r.status_code == 400
        assert "Failed to download" in r.json()["detail"]

    def test_validates_request_body(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        # Missing 'url' field → 422 from pydantic.
        r = client.post("/analyze-url", json={})
        assert r.status_code == 422

    def test_blocks_loopback_ip(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        r = client.post("/analyze-url", json={"url": "http://127.0.0.1/audio.mp3"})
        assert r.status_code == 400
        assert "forbidden" in r.json()["detail"].lower()

    def test_blocks_private_subnet_ip(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        r = client.post("/analyze-url", json={"url": "http://192.168.1.1/lecture.wav"})
        assert r.status_code == 400
        assert "forbidden" in r.json()["detail"].lower()

    def test_blocks_aws_metadata_ip(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        r = client.post("/analyze-url", json={"url": "http://169.254.169.254/latest/meta-data/"})
        assert r.status_code == 400
        assert "forbidden" in r.json()["detail"].lower()

    def test_blocks_invalid_scheme(self, client, monkeypatch):
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        r = client.post("/analyze-url", json={"url": "ftp://example.com/audio.mp3"})
        assert r.status_code == 400
        assert "scheme" in r.json()["detail"].lower()

    def test_rate_limiting(self, client, monkeypatch):
        import api
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        monkeypatch.setattr(api, "_RATE_LIMIT_MAX_REQ", 2)
        api._RATE_LIMIT_STORE.clear()

        # Health is bypassed
        r1 = client.get("/health")
        assert r1.status_code == 200

        # Normal requests
        r2 = client.post("/analyze-url", json={"url": "ftp://bad1"})
        assert r2.status_code == 400

        r3 = client.post("/analyze-url", json={"url": "ftp://bad2"})
        assert r3.status_code == 400

        # Exceeds max 2 requests
        r4 = client.post("/analyze-url", json={"url": "ftp://bad3"})
        assert r4.status_code == 429
        assert "Retry-After" in r4.headers
        assert "slow down" in r4.json()["detail"].lower()

    def test_rate_limiting_forwarded_for(self, client, monkeypatch):
        import api
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        monkeypatch.setattr(api, "_RATE_LIMIT_MAX_REQ", 1)
        api._RATE_LIMIT_STORE.clear()

        # IP A hits limit
        r1 = client.post("/analyze-url", json={"url": "ftp://ip-a-1"}, headers={"x-forwarded-for": "198.51.100.1"})
        assert r1.status_code == 400
        r2 = client.post("/analyze-url", json={"url": "ftp://ip-a-2"}, headers={"x-forwarded-for": "198.51.100.1"})
        assert r2.status_code == 429

        # IP B has independent window
        r3 = client.post("/analyze-url", json={"url": "ftp://ip-b-1"}, headers={"x-forwarded-for": "198.51.100.2"})
        assert r3.status_code == 400

    def test_rate_limiting_spoofed_header(self, client, monkeypatch):
        import api
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        monkeypatch.setattr(api, "_RATE_LIMIT_MAX_REQ", 1)
        api._RATE_LIMIT_STORE.clear()

        # Attacker tries to spoof client IP by prepending fake IPs to X-Forwarded-For
        # Real proxy appends the verified IP '203.0.113.50' at the end of the header
        r1 = client.post("/analyze-url", json={"url": "ftp://req1"},
                         headers={"x-forwarded-for": "1.1.1.1, 203.0.113.50"})
        assert r1.status_code == 400

        # Second request with a different spoofed leading IP but same proxy edge IP
        r2 = client.post("/analyze-url", json={"url": "ftp://req2"},
                         headers={"x-forwarded-for": "2.2.2.2, 203.0.113.50"})
        assert r2.status_code == 429

    def test_rate_limiting_keyed_by_api_key(self, client, monkeypatch):
        import api
        monkeypatch.setenv("ALLOW_URL_INGEST", "1")
        monkeypatch.setattr(api, "_RATE_LIMIT_MAX_REQ", 1)
        api._RATE_LIMIT_STORE.clear()

        # Same IP, different API keys
        r1 = client.post("/analyze-url", json={"url": "ftp://req1"},
                         headers={"x-api-key": "key-alpha", "x-forwarded-for": "198.51.100.1"})
        assert r1.status_code == 400

        # Key alpha hits 429
        r2 = client.post("/analyze-url", json={"url": "ftp://req2"},
                         headers={"x-api-key": "key-alpha", "x-forwarded-for": "198.51.100.1"})
        assert r2.status_code == 429

        # Key beta succeeds even from the same IP
        r3 = client.post("/analyze-url", json={"url": "ftp://req3"},
                         headers={"x-api-key": "key-beta", "x-forwarded-for": "198.51.100.1"})
        assert r3.status_code == 400


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


# =============================================================================
# Task B1: API Key Auth, Upload Cap, Audio MIME Allowlist
# =============================================================================
class TestApiKeyAuth:

    def test_auth_bypassed_when_env_unset(self, client, monkeypatch):
        monkeypatch.delenv("API_BEARER_TOKEN", raising=False)
        r = client.get("/health")
        assert r.status_code == 200
        r_post = client.post(
            "/transcribe",
            files={"audio": ("a.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
        )
        assert r_post.status_code == 200

    def test_health_always_public_even_with_token(self, client, monkeypatch):
        monkeypatch.setenv("API_BEARER_TOKEN", "prod-secret-123")
        r = client.get("/health")
        assert r.status_code == 200

    def test_protected_endpoint_rejects_missing_auth(self, client, monkeypatch):
        monkeypatch.setenv("API_BEARER_TOKEN", "prod-secret-123")
        r = client.post(
            "/transcribe",
            files={"audio": ("a.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
        )
        assert r.status_code == 401
        assert "Unauthorized" in r.json()["detail"]

    def test_protected_endpoint_rejects_wrong_key(self, client, monkeypatch):
        monkeypatch.setenv("API_BEARER_TOKEN", "prod-secret-123")
        r = client.post(
            "/transcribe",
            files={"audio": ("a.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
            headers={"x-api-key": "wrong-secret"}
        )
        assert r.status_code == 401

    def test_protected_endpoint_accepts_valid_x_api_key(self, client, monkeypatch):
        monkeypatch.setenv("API_BEARER_TOKEN", "prod-secret-123")
        r = client.post(
            "/transcribe",
            files={"audio": ("a.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
            headers={"x-api-key": "prod-secret-123"}
        )
        assert r.status_code == 200

    def test_protected_endpoint_accepts_valid_bearer_token(self, client, monkeypatch):
        monkeypatch.setenv("API_BEARER_TOKEN", "prod-secret-123")
        r = client.post(
            "/transcribe",
            files={"audio": ("a.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
            headers={"authorization": "Bearer prod-secret-123"}
        )
        assert r.status_code == 200


class TestUploadValidation:

    def test_invalid_mime_type_returns_415(self, client):
        r = client.post(
            "/transcribe",
            files={"audio": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
        )
        assert r.status_code == 415
        assert "Unsupported media type" in r.json()["detail"]

    def test_allowed_mime_types(self, client):
        for mime in ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/x-m4a", "audio/m4a", "audio/webm"]:
            r = client.post(
                "/transcribe",
                files={"audio": ("sample.audio", io.BytesIO(_FAKE_WAV), mime)},
            )
            assert r.status_code == 200

    def test_upload_size_limit_exceeded_returns_413(self, client, monkeypatch):
        import api
        monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 50)
        oversized = b"x" * 60
        r = client.post(
            "/transcribe",
            files={"audio": ("big.wav", io.BytesIO(oversized), "audio/wav")},
        )
        assert r.status_code == 413
        assert "exceeds maximum permitted size" in r.json()["detail"]


# =============================================================================
# Task B2: Session Store with TTL & Max Capacity
# =============================================================================
class TestSessionStoreTTLAndCapacity:

    def test_session_records_created_at(self, client):
        import api
        t0 = time.time()
        sid = _upload(client).json()["session_id"]
        assert sid in api._SESSIONS
        assert sid in api._RUNTIME
        assert api._SESSIONS[sid].created_at >= t0
        assert api._RUNTIME[sid]["created_at"] >= t0

    def test_session_ttl_eviction_on_drilldown(self, client, monkeypatch):
        import api
        monkeypatch.setattr(api, "SESSION_TTL_SECONDS", 10)
        sid = _upload(client).json()["session_id"]

        # Backdate created_at to 20s ago
        api._SESSIONS[sid].created_at = time.time() - 20
        api._RUNTIME[sid]["created_at"] = time.time() - 20

        # Drill-down should evict it and return 404
        r = client.post("/drill-down", json={"session_id": sid, "takeaway": "فكرة 1"})
        assert r.status_code == 404
        assert "Unknown or expired" in r.json()["detail"]
        assert sid not in api._SESSIONS
        assert sid not in api._RUNTIME

    def test_session_ttl_eviction_on_pdf_download(self, client, monkeypatch):
        import api
        monkeypatch.setattr(api, "SESSION_TTL_SECONDS", 10)
        sid = _upload(client).json()["session_id"]

        api._SESSIONS[sid].created_at = time.time() - 20
        api._RUNTIME[sid]["created_at"] = time.time() - 20

        r = client.get(f"/cheat-sheet/{sid}.pdf")
        assert r.status_code == 404
        assert "Unknown or expired" in r.json()["detail"]

    def test_session_max_capacity_evicts_oldest_first(self, client, monkeypatch):
        import api
        monkeypatch.setattr(api, "MAX_SESSIONS", 3)
        api._SESSIONS.clear()
        api._RUNTIME.clear()

        now = time.time()
        # Create session 1 (oldest: 300s ago)
        s1 = _upload(client).json()["session_id"]
        api._SESSIONS[s1].created_at = now - 300

        # Create session 2 (200s ago)
        s2 = _upload(client).json()["session_id"]
        api._SESSIONS[s2].created_at = now - 200

        # Create session 3 (100s ago)
        s3 = _upload(client).json()["session_id"]
        api._SESSIONS[s3].created_at = now - 100

        assert len(api._SESSIONS) == 3

        # Create session 4 -> should evict s1 (oldest, now - 300)
        s4 = _upload(client).json()["session_id"]
        assert len(api._SESSIONS) == 3
        assert s1 not in api._SESSIONS
        assert s1 not in api._RUNTIME
        assert s2 in api._SESSIONS
        assert s3 in api._SESSIONS
        assert s4 in api._SESSIONS

    def test_session_eviction_removes_temp_pdf(self, client, tmp_path, monkeypatch):
        import api
        fake_pdf = tmp_path / "test_session_evict.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4\n")
        import pipeline
        monkeypatch.setattr(pipeline, "generate_cheat_sheet",
                            lambda c, i: ("# md", str(fake_pdf)))
        monkeypatch.setattr(api, "SESSION_TTL_SECONDS", 5)

        sid = _upload(client).json()["session_id"]
        assert fake_pdf.exists()

        # Age the session
        api._SESSIONS[sid].created_at = time.time() - 10
        # Trigger eviction
        api.evict_expired_sessions()
        assert not fake_pdf.exists()


# =============================================================================
# Task B3: Spawn-and-Poll Async Processing for Long Lectures
# =============================================================================
class TestAsyncSpawnAndPoll:

    def test_analyze_async_returns_202_and_poll_url(self, client):
        r = client.post(
            "/analyze-async",
            files={"audio": ("lecture.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
        )
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert "poll_url" in body
        assert body["poll_url"] == f"/jobs/{body['job_id']}"

    def test_poll_job_returns_completed_result(self, client):
        r = client.post(
            "/analyze-async",
            files={"audio": ("lecture.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
        )
        job_id = r.json()["job_id"]
        poll_r = client.get(f"/jobs/{job_id}")
        assert poll_r.status_code == 200
        poll_body = poll_r.json()
        assert poll_body["job_id"] == job_id
        assert poll_body["status"] == "completed"
        assert poll_body["result"] is not None
        assert "session_id" in poll_body["result"]
        assert "cheat_sheet" in poll_body["result"]

    def test_poll_unknown_job_is_404(self, client):
        assert client.get("/jobs/no-such-job").status_code == 404

    def test_job_ttl_eviction(self, client, monkeypatch):
        import api
        monkeypatch.setattr(api, "SESSION_TTL_SECONDS", 10)
        r = client.post(
            "/analyze-async",
            files={"audio": ("lecture.wav", io.BytesIO(_FAKE_WAV), "audio/wav")},
        )
        job_id = r.json()["job_id"]
        # Age the job
        api._JOBS[job_id]["created_at"] = time.time() - 20
        # Polling should trigger eviction and 404
        assert client.get(f"/jobs/{job_id}").status_code == 404



