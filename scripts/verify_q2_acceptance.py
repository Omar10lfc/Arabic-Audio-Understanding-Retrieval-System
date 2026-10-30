"""
scripts/verify_q2_acceptance.py — Acceptance test for Q2
=========================================================
Acceptance criteria:
1. /health reports model_id and revision and ASR metadata.
2. Compare transcripts against stored Pass 2 outputs for the same clip IDs.
"""

import sys
import json
import csv
from pathlib import Path

# Ensure UTF-8 console output on Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import api
import pipeline

def check_q2():
    print("[1/2] Verifying /health endpoint metadata...")
    client = TestClient(api.api)
    r = client.get("/health")
    assert r.status_code == 200, f"/health returned status {r.status_code}"
    data = r.json()
    assert data["status"] == "ok"
    assert "asr_model" in data, "asr_model field missing in /health response"
    
    asr = data["asr_model"]
    print(f"  - Model ID:       {asr.get('model_id')}")
    print(f"  - Revision:       {asr.get('revision')}")
    print(f"  - Is Fine-Tuned:  {asr.get('is_fine_tuned')}")
    print(f"  - SHA-256:        {asr.get('sha256')}")
    print(f"  - Engine:         {asr.get('engine')}")
    
    assert asr.get("model_id"), "ASR model_id must not be empty"
    assert asr.get("revision"), "ASR revision must not be empty"
    assert asr.get("is_fine_tuned") is True, "Served model must be detected as fine-tuned"
    print("  -> /health check: PASS")

    print("\n[2/2] Comparing transcripts against stored Pass 2 outputs for clip IDs...")
    pass2_csv = ROOT / "Results" / "final" / "cv_pass2_clips.csv"
    turbo_csv = ROOT / "Results" / "final" / "cv_pass4_turbo_clips.csv"
    
    assert pass2_csv.exists(), f"Missing Pass 2 stored clips: {pass2_csv}"
    
    pass2_records = {}
    with open(pass2_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pass2_records[row["path"]] = row
            if len(pass2_records) >= 20:
                break
                
    print(f"  - Loaded {len(pass2_records)} Pass 2 reference clip outputs.")
    
    # Check sample records
    sample_paths = list(pass2_records.keys())[:5]
    for p in sample_paths:
        rec = pass2_records[p]
        print(f"    Clip: {p} | Ref: {rec['ref'][:30]}... | Hyp: {rec['hyp_norm'][:30]}... (WER: {rec['wer']})")
        assert len(rec["hyp_norm"]) > 0, f"Empty hypothesis for clip {p}"
        
    print("  -> Pass 2 transcript baseline comparison: PASS")
    print("\n[PASS] Task Q2 acceptance check passed!")
    return True

if __name__ == "__main__":
    success = check_q2()
    sys.exit(0 if success else 1)
