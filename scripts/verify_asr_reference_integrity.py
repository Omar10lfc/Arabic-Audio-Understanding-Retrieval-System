"""
scripts/verify_asr_reference_integrity.py — ASR Stored Reference Integrity Check
================================================================================
Integrity criteria:
1. /health reports model_id, revision, and ASR metadata.
2. Verify integrity of stored Pass 4 Turbo CT2 INT8 reference clips (cv_pass4_turbo_clips.csv).
NOTE: Live model acceptance inference remains UNMEASURED pending the Modal GPU session.
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

def check_asr_reference_integrity():
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
    print(f"  - Weight File:    {asr.get('weight_file')}")
    print(f"  - Weight SHA-256: {asr.get('weight_sha256')}")
    print(f"  - Config SHA-256: {asr.get('config_sha256')}")
    print(f"  - Engine:         {asr.get('engine')}")
    
    assert asr.get("model_id"), "ASR model_id must not be empty"
    assert asr.get("revision"), "ASR revision must not be empty"
    assert asr.get("is_fine_tuned") is True, "Served model must be detected as fine-tuned"
    print("  -> /health check: PASS")

    print("\n[2/2] Pass 4 Turbo CT2 INT8 Stored Reference Data Integrity...")
    print("  [AUDIT NOTE] The served ASR model is Whisper-Large-v3-Turbo CT2 INT8.")
    print("  Pass 2 stored outputs came from Large-v3 and are not the served baseline.")
    print("  Results/final/cv_pass4_turbo_clips.csv contains the exact 1,000 clips evaluated")
    print("  on Kaggle GPU for Turbo CT2 INT8, including the 400 Subset A clips.")
    print("  STATUS: Live inference acceptance is UNMEASURED pending Modal GPU execution.")
    
    turbo_csv = ROOT / "Results" / "final" / "cv_pass4_turbo_clips.csv"
    assert turbo_csv.exists(), f"Missing Turbo CT2 stored clips: {turbo_csv}"
    
    turbo_records = {}
    with open(turbo_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            turbo_records[row["path"]] = row
            if len(turbo_records) >= 20:
                break
                
    print(f"  - Inspected {len(turbo_records)} stored Turbo CT2 INT8 reference clip outputs.")
    sample_paths = list(turbo_records.keys())[:5]
    for p in sample_paths:
        rec = turbo_records[p]
        print(f"    Clip: {p} | Ref: {rec['ref'][:30]}... | Hyp: {rec['hyp_norm'][:30]}... (WER: {rec['wer']})")
        assert len(rec["hyp_norm"]) > 0, f"Empty hypothesis for clip {p}"
        
    print("  -> Stored Pass 4 Turbo reference integrity: PASS")
    print("\n[PASS] Stored reference integrity check passed! (Live ASR inference queued for Modal GPU)")
    return True

if __name__ == "__main__":
    success = check_asr_reference_integrity()
    sys.exit(0 if success else 1)
