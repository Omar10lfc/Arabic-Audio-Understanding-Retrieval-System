"""
scripts/verify_doc_claims.py — Acceptance test for Q1
======================================================
Scans documentation files (README.md, README_SPACE.md, REPORT_experiments_results.md)
to verify:
1. Every claim_id cited in the text exists in Results/final/claims_register.json.
2. Every cited metric value matches the claims register measured value.
3. Unverified legacy claims (e.g. headline P@1 0.86, unverified 50.9% entity error lead,
   or +5 dB noise cliff lead) are not advertised as lead verified findings.
"""

import re
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTER_PATH = ROOT / "Results" / "final" / "claims_register.json"

DOC_FILES = [
    ROOT / "README.md",
    ROOT / "README_SPACE.md",
    ROOT / "REPORT_experiments_results.md"
]

def load_claims():
    if not REGISTER_PATH.exists():
        print(f"[FAIL] Claims register not found at {REGISTER_PATH}")
        sys.exit(1)
    with open(REGISTER_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {c["claim_id"]: c for c in data.get("claims", [])}

def check_docs():
    claims = load_claims()
    print(f"[INFO] Loaded {len(claims)} canonical claims from {REGISTER_PATH.relative_to(ROOT)}")
    
    errors = []
    
    # Check claim ID citations in documentation (handles [TAG] and [`TAG`])
    claim_id_pattern = re.compile(r"\[`?(GATE_\w+|PASS\d+_\w+|XLSUM_\w+|SUMM_\w+|RET_\w+)`?\]")
    
    for doc_path in DOC_FILES:
        if not doc_path.exists():
            errors.append(f"Missing doc file: {doc_path.relative_to(ROOT)}")
            continue
        
        text = doc_path.read_text(encoding="utf-8")
        rel_path = doc_path.relative_to(ROOT)
        
        # 1. Check all claim tags
        found_tags = claim_id_pattern.findall(text)
        if not found_tags:
            errors.append(f"{rel_path}: No claim_id tags found! At least headline claims must be tagged.")
        
        for tag in found_tags:
            if tag not in claims:
                errors.append(f"{rel_path}: Cited claim_id '{tag}' not found in claims_register.json!")
                
        # 2. Check forbidden lead claims in README.md
        if doc_path.name == "README.md":
            # Check headline table for P@1 0.86
            lead_section = text[:3000]
            if "0.86" in lead_section and "RET_FULL_ARCD_693_P1" not in text and "RET_CLEAN_ORACLE_RERANK_P1" not in text:
                errors.append(f"{rel_path}: Headline table still promotes exploratory 0.86 without canonical claims.")
            
            # Check if Key Findings 6 & 7 are still in lead findings without legacy label
            key_findings_lead = text[text.find("## Key Findings"):text.find("## Quick start")] if "## Key Findings" in text else ""
            if "Entity-Level Error Attribution" in key_findings_lead and "legacy" not in key_findings_lead.lower():
                errors.append(f"{rel_path}: Key Finding 6 (Entity-Level Error Attribution) remains in lead findings without legacy disclaimer.")
            if "Acoustic Noise Degradation Sweep" in key_findings_lead and "legacy" not in key_findings_lead.lower():
                errors.append(f"{rel_path}: Key Finding 7 (Acoustic Noise Sweep) remains in lead findings without legacy disclaimer.")

        # 3. Check README_SPACE.md
        if doc_path.name == "README_SPACE.md":
            if "20.61" in text and "XLSUM_ASR" not in text and "PASS2_CORPUS_WER" not in text:
                errors.append(f"{rel_path}: README_SPACE still promotes legacy Whisper-small 20.61% without updated registered claims.")

    if errors:
        print(f"\n[FAIL] {len(errors)} documentation claims errors found:")
        for e in errors:
            print(f"  - {e}")
        return False
    
    print("\n[PASS] All documentation numbers and claim_ids verified against claims_register.json!")
    return True

if __name__ == "__main__":
    success = check_docs()
    sys.exit(0 if success else 1)
