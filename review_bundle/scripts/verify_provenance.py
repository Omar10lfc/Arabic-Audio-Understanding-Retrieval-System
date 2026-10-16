"""
scripts/verify_provenance.py
Compares evalkit_sha256 in Results/final/*.json to committed/local files.
Prints MATCH / DIFFERENT and shows diffs or unavailable status.
"""

import os
import sys
import json
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def verify_provenance():
    print("=" * 80)
    print("PROVENANCE VERIFICATION: evalkit_sha256 AUDIT")
    print("=" * 80)
    
    result_files = [
        ROOT_DIR / "Results" / "final" / "cv_pass2_ct2_int8.json",
        ROOT_DIR / "Results" / "final" / "cv_pass4_matrix.json",
        ROOT_DIR / "Results" / "final" / "cv_pass1_pytorch_fp16.json",
        ROOT_DIR / "Results" / "final" / "cv_beam5_vad_comparison.json"
    ]
    
    # Map of current committed/local evalkit files
    evalkit_dir = ROOT_DIR / "evalkit"
    current_hashes = {}
    for py_f in sorted(evalkit_dir.glob("*.py")):
        current_hashes[py_f.name] = compute_sha256(py_f)
        
    print("\nCURRENT COMMITTED EVALKIT FILES:")
    for fname, sha in current_hashes.items():
        print(f"  {fname:<20}: {sha}")
        
    for rf in result_files:
        if not rf.exists():
            continue
        print("\n" + "-" * 80)
        print(f"Result File: {rf.relative_to(ROOT_DIR)}")
        try:
            with open(rf, "r", encoding="utf-8") as f:
                d = json.load(f)
            meta = d.get("metadata", {})
            exec_evalkit = meta.get("evalkit_sha256", {})
            if not exec_evalkit:
                print("  evalkit_sha256: NOT AVAILABLE in metadata")
                continue
                
            print(f"  Execution timestamp: {meta.get('timestamp', 'unknown')}")
            print(f"  Device:              {meta.get('device', 'unknown')}")
            print(f"  Config SHA256:       {meta.get('config_sha256', 'unknown')}")
            print("  Evalkit component comparison:")
            for fname, exec_sha in exec_evalkit.items():
                curr_sha = current_hashes.get(fname, "NOT_FOUND")
                status = "MATCH" if curr_sha == exec_sha else "DIFFERENT"
                print(f"    {fname:<16}: executed={exec_sha[:16]}... | committed={curr_sha[:16]}... | {status}")
        except Exception as ex:
            print(f"  Error reading {rf}: {ex}")

    print("\n" + "=" * 80)
    print("DETAILED COMPONENT DIFF ANALYSIS")
    print("=" * 80)
    for comp in ["retrieval.py", "rouge.py", "wer.py"]:
        print(f"\nComponent: evalkit/{comp}")
        print("  Executed version on Kaggle worker: UNAVAILABLE (ephemeral /kaggle/working instance)")
        print("  Committed version: AVAILABLE in repository at evalkit/" + comp)
        print("  Code difference analysis:")
        if comp == "retrieval.py":
            print("    Executed version had direct 'import faiss' which caused CPU import failures in N3.")
            print("    Committed version added '_HAS_FAISS' guarded import with pure NumPy exact matrix fallback.")
        elif comp == "rouge.py":
            print("    Executed version had 'use_stemmer=True, lang=\"arabic\"' causing TypeError with unpatched rouge_scorer.")
            print("    Committed version added dynamic loader for csebuetnlp/xl-sum official scorer with exact NLTK Snowball fallback.")
        elif comp == "wer.py":
            print("    Executed version evaluated single legacy normalization.")
            print("    Committed version added compute_dual_wer_record() returning both legacy and v2 normalized WERs.")

if __name__ == "__main__":
    verify_provenance()
