"""
Review Inspection Script (scripts/review_inspect.py)
Gathers raw data, model provenance, literal scans, and configuration facts
for the External Review Packet.
"""

import os
import sys
import json
import hashlib
import glob
import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_md5(filepath):
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def inspect_section_0e():
    print("=" * 80)
    print("SECTION 0.e: Results/final/ File Inventory & Metadata")
    print("=" * 80)
    p = ROOT_DIR / "Results" / "final"
    for f in sorted(p.glob("*")):
        if f.is_file():
            stat = f.stat()
            md5_val = compute_md5(f)
            sha256_val = compute_sha256(f)
            print(f"File: {f.name}")
            print(f"  Path: Results/final/{f.name}")
            print(f"  Bytes: {stat.st_size}")
            print(f"  MTime: {stat.st_mtime}")
            print(f"  MD5: {md5_val}")
            print(f"  SHA256: {sha256_val}")
            
            # Inspect metadata block if json
            if f.suffix == ".json":
                try:
                    with open(f, "r", encoding="utf-8") as jf:
                        d = json.load(jf)
                    meta = d.get("metadata", {})
                    print(f"  Metadata block: {json.dumps(meta, indent=4)}")
                except Exception as ex:
                    print(f"  Metadata block: Error reading ({ex})")
            print()

def inspect_section_0f():
    print("=" * 80)
    print("SECTION 0.f: Literal Scan over Notebooks and evalkit/")
    print("=" * 80)
    
    # Target values: values in Results/**/*.json (3+ sig digits), plus 12.51, 0.6200, 0.5000, 17.63
    targets = ["12.51", "0.6200", "0.5000", "17.63", "13.03", "20.61", "42.69", "13.48", "29.56", "16.08", "33.85", "39.03", "18.64"]
    
    # Search evalkit/
    evalkit_files = list((ROOT_DIR / "evalkit").glob("**/*.py"))
    nb_files = list((ROOT_DIR / "Notebooks").glob("**/*.ipynb")) + list(ROOT_DIR.glob("*.ipynb"))
    
    print("Scanning evalkit/ Python files for hardcoded literals & suspicious assignments:")
    for py_f in evalkit_files:
        content = py_f.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(content.splitlines(), 1):
            for t in targets:
                if t in line and not line.strip().startswith("#"):
                    print(f"  {py_f.relative_to(ROOT_DIR)}:{line_no} -> Match {t}: {line.strip()}")
            if re.search(r'\b(wer|rouge|p1|p3|mrr|throughput)\s*=\s*[0-9]+(\.[0-9]+)?\b', line, re.I):
                print(f"  {py_f.relative_to(ROOT_DIR)}:{line_no} -> Literal assignment: {line.strip()}")

    print()
    print("Scanning Notebook code cells for suspicious hardcoded metric assignments:")
    for nb_f in nb_files:
        try:
            with open(nb_f, "r", encoding="utf-8") as nf:
                nb = json.load(nf)
            for idx, cell in enumerate(nb.get("cells", [])):
                if cell.get("cell_type") == "code":
                    src = "".join(cell.get("source", []))
                    for line_no, line in enumerate(src.splitlines(), 1):
                        for t in targets:
                            if t in line and not line.strip().startswith("#"):
                                # Exclude test assertions or print statements
                                if "=" in line and not "==" in line:
                                    print(f"  {nb_f.relative_to(ROOT_DIR)} [Cell {idx}:{line_no}] -> {line.strip()}")
                        if re.search(r'\b(wer|rouge|p1|p3|mrr|throughput)\s*=\s*[0-9]+(\.[0-9]+)?\b', line, re.I):
                            print(f"  {nb_f.relative_to(ROOT_DIR)} [Cell {idx}:{line_no}] -> Assignment: {line.strip()}")
        except Exception as ex:
            pass
    print()

def inspect_section_0i():
    print("=" * 80)
    print("SECTION 0.i: N1a Gate Check & Filtering Verification")
    print("=" * 80)
    # Check Results/results-Whisper-large-v3-qlora.json
    res_path = ROOT_DIR / "Results" / "results-Whisper-large-v3-qlora.json"
    if res_path.exists():
        print(f"File: Results/results-Whisper-large-v3-qlora.json")
        print(f"Content:\n{res_path.read_text(encoding='utf-8')}")
    else:
        print("Results/results-Whisper-large-v3-qlora.json: NOT AVAILABLE")
        
    # Check cv_pass1_checkpoint.csv first 5 clips
    csv_path = ROOT_DIR / "Results" / "final" / "cv_pass1_checkpoint.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path, nrows=5)
        print("\nFirst 5 clip rows in cv_pass1_checkpoint.csv:")
        for idx, row in df.iterrows():
            print(f"  Clip {idx}: path={row.get('path')}, legacy_wer={row.get('legacy_wer')}, v2_wer={row.get('v2_wer')}")
    print()

def inspect_section_1():
    print("=" * 80)
    print("SECTION 1: MODEL PROVENANCE (CPU CHECKS)")
    print("=" * 80)
    
    # 1.a & 1.b Whisper Medium QLoRA adapter & merged models
    med_dir = ROOT_DIR / "whisper_medium_output" / "whisper-medium-arabic-qlora-stage2"
    if med_dir.exists():
        print(f"Whisper-medium local directory: {med_dir}")
        for item in med_dir.glob("*"):
            if item.is_file():
                print(f"  File: {item.name:<35} | Bytes: {item.stat().st_size:>12}")
            elif item.is_dir():
                print(f"  Dir:  {item.name}/")
                for sub in item.glob("*"):
                    if sub.is_file():
                        print(f"    File: {sub.name:<33} | Bytes: {sub.stat().st_size:>12}")
                        
        ac_file = med_dir / "adapter_config.json"
        if ac_file.exists():
            print(f"\nadapter_config.json:\n{ac_file.read_text(encoding='utf-8')}")
            
        # Check CT2 directory
        ct2_dir = med_dir / "ct2_int8"
        if ct2_dir.exists():
            print(f"\nCT2 directory contents ({ct2_dir}):")
            for cf in ct2_dir.glob("*"):
                print(f"  {cf.name:<30} | {cf.stat().st_size:>10} bytes")
    else:
        print("whisper_medium_output/whisper-medium-arabic-qlora-stage2: NOT AVAILABLE")
        
    # Check Summarizer/
    summ_dir = ROOT_DIR / "Summarizer"
    if summ_dir.exists():
        print(f"\nSummarizer/ directory contents:")
        for sf in summ_dir.glob("*"):
            if sf.is_file():
                print(f"  {sf.name:<35} | {sf.stat().st_size:>12} bytes")
                
    # Check Whisper-Fine-tuned-final-model/
    wft_dir = ROOT_DIR / "Whisper-Fine-tuned-final-model"
    if wft_dir.exists():
        print(f"\nWhisper-Fine-tuned-final-model/ directory contents:")
        for wf in wft_dir.glob("*"):
            if wf.is_file():
                print(f"  {wf.name:<35} | {wf.stat().st_size:>12} bytes")
    print()

def inspect_section_2():
    print("=" * 80)
    print("SECTION 2: DATA INTEGRITY & LEAKAGE")
    print("=" * 80)
    
    # 2.a Common Voice test.tsv
    tsv_p = ROOT_DIR / "scratch" / "test.tsv"
    if tsv_p.exists():
        stat = tsv_p.stat()
        md5_val = compute_md5(tsv_p)
        print(f"Common Voice test.tsv:")
        print(f"  Path: scratch/test.tsv")
        print(f"  Bytes: {stat.st_size}")
        print(f"  MD5: {md5_val}")
        import pandas as pd
        df = pd.read_csv(tsv_p, sep="\t")
        print(f"  Total raw rows in test.tsv: {len(df)}")
        print(f"  Unique client_ids: {df['client_id'].nunique() if 'client_id' in df else 'NOT AVAILABLE'}")
        
        # Check train.tsv
        train_p = ROOT_DIR / "scratch" / "train.tsv"
        dev_p = ROOT_DIR / "scratch" / "dev.tsv"
        print(f"  train.tsv exists: {train_p.exists()} ({'NOT AVAILABLE' if not train_p.exists() else 'AVAILABLE'})")
        print(f"  dev.tsv exists:   {dev_p.exists()} ({'NOT AVAILABLE' if not dev_p.exists() else 'AVAILABLE'})")
    else:
        print("scratch/test.tsv: NOT AVAILABLE")
        
    # Check ARCD data
    arcd_files = list(ROOT_DIR.glob("**/*arcd*")) + list(ROOT_DIR.glob("**/train.csv"))
    print("\nARCD files found in workspace:")
    for af in arcd_files:
        if af.is_file() and not ".system_generated" in str(af):
            print(f"  {af.relative_to(ROOT_DIR)} ({af.stat().st_size} bytes)")
    print()

if __name__ == "__main__":
    inspect_section_0e()
    inspect_section_0f()
    inspect_section_0i()
    inspect_section_1()
    inspect_section_2()
