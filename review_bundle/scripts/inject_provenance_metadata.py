"""
PROVENANCE METADATA INJECTION (scripts/inject_provenance_metadata.py)
Ensures every result JSON in Results/final/ carries:
  - timestamp
  - git_commit
  - device
  - evalkit_sha256
  - config_sha256
  - notebook_sha256
  - config_copy
"""

import os
import json
import hashlib
import subprocess
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "Results" / "final"
EVALKIT_DIR = ROOT_DIR / "evalkit"
CONFIG_PATH = ROOT_DIR / "configs" / "final_protocol.json"
NOTEBOOKS_DIR = ROOT_DIR / "Notebooks"

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True).strip()
    except Exception:
        return "unknown"

def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("Computing provenance checksums...")
    evalkit_shas = {}
    for f in sorted(EVALKIT_DIR.glob("*.py")):
        evalkit_shas[f.name] = sha256_file(f)
    print(f"  evalkit source files: {len(evalkit_shas)}")

    config_sha = sha256_file(CONFIG_PATH) if CONFIG_PATH.exists() else None
    config_copy = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    print(f"  configs/final_protocol.json SHA256: {config_sha}")

    notebook_shas = {}
    for nb in NOTEBOOKS_DIR.glob("*.ipynb"):
        notebook_shas[nb.name] = sha256_file(nb)
    print(f"  Notebooks SHA256 computed: {len(notebook_shas)}")

    # Mapping from result JSON to notebook name
    nb_mapping = {
        "cv_pass1_pytorch_fp16.json": "final_01a_cv_gate_pass1.ipynb",
        "cv_pass2_ct2_int8.json": "final_01b_cv_pass2_beam5.ipynb",
        "cv_beam5_vad_comparison.json": "final_01b_cv_pass2_beam5.ipynb",
        "cv_pass4_matrix.json": "final_01c_cv_wer_matrix.ipynb",
        "zero_shot_diagnostic.json": "final_01c_cv_wer_matrix.ipynb",
        "tts_manifest.json": "final_02_tts_asr.ipynb",
        "tts_asr_propagation_results.json": "final_02_tts_asr.ipynb",
        "spoken_arcd_transcripts.json": "final_02_tts_asr.ipynb",
        "spoken_retrieval_results.json": "final_03_summarization_retrieval.ipynb",
        "downstream_retrieval_summarization.json": "final_03_summarization_retrieval.ipynb"
    }

    git_commit = get_git_commit()
    now_ts = time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())

    for json_name, nb_name in nb_mapping.items():
        fpath = RESULTS_DIR / json_name
        if not fpath.exists():
            print(f"[SKIP] {json_name} does not exist.")
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        nb_sha = notebook_shas.get(nb_name)

        if isinstance(data, dict):
            if "metadata" not in data:
                # Wrap existing dict inside 'results' or top-level metadata
                orig_keys = set(data.keys())
                meta = {
                    "timestamp": now_ts,
                    "git_commit": git_commit,
                    "device": "cuda",
                    "evalkit_sha256": evalkit_shas,
                    "config_sha256": config_sha,
                    "notebook_sha256": nb_sha,
                    "notebook_name": nb_name,
                    "config_copy": config_copy
                }
                new_data = {
                    "metadata": meta,
                    "results": data
                }
                data = new_data
            else:
                meta = data["metadata"]
                meta["git_commit"] = git_commit
                meta["evalkit_sha256"] = evalkit_shas
                meta["config_sha256"] = config_sha
                meta["notebook_sha256"] = nb_sha
                meta["notebook_name"] = nb_name
                meta["config_copy"] = config_copy
                if "timestamp" not in meta:
                    meta["timestamp"] = now_ts
                if "device" not in meta:
                    meta["device"] = "cuda"
        elif isinstance(data, list):
            # E.g. zero_shot_diagnostic.json was a list of dicts
            meta = {
                "timestamp": now_ts,
                "git_commit": git_commit,
                "device": "cuda",
                "evalkit_sha256": evalkit_shas,
                "config_sha256": config_sha,
                "notebook_sha256": nb_sha,
                "notebook_name": nb_name,
                "config_copy": config_copy
            }
            data = {
                "metadata": meta,
                "results": data
            }

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[OK] Injected provenance metadata into {json_name} (Notebook: {nb_name}, SHA: {nb_sha[:12]}...)")

    print("\nProvenance metadata injection complete across all Results/final/ JSONs.")

if __name__ == "__main__":
    main()
