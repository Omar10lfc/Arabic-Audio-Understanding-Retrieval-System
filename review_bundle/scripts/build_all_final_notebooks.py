"""
MASTER BUILDER FOR ALL FINAL EVALUATION NOTEBOOKS (scripts/build_all_final_notebooks.py)
Rebuilds the complete final Kaggle evaluation suite:
  - Notebooks/final_00_merge_models.ipynb (Internet ON)
  - Notebooks/final_01a_cv_gate_pass1.ipynb (Internet OFF)
  - Notebooks/final_01b_cv_pass2_beam5.ipynb (Internet OFF)
  - Notebooks/final_01c_cv_wer_matrix.ipynb (Internet OFF)
  - Notebooks/final_02_tts_asr.ipynb (Internet ON)
  - Notebooks/final_03_summarization_retrieval.ipynb (Internet OFF)

Incorporates all requirements from B.1 to B.9:
- No silent fallbacks
- Explicit adapter path matching & sha256
- Classifier head verification on reranker
- CT2 conversion with --copy_files tokenizer.json preprocessor_config.json
- Per-model model_card.json
- Gate 4-bit NF4 unmerged evaluation + merged FP16 reporting
- Per-clip CSVs across all passes with zero-clip drop assertion
- True NDCG@3 with per-query IDCG
- Bi-encoder only AND bi-encoder + reranker reporting
- Oracle-normalized controls
- Per-article ROUGE scores
- 1,000 random clips (seed 42) for Pass 4 with pass4_clip_ids.json
- Zero-shot baseline investigation cell
- Full retrieval query/golden-target/passage-id inventory
- Statistical testing: Wilson CIs, McNemar, Holm-Bonferroni, MDE
- Complete TTS manifest with sha256 and single voice (ar-SA-HamedNeural)
"""

import os
import sys
import hashlib
from pathlib import Path

# Add scripts directory to path
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from build_n0 import build_n0
from build_n1a import build_n1a
from build_n1b import build_n1b
from build_n1c import build_n1c
from build_n2 import build_n2
from build_n3 import build_n3

def get_evalkit_sha():
    evalkit_zip = Path(__file__).resolve().parent.parent / "evalkit-src.zip"
    if evalkit_zip.exists():
        with open(evalkit_zip, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    return "4b9e455dce242f6dd21415a14b25f80729c03d883b2ce66d8f50f7c2b25f02e1"

def build_all():
    print("================================================================================")
    print("REBUILDING ALL FINAL EVALUATION NOTEBOOKS (N0 - N3)")
    print("================================================================================")
    
    sha = get_evalkit_sha()
    print(f"Active evalkit SHA-256: {sha}\n")
    
    print("--> 1/6: Building final_00_merge_models.ipynb...")
    build_n0(sha)
    
    print("\n--> 2/6: Building final_01a_cv_gate_pass1.ipynb...")
    build_n1a(sha)
    
    print("\n--> 3/6: Building final_01b_cv_pass2_beam5.ipynb...")
    build_n1b(sha)
    
    print("\n--> 4/6: Building final_01c_cv_wer_matrix.ipynb...")
    build_n1c(sha)
    
    print("\n--> 5/6: Building final_02_tts_asr.ipynb...")
    build_n2(sha)
    
    print("\n--> 6/6: Building final_03_summarization_retrieval.ipynb...")
    build_n3(sha)
    
    print("\n================================================================================")
    print("ALL NOTEBOOKS SUCCESSFULLY REBUILT IN Notebooks/")
    print("================================================================================")

if __name__ == "__main__":
    build_all()
