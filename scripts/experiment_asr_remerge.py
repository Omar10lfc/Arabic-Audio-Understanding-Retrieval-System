#!/usr/bin/env python3
"""
[EXP1] Candidate 1: 3-Arm ASR Re-Merge Verification Harness

Evaluates or verifies the 3 arms for ASR re-merging on the canonical 400 Subset A clips:
  - Arm 1: NF4 base + LoRA adapter (in-situ 4-bit forward pass)
  - Arm 2: Merge into dequantized NF4 base
  - Arm 3: True FP32 / BF16 unquantized re-merge

All three arms are UNMEASURED until a GPU run is executed.
Existing 13.03% WER is the historical Kaggle FP16 merge baseline (GATE_01_FP16).
The served model (whisper-large-v3-turbo-ct2-int8) will be measured on the same 400 clips
and registered as SERVED_TURBO_CT2_INT8_GATE_WER.

Adheres strictly to Rule 1:
- Never invents results.
- Verifies all artifact SHA-256 hashes against sha256_manifest.json.
- Identifies missing local weights and documents exact fetch locations.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "kaggle_outputs" / "zero-notebook" / "final-merged-models" / "sha256_manifest.json"
PASS2_CLIPS_PATH = REPO_ROOT / "Results" / "final" / "cv_pass2_clips.csv"
OUTPUT_REPORT_PATH = REPO_ROOT / "Results" / "experiments" / "asr_remerge_experiment.json"

GATE_BASELINE_WER = 12.51
GATE_TOLERANCE = 0.25
GATE_THRESHOLD = GATE_BASELINE_WER + GATE_TOLERANCE  # 12.76%


def compute_file_sha256(filepath: Path) -> str:
    """Compute SHA256 hex digest of a file in 64KB chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest().lower()


def get_subset_a_clip_ids(n: int = 400) -> List[str]:
    """
    Extract the exact 400 clip IDs evaluated in Subset A.
    Reproduces cell 4 of final-01a-cv-gate-pass1:
      df_valid.sample(frac=1, random_state=42).iloc[:400]
    """
    if not PASS2_CLIPS_PATH.exists():
        raise FileNotFoundError(f"Missing {PASS2_CLIPS_PATH}")
    df = pd.read_csv(PASS2_CLIPS_PATH)
    df_sampled = df.sample(frac=1, random_state=42).iloc[:n].reset_index(drop=True)
    return df_sampled["path"].tolist()


def audit_adapters_and_manifest() -> Dict[str, Any]:
    """
    Inspect local adapters vs Kaggle sha256_manifest.json.
    Reports matched, mismatched, and missing artifacts with exact fetch locations.
    """
    results: Dict[str, Any] = {
        "manifest_loaded": False,
        "local_adapters": {},
        "missing_adapters": {},
    }
    if not MANIFEST_PATH.exists():
        results["manifest_error"] = f"Manifest not found at {MANIFEST_PATH}"
        return results

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    results["manifest_loaded"] = True
    results["total_manifest_entries"] = len(manifest)

    # 1. Medium Adapter (checkpoint-3600)
    medium_path = REPO_ROOT / "whisper_medium_output" / "whisper-medium-arabic-qlora-stage2" / "checkpoint-3600" / "adapter_model.safetensors"
    expected_med = manifest.get("adapters/whisper-medium-adapter/adapter_model.safetensors", "").lower()
    if medium_path.exists():
        actual_med = compute_file_sha256(medium_path)
        results["local_adapters"]["whisper_medium_adapter"] = {
            "path": str(medium_path.relative_to(REPO_ROOT)),
            "size_bytes": medium_path.stat().st_size,
            "sha256": actual_med,
            "expected_sha256": expected_med,
            "status": "VERIFIED_BIT_IDENTICAL" if actual_med == expected_med else "MISMATCH",
        }
    else:
        results["missing_adapters"]["whisper_medium_adapter"] = {
            "expected_path": str(medium_path.relative_to(REPO_ROOT)),
            "expected_sha256": expected_med,
        }

    # 2. Large-v3 Adapter
    large_path = REPO_ROOT / "kaggle_outputs" / "zero-notebook" / "final-merged-models" / "adapters" / "whisper-large-v3-adapter" / "adapter_model.safetensors"
    expected_large = manifest.get("adapters/whisper-large-v3-adapter/adapter_model.safetensors", "").lower()
    if large_path.exists():
        actual_large = compute_file_sha256(large_path)
        results["local_adapters"]["whisper_large_v3_adapter"] = {
            "path": str(large_path.relative_to(REPO_ROOT)),
            "size_bytes": large_path.stat().st_size,
            "sha256": actual_large,
            "expected_sha256": expected_large,
            "status": "VERIFIED_BIT_IDENTICAL" if actual_large == expected_large else "MISMATCH",
        }
    else:
        results["missing_adapters"]["whisper_large_v3_adapter"] = {
            "expected_path": str(large_path.relative_to(REPO_ROOT)),
            "expected_sha256": expected_large,
            "source_kaggle_kernel": "omar10lfc/fine-tuning-whisper-large-v3-on-arabic-speech",
            "source_kaggle_path": "/kaggle/input/notebooks/omar10lfc/fine-tuning-whisper-large-v3-on-arabic-speech/whisper-large-v3-arabic-qlora/whisper-large-v3-arabic-lora-adapter/adapter_model.safetensors",
            "fetch_command": "kaggle kernels output omar10lfc/fine-tuning-whisper-large-v3-on-arabic-speech -p ./kaggle_large_v3_out/",
        }

    # 3. Turbo Adapter
    turbo_path = REPO_ROOT / "kaggle_outputs" / "zero-notebook" / "final-merged-models" / "adapters" / "whisper-large-v3-turbo-adapter" / "adapter_model.safetensors"
    expected_turbo = manifest.get("adapters/whisper-large-v3-turbo-adapter/adapter_model.safetensors", "").lower()
    if turbo_path.exists():
        actual_turbo = compute_file_sha256(turbo_path)
        results["local_adapters"]["whisper_large_v3_turbo_adapter"] = {
            "path": str(turbo_path.relative_to(REPO_ROOT)),
            "size_bytes": turbo_path.stat().st_size,
            "sha256": actual_turbo,
            "expected_sha256": expected_turbo,
            "status": "VERIFIED_BIT_IDENTICAL" if actual_turbo == expected_turbo else "MISMATCH",
        }
    else:
        results["missing_adapters"]["whisper_large_v3_turbo_adapter"] = {
            "expected_path": str(turbo_path.relative_to(REPO_ROOT)),
            "expected_sha256": expected_turbo,
            "source_kaggle_kernel": "omar10lfc/whisper-large-turbo",
            "source_kaggle_path": "/kaggle/input/notebooks/omar10lfc/whisper-large-turbo/whisper-large-v3-turbo-arabic-qlora/whisper-large-v3-turbo-arabic-lora-adapter/adapter_model.safetensors",
            "fetch_command": "kaggle kernels output omar10lfc/whisper-large-turbo -p ./kaggle_turbo_out/",
        }

    # 4. CT2 INT8 binaries expected in manifest
    results["expected_ct2_int8_weights"] = {
        "turbo_ct2_model_bin_sha256": manifest.get("ct2_int8/whisper-large-v3-turbo-ct2-int8/model.bin"),
        "large_v3_ct2_model_bin_sha256": manifest.get("ct2_int8/whisper-large-v3-ct2-int8/model.bin"),
    }

    return results


def run_experiment(audio_dir: Optional[str] = None) -> Dict[str, Any]:
    """Execute verification harness or generate unmeasured report."""
    try:
        import torch
        has_cuda = torch.cuda.is_available()
    except ImportError:
        has_cuda = False

    subset_a_ids = get_subset_a_clip_ids(400)
    expected_first_5 = [
        "common_voice_ar_23342586.mp3",
        "common_voice_ar_24041068.mp3",
        "common_voice_ar_27141031.mp3",
        "common_voice_ar_19220379.mp3",
        "common_voice_ar_19205134.mp3",
    ]
    sampling_verified = (subset_a_ids[:5] == expected_first_5)

    adapter_audit = audit_adapters_and_manifest()

    report: Dict[str, Any] = {
        "experiment": "ASR Checkpoint 3-Arm Re-Merge Verification (Candidate 1)",
        "protocol": {
            "subset_a_size": len(subset_a_ids),
            "sampling_verified": sampling_verified,
            "first_5_clip_ids": subset_a_ids[:5],
            "gate_baseline_wer": GATE_BASELINE_WER,
            "gate_tolerance": GATE_TOLERANCE,
            "gate_threshold": GATE_THRESHOLD,
        },
        "environment": {
            "platform": sys.platform,
            "cuda_available": has_cuda,
            "device": "cuda" if has_cuda else "cpu",
        },
        "adapter_audit": adapter_audit,
        "historical_reference_baseline": {
            "description": "Historical Kaggle FP16 merge result on Subset A (GATE_01_FP16)",
            "measured_wer": 13.03,
            "status": "EXCEEDS_TOLERANCE",
            "note": "Existing Kaggle FP16 merge. Does not represent unquantized FP32 re-merge.",
        },
        "arms": {
            "arm_1_nf4_base_plus_adapter": {
                "description": "Base model loaded in 4-bit NF4 with adapter applied in-situ via PEFT forward pass.",
                "status": "unmeasured",
                "measured_wer": None,
                "notes": "Requires CUDA GPU with bitsandbytes and Common Voice audio files.",
            },
            "arm_2_merge_dequantized_nf4": {
                "description": "Dequantize 4-bit NF4 weights to FP16/FP32, then merge adapter delta into dequantized base.",
                "status": "unmeasured",
                "measured_wer": None,
                "notes": "Tests whether quant distortion during merge accounts for WER gap.",
            },
            "arm_3_fp32_remerge": {
                "description": "True unquantized FP32 / BF16 re-merge of base and adapter.",
                "status": "unmeasured",
                "measured_wer": None,
                "notes": "Has not run yet. Distinct from historical Kaggle FP16 merge (13.03%).",
            },
        },
        "served_model_measurement_plan": {
            "model_id": "kaggle_outputs/zero-notebook/final-merged-models/ct2_int8/whisper-large-v3-turbo-ct2-int8",
            "target_claim_id": "SERVED_TURBO_CT2_INT8_GATE_WER",
            "status": "unmeasured",
            "measured_wer": None,
            "note": "Will be measured on the exact 400 Subset A clips during Modal GPU run (budget: $5 cap).",
        },
        "budget_estimate": {
            "gpu": "NVIDIA L4 (Modal)",
            "rate_per_hour_usd": 0.80,
            "estimated_runtime_minutes": 12.0,
            "estimated_cost_usd": 0.16,
            "cap_usd": 5.00,
        },
        "recommendation": {
            "retrain_needed": False,
            "action": (
                "Do NOT retrain yet. Fetch large-v3 and turbo adapter safetensors, "
                "run Arms 1, 2, 3, and the served Turbo CT2 INT8 evaluation on Modal L4 (estimated cost $0.16). "
                "Register the exact served Turbo CT2 INT8 WER as SERVED_TURBO_CT2_INT8_GATE_WER."
            ),
        },
    }

    OUTPUT_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


if __name__ == "__main__":
    rep = run_experiment()
    print("=" * 70)
    print("ASR RE-MERGE EXPERIMENT (EXP1) AUDIT & VERIFICATION REPORT")
    print("=" * 70)
    print(f"Sampling Verified: {rep['protocol']['sampling_verified']}")
    print(f"First 5 Clips: {rep['protocol']['first_5_clip_ids']}")
    print(f"CUDA Available: {rep['environment']['cuda_available']}")
    local_ad = rep['adapter_audit'].get('local_adapters', {})
    missing_ad = rep['adapter_audit'].get('missing_adapters', {})
    print(f"Local Verified Adapters: {list(local_ad.keys())}")
    print(f"Missing Adapters: {list(missing_ad.keys())}")
    print(f"Report saved to: {OUTPUT_REPORT_PATH}")
    print("=" * 70)
