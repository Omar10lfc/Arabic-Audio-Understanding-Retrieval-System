"""
Model and Data Integrity Checks (scripts/review_models_check.py)
Checks:
1. Merged models vs base differences (1.b)
2. CT2 directory files check (1.c)
3. Reranker classifier weights and scoring on 5 fixed pairs (1.d)
4. CAMeL-BERT embeddings on 5 fixed texts (1.d)
5. AraBART output on 3 fixed texts (1.d)
6. Checkpoint sha256 comparison vs published (1.e)
7. Minimum detectable P@1 effect size (5.b)
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import numpy as np
import torch

if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def check_1b_merged_vs_base():
    print("=" * 80)
    print("CHECK 1.b: Merged Model vs Base Difference (Whisper-medium)")
    print("=" * 80)
    from safetensors.torch import load_file
    merged_path = ROOT_DIR / "whisper_medium_output" / "whisper-medium-arabic-qlora-stage2" / "whisper-medium-arabic-stage2-final-merged" / "model.safetensors"
    
    # Base model path in cache
    cache_base_dir = Path(os.path.expanduser("~/.cache/huggingface/hub/models--openai--whisper-medium/snapshots"))
    base_file = None
    if cache_base_dir.exists():
        for snap in cache_base_dir.glob("*"):
            cand = snap / "model.safetensors"
            if cand.exists():
                base_file = cand
                break
                
    if not merged_path.exists():
        print(f"Merged model path not found: {merged_path} (NOT AVAILABLE)")
        return
    if not base_file:
        print(f"Base model openai/whisper-medium model.safetensors: NOT AVAILABLE in local cache.")
        print(f"Tensors diff vs base: NOT AVAILABLE (Base weights not present locally; downloading disabled per read-only audit protocol).")
        print()
        return

    print(f"Merged path: {merged_path}")
    print(f"Base path:   {base_file}")
    
    merged_weights = load_file(str(merged_path))
    base_weights = load_file(str(base_file))
    
    differing_tensors = 0
    max_abs_diff = 0.0
    equal_tensors = 0
    
    for k in merged_weights.keys():
        if k in base_weights:
            t_m = merged_weights[k].float()
            t_b = base_weights[k].float()
            diff = torch.max(torch.abs(t_m - t_b)).item()
            if diff > 1e-6:
                differing_tensors += 1
                if diff > max_abs_diff:
                    max_abs_diff = diff
            else:
                equal_tensors += 1
                
    print(f"Total evaluated tensors: {differing_tensors + equal_tensors}")
    print(f"Number of tensors differing from base: {differing_tensors}")
    print(f"Number of tensors equal to base:       {equal_tensors}")
    print(f"Maximum absolute difference:           {max_abs_diff:.6f}")
    if differing_tensors == 0:
        print("Verdict: Merged model EQUALS the base model.")
    else:
        print("Verdict: Merged model differs from base with fine-tuned LoRA weights successfully merged.")
    print()

def check_1c_ct2():
    print("=" * 80)
    print("CHECK 1.c: CT2 Directory Verification")
    print("=" * 80)
    ct2_dir = ROOT_DIR / "whisper_medium_output" / "whisper-medium-arabic-qlora-stage2" / "whisper-medium-arabic-ct2"
    if not ct2_dir.exists():
        print(f"CT2 directory not found: {ct2_dir}")
        return
        
    print(f"CT2 Directory: {ct2_dir}")
    files = list(ct2_dir.glob("*"))
    for f in sorted(files):
        print(f"  {f.name:<30} | {f.stat().st_size:>12} bytes | sha256: {compute_sha256(f)}")
        
    required = ["model.bin", "vocabulary.json"]
    for r in required:
        exists = (ct2_dir / r).exists()
        print(f"  Requirement '{r}': {'CONFIRMED' if exists else 'MISSING'}")
        
    # Check parent adapter for tokenizer.json & preprocessor_config.json
    adapter_dir = ROOT_DIR / "whisper_medium_output" / "whisper-medium-arabic-qlora-stage2" / "whisper-medium-arabic-lora-stage2-adapter"
    for r in ["tokenizer.json", "preprocessor_config.json"]:
        exists = (adapter_dir / r).exists()
        print(f"  Tokenizer requirement '{r}' in adapter folder: {'CONFIRMED' if exists else 'MISSING'}")
    print()

def check_1d_inference():
    print("=" * 80)
    print("CHECK 1.d: Model Scoring on Fixed Pairs & Inputs (CPU Checks)")
    print("=" * 80)
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoModelForSeq2SeqLM
    
    # 1. Reranker
    reranker_id = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    print(f"1. Reranker: {reranker_id}")
    r_tok = AutoTokenizer.from_pretrained(reranker_id)
    r_model = AutoModelForSequenceClassification.from_pretrained(reranker_id)
    r_model.eval()
    
    print(f"   Architecture: {r_model.config.architectures}")
    classifier_weights_exist = hasattr(r_model, "classifier")
    print(f"   Classifier weights exist: {classifier_weights_exist}")
    
    # 5 fixed query-passage pairs
    pairs = [
        ("ما هي عاصمة مصر؟", "القاهرة هي عاصمة جمهورية مصر العربية وأكبر مدنها."),
        ("ما هي عاصمة مصر؟", "تعتبر بغداد عاصمة جمهورية العراق."),
        ("متى تأسست جامعة القاهرة؟", "تأسست جامعة القاهرة في 21 ديسمبر 1908."),
        ("ما هي لغة البرازيل؟", "اللغة الرسمية في البرازيل هي البرتغالية."),
        ("أين يقع برج إيفل؟", "يقع برج إيفل في العاصمة الفرنسية باريس بالقرب من نهر السين.")
    ]
    
    print("   Scores on 5 fixed query-document pairs:")
    with torch.no_grad():
        for q, p in pairs:
            inputs = r_tok(q, p, return_tensors="pt", truncation=True, max_length=128)
            score = r_model(**inputs).logits.squeeze().item()
            print(f"     Q: '{q}' | P: '{p[:30]}...' -> Score: {score:.4f}")
    print()

    # 2. CAMeL-BERT embeddings
    camel_id = "CAMeL-Lab/bert-base-arabic-camelbert-msa"
    print(f"2. CAMeL-BERT Bi-Encoder: {camel_id}")
    c_tok = AutoTokenizer.from_pretrained(camel_id)
    c_model = AutoModel.from_pretrained(camel_id)
    c_model.eval()
    
    texts = [
        "جامعة القاهرة تأسست في مصر",
        "تاريخ التعليم العالي في العالم العربي",
        "تقنيات الذكاء الاصطناعي ومعالجة اللغات الطبيعية",
        "الطقس اليوم مشمس وجميل في الإسكندرية",
        "أهمية القراءة وتأثيرها على الثقافة"
    ]
    print("   Embeddings norm & first 3 components on 5 fixed texts:")
    with torch.no_grad():
        for t in texts:
            inputs = c_tok(t, return_tensors="pt", truncation=True, max_length=64)
            emb = c_model(**inputs).last_hidden_state[:, 0, :].squeeze()
            norm = torch.norm(emb).item()
            first3 = emb[:3].tolist()
            first3_str = ", ".join(f"{x:.4f}" for x in first3)
            print(f"     Text: '{t[:30]}...' | Norm: {norm:.4f} | First 3: [{first3_str}]")
    print()

    # 3. AraBART output
    arabart_path = ROOT_DIR / "Summarizer"
    print(f"3. AraBART Local Summarizer: {arabart_path}")
    a_tok = AutoTokenizer.from_pretrained(str(arabart_path))
    a_model = AutoModelForSeq2SeqLM.from_pretrained(str(arabart_path))
    a_model.eval()
    
    sample_articles = [
        "أعلنت وزارة الصحة والسكان المصرية عن تدشين حملة جديدة للتطعيم ضد الأمراض الموسمية في كافة المحافظات، مؤكدة توافر كافة اللقاحات في المراكز الصحية مجاناً لجميع المواطنين.",
        "سجل مؤشر البورصة الرئيسي ارتفاعاً ملحوظاً في ختام تعاملات الأسبوع، مدفوعاً بمشتريات قوية من المؤسسات المحلية والأجنبية في قطاع التكنولوجيا والاتصالات.",
        "افتتح وزير الشباب والرياضة البطولة الإقليمية لألعاب القوى بمشاركة وفود من عشرين دولة عربية، مشيراً إلى جاهزية المنشآت الرياضية لاستضافة البطولات الدولية."
    ]
    print("   Summarization on 3 sample articles:")
    for idx, art in enumerate(sample_articles, 1):
        inputs = a_tok(art, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            out_ids = a_model.generate(**inputs, max_length=64, num_beams=2)
            summary = a_tok.decode(out_ids[0], skip_special_tokens=True)
            print(f"     Article {idx}: '{art[:40]}...'")
            print(f"     Summary   : '{summary}'")
    print()

def check_1e_published_sha256():
    print("=" * 80)
    print("CHECK 1.e: Published Checkpoint sha256 against Evaluated Weights")
    print("=" * 80)
    
    # 1. Whisper-small
    local_w_small = ROOT_DIR / "Whisper-Fine-tuned-final-model" / "model.safetensors"
    if local_w_small.exists():
        sha_small = compute_sha256(local_w_small)
        print(f"Local Whisper-small model.safetensors:")
        print(f"  Path: Whisper-Fine-tuned-final-model/model.safetensors")
        print(f"  Bytes: {local_w_small.stat().st_size}")
        print(f"  SHA256: {sha_small}")
    else:
        print("Local Whisper-small: NOT AVAILABLE")
        
    # Check cache for Omar10lfc/whisper-small-arabic
    cache_w_small = Path(os.path.expanduser("~/.cache/huggingface/hub/models--Omar10lfc--whisper-small-arabic/snapshots"))
    if cache_w_small.exists():
        for snap in cache_w_small.glob("*"):
            f = snap / "model.safetensors"
            if f.exists():
                sha_hub = compute_sha256(f)
                print(f"HuggingFace Hub Omar10lfc/whisper-small-arabic model.safetensors:")
                print(f"  Path: {f}")
                print(f"  SHA256: {sha_hub}")
                print(f"  Match: {'CONFIRMED MATCH' if sha_hub == sha_small else 'DIFFERENT'}")
                
    # 2. AraBART
    local_arabart = ROOT_DIR / "Summarizer" / "model.safetensors"
    if local_arabart.exists():
        sha_arabart = compute_sha256(local_arabart)
        print(f"\nLocal AraBART model.safetensors:")
        print(f"  Path: Summarizer/model.safetensors")
        print(f"  Bytes: {local_arabart.stat().st_size}")
        print(f"  SHA256: {sha_arabart}")
    else:
        print("Local AraBART: NOT AVAILABLE")
        
    cache_arabart = Path(os.path.expanduser("~/.cache/huggingface/hub/models--Omar10lfc--arabart-xlsum-arabic/snapshots"))
    if cache_arabart.exists():
        for snap in cache_arabart.glob("*"):
            f = snap / "model.safetensors"
            if f.exists():
                sha_hub = compute_sha256(f)
                print(f"HuggingFace Hub Omar10lfc/arabart-xlsum-arabic model.safetensors:")
                print(f"  Path: {f}")
                print(f"  SHA256: {sha_hub}")
                print(f"  Match: {'CONFIRMED MATCH' if sha_hub == sha_arabart else 'DIFFERENT'}")
    print()

def check_5b_mde():
    print("=" * 80)
    print("CHECK 5.b: Minimum Detectable P@1 Difference at Observed n and Discordance")
    print("=" * 80)
    # Under exact McNemar test:
    # n = 50 queries
    # b = count where model A is correct and model B is incorrect
    # c = count where model A is incorrect and model B is correct
    # discordance d = b + c
    # Observed discordance in Set (iii):
    # LV3 vs Turbo: b=6, c=3, d=9. Diff = (6 - 3)/50 = 3/50 = 6.0% (p = 0.5078)
    # LV3 vs Medium: b=10, c=3, d=13. Diff = (10 - 3)/50 = 7/50 = 14.0% (p = 0.0923)
    # LV3 vs Small: b=11, c=4, d=15. Diff = (11 - 4)/50 = 7/50 = 14.0% (p = 0.1185)
    
    from scipy.stats import binom
    
    print("Minimum Detectable Difference Calculation (Two-sided alpha = 0.05):")
    for d in [9, 13, 15, 20]:
        # Find minimum b such that 2 * (1 - binom.cdf(b - 1, d, 0.5)) <= 0.05
        min_b = None
        for b in range(d // 2 + 1, d + 1):
            p_val = 2.0 * binom.sf(b - 1, d, 0.5)
            if p_val <= 0.05:
                min_b = b
                break
        if min_b is not None:
            min_diff_count = 2 * min_b - d
            min_diff_pct = (min_diff_count / 50.0) * 100.0
            print(f"  Discordant pairs d={d}: required b >= {min_b} (c <= {d - min_b}) -> |b - c| >= {min_diff_count} queries -> Min Detectable Delta = {min_diff_pct:.1f}%")
        else:
            print(f"  Discordant pairs d={d}: Cannot achieve p <= 0.05 even if b={d} (max power insufficient at d={d})")
    print()

if __name__ == "__main__":
    check_1b_merged_vs_base()
    check_1c_ct2()
    check_1d_inference()
    check_1e_published_sha256()
    check_5b_mde()
