import json
import os
import ast

def build_audit_cv_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.12"
            },
            "accelerator": "GPU"
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.split("\n")]
        })

    def add_code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.split("\n")]
        })

    # 1. Header
    add_md("""# 🎙️ Common Voice Arabic: Empirical WER Gap Isolation
## Isolating Sample Count ($N=400$ vs $100$) vs Sampling Method (Random vs Sequential) vs Engine (PyTorch FP16 vs CT2 INT8)

**Research Objective:**
Strictly isolate the exact contributors explaining the difference between:
- **12.51% WER** (Training Evaluation Protocol: 400 randomly sampled clips, seed 42, PyTorch FP16 Greedy)
- **17.63% WER** (Downstream Evaluation Protocol: first 100 sequential clips, CTranslate2 Static INT8)

**4 Experimental Subsets (PyTorch FP16 Greedy):**
- **Subset A (400 Random, Seed 42):** Direct replication of training evaluation protocol (~12.51%).
- **Subset B (400 Sequential):** Isolates sampling method holding $N=400$ fixed ($\Delta = B - A$).
- **Subset C (100 Random, Seed 42):** Isolates sample count holding random sampling fixed ($\Delta = C - A$).
- **Subset D (100 Sequential):** Direct replica of downstream sample set ($\Delta = E - D$ isolates engine).
- **Condition E (100 Sequential, CT2 Static INT8):** Downstream production value from `whisper_large_downstream_results.json`.""")

    # 2. Dependencies
    add_code("""# ── 1. Install Dependencies ──────────────────────────────────────────────────
!pip install -q transformers datasets jiwer librosa tabulate soundfile peft accelerate
print("[OK] All required packages installed.")""")

    # 3. Environment & Seeding
    add_code("""# ── 2. Environment Setup & Deterministic Seeding ─────────────────────────────
import os, sys, time, re, glob, json, random, gc
import numpy as np
import pandas as pd
import torch
import librosa
import jiwer
from tabulate import tabulate
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_all_seeds(42)

def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[إأآٱ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name     : {torch.cuda.get_device_name(0)}")
print("[OK] Deterministic seed set to 42.")""")

    # 4. Checkpoint Discovery & Dynamic LoRA Merging
    add_code("""# ── 3. Model Discovery & Dynamic LoRA Merging ────────────────────────────────
print("="*80)
print("1. SCANNING FOR WHISPER-LARGE-V3 CHECKPOINTS")
print("="*80)

# Safety bypass for torchao version incompatibility in PEFT during merge
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

BASE_MODEL_NAME = "openai/whisper-large-v3"
MERGED_OUTPUT_DIR = "/kaggle/working/whisper-large-v3-arabic-stage2-final-merged"

model_path = None
merged_candidates = [
    MERGED_OUTPUT_DIR,
    "/kaggle/working/whisper-large-v3-arabic-final-merged",
    "/kaggle/input/whisper-large-v3-arabic-stage2-final-merged",
    "/kaggle/input/datasets/omar10lfc/whisper-large-v3-arabic-stage2-final-merged",
    "whisper_large_output/whisper-large-v3-arabic-stage2-final-merged",
]

# Deep scan /kaggle/input for any pre-merged standalone directory
for root, dirs, files in os.walk("/kaggle/input"):
    if ("model.safetensors" in files or "model.safetensors.index.json" in files or "pytorch_model.bin" in files) and "adapter_config.json" not in files:
        low_root = root.lower()
        if ("large" in low_root or "whisper" in low_root) and ("merged" in low_root or "final" in low_root or "stage2" in low_root or "qlora" in low_root):
            if "turbo" not in low_root and "medium" not in low_root and "small" not in low_root:
                merged_candidates.insert(0, root)

for cand in merged_candidates:
    if os.path.exists(cand):
        files = os.listdir(cand)
        if any(f.endswith(".safetensors") or f.endswith(".bin") or f == "model.safetensors.index.json" for f in files):
            if "config.json" in files:
                model_path = cand
                print(f"  [FOUND] Pre-merged standalone model at: {model_path}")
                break

# If no pre-merged model is found, scan for LoRA adapter and merge dynamically
if not model_path:
    print("  Pre-merged model not found. Scanning for fine-tuned LoRA adapter checkpoints...")
    adapter_candidates = []
    
    search_roots = ["/kaggle/input", "/kaggle/working", "."]
    for sroot in search_roots:
        if not os.path.exists(sroot):
            continue
        for root, dirs, files in os.walk(sroot):
            if "adapter_config.json" in files and any(f.startswith("adapter_model") for f in files):
                low_root = root.lower()
                if "turbo" not in low_root and "medium" not in low_root and "small" not in low_root:
                    adapter_candidates.append(root)

    if not adapter_candidates:
        msg = ("FATAL: Neither a pre-merged Whisper-Large-v3 model nor a LoRA adapter was found in /kaggle/input or /kaggle/working. "
               "Please ensure the 'Fine-Tuning Whisper-Large-v3 on Arabic' notebook output is attached as an input.")
        raise FileNotFoundError(msg)

    # Sort adapter candidates to pick the best/final checkpoint
    def adapter_priority(p):
        low = p.lower()
        score = 0
        if "final" in low or "lora-adapter" in low:
            score += 100000
        step_match = re.search(r'checkpoint-(\d+)', p)
        if step_match:
            score += int(step_match.group(1))
        return score

    adapter_candidates.sort(key=adapter_priority, reverse=True)
    selected_adapter = adapter_candidates[0]
    print(f"  [FOUND] LoRA adapter located at: {selected_adapter}")
    print(f"  Discovered candidates: {adapter_candidates}")

    # Perform FP16 merge
    print("\\n" + "="*80)
    print("2. MERGING LORA ADAPTER INTO BASE WHISPER-LARGE-V3 (FP16)")
    print("="*80)
    t0_merge = time.time()
    os.makedirs(MERGED_OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading unquantized base model ({BASE_MODEL_NAME}) in float16 on CPU...")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    processor = WhisperProcessor.from_pretrained(
        selected_adapter if os.path.exists(os.path.join(selected_adapter, "tokenizer.json")) else BASE_MODEL_NAME
    )

    print(f"Attaching LoRA adapter from {selected_adapter} and fusing weights...")
    peft_model = PeftModel.from_pretrained(base_model, selected_adapter)
    merged_model = peft_model.merge_and_unload()

    # Configure generation parameters
    merged_model.generation_config.language = "arabic"
    merged_model.generation_config.task = "transcribe"
    merged_model.generation_config.forced_decoder_ids = None

    print(f"Saving merged standalone model to: {MERGED_OUTPUT_DIR} ...")
    merged_model.save_pretrained(MERGED_OUTPUT_DIR)
    processor.save_pretrained(MERGED_OUTPUT_DIR)

    del base_model, peft_model, merged_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"[OK] Standalone model successfully merged and saved in {time.time()-t0_merge:.1f}s.")
    model_path = MERGED_OUTPUT_DIR

print(f"\\nActive Model Path for Evaluation: {model_path}")
from transformers import AutoConfig
chk_cfg = AutoConfig.from_pretrained(model_path)
print(f"Model Architecture  : {chk_cfg.model_type}")
print(f"Encoder Layer Count : {getattr(chk_cfg, 'encoder_layers', None)} (Large=32, Medium=24)")
print(f"Decoder Layer Count : {getattr(chk_cfg, 'decoder_layers', None)} (Large=32, Medium=24)")""")

    # 5. Dataset Discovery & Subsets Slicing
    add_code("""# ── 4. Locate Common Voice Arabic Dataset & Slice Subsets ────────────────────
print("="*80)
print("3. LOCATING COMMON VOICE ARABIC TEST SET")
print("="*80)

cv_candidates = [
    '/kaggle/input/datasets/omar10lfc/common-voice-scripted-speech-25-0-arabic',
    '/kaggle/input/common-voice-scripted-speech-25-0-arabic',
    '/kaggle/input/*common*voice*',
    './Data/common_voice_arabic',
]
cv_tsv = None
audio_map = {}
for cand in cv_candidates:
    for croot in glob.glob(cand):
        if os.path.exists(croot):
            for root, _, files in os.walk(croot):
                for f in files:
                    if f == 'test.tsv':
                        cv_tsv = os.path.join(root, f)
                    elif f.endswith(('.mp3', '.wav')):
                        stem = os.path.splitext(f)[0]
                        audio_map[stem] = os.path.join(root, f)
            if cv_tsv:
                break
    if cv_tsv:
        break

if not cv_tsv:
    for root, _, files in os.walk('/kaggle/input'):
        if 'test.tsv' in files and ('common' in root.lower() or 'cv' in root.lower() or 'voice' in root.lower()):
            cv_tsv = os.path.join(root, 'test.tsv')
            for croot, _, cfiles in os.walk(root):
                for cf in cfiles:
                    if cf.endswith(('.mp3', '.wav')):
                        audio_map[os.path.splitext(cf)[0]] = os.path.join(croot, cf)
            if cv_tsv:
                break

if not cv_tsv:
    raise FileNotFoundError("Could not find Common Voice test.tsv on Kaggle filesystem.")

print(f"Found test.tsv: {cv_tsv} ({len(audio_map)} audio files indexed)")

df_raw = pd.read_csv(cv_tsv, sep='\\t', low_memory=False)
path_col = 'path' if 'path' in df_raw.columns else df_raw.columns[1]
text_col = 'sentence' if 'sentence' in df_raw.columns else 'text'

def get_audio(raw):
    return audio_map.get(os.path.splitext(os.path.basename(str(raw)))[0])

df_raw['audio_path'] = df_raw[path_col].apply(get_audio)
df_raw['clean_ref'] = df_raw[text_col].apply(normalize_arabic)
df_raw = df_raw.dropna(subset=['audio_path'])
df_raw = df_raw[df_raw['clean_ref'].str.len() > 2].reset_index(drop=True)
print(f"Total valid test clips available: {len(df_raw)}")

# Construct the 4 subsets:
# Subset A: 400 random clips (seed 42) -> Training eval replica
subset_a_400_rand = df_raw.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:400].copy()

# Subset B: 400 sequential clips -> Isolates sampling method at N=400
subset_b_400_seq = df_raw.iloc[:400].copy()

# Subset C: 100 random clips (seed 42) -> Isolates sample count at Random method
subset_c_100_rand = df_raw.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:100].copy()

# Subset D: 100 sequential clips -> Downstream eval replica
subset_d_100_seq = df_raw.iloc[:100].copy()

print("[OK] Sliced Subsets A (400 Rand), B (400 Seq), C (100 Rand), D (100 Seq).")

client_col = 'client_id' if 'client_id' in df_raw.columns else None
if client_col:
    print(f"\\nSpeaker Diversity (Unique client_ids):")
    print(f"  Subset A (400 Rand, seed 42) : {subset_a_400_rand[client_col].nunique()} unique speakers")
    print(f"  Subset B (400 Seq)           : {subset_b_400_seq[client_col].nunique()} unique speakers")
    print(f"  Subset C (100 Rand, seed 42) : {subset_c_100_rand[client_col].nunique()} unique speakers")
    print(f"  Subset D (100 Seq)           : {subset_d_100_seq[client_col].nunique()} unique speakers")
""")

    # 6. Evaluation with PyTorch FP16 Greedy
    add_code("""# ── 5. Evaluate Subsets with PyTorch FP16 Greedy ─────────────────────────────
print("="*80)
print("4. EVALUATING SUBSETS WITH PYTORCH FP16 GREEDY")
print("="*80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading merged model onto {device}...")
processor = WhisperProcessor.from_pretrained(model_path)
model = WhisperForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device).eval()

def evaluate_subset(df_subset, name="Subset"):
    print(f"\\nEvaluating {name} (N={len(df_subset)} clips)...")
    preds = []
    t0 = time.time()
    for idx, path in enumerate(df_subset['audio_path']):
        audio, _ = librosa.load(path, sr=16000)
        inps = processor(audio, sampling_rate=16000, return_tensors='pt').input_features.to(
            device, dtype=torch.float16 if device == "cuda" else torch.float32
        )
        with torch.no_grad():
            gen_ids = model.generate(inps, language='ar', task='transcribe', num_beams=1, do_sample=False)
        p_txt = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]
        preds.append(normalize_arabic(p_txt))
        if (idx + 1) % 100 == 0 or (idx + 1) == len(df_subset):
            print(f"  Processed {idx+1}/{len(df_subset)} clips...")
    elapsed = time.time() - t0
    wer_val = 100.0 * jiwer.wer(df_subset['clean_ref'].tolist(), preds)
    print(f"  -> {name} WER: {wer_val:.2f}% (Time: {elapsed:.1f}s)")
    return wer_val, elapsed, preds

def compute_speaker_summary(df_subset, preds):
    if 'client_id' not in df_subset.columns: return {}
    spk_data = {}
    for idx, (spk, ref, pr) in enumerate(zip(df_subset['client_id'], df_subset['clean_ref'], preds)):
        if spk not in spk_data: spk_data[spk] = {'refs': [], 'preds': []}
        spk_data[spk]['refs'].append(ref)
        spk_data[spk]['preds'].append(pr)
    spk_wers = [100.0 * jiwer.wer(v['refs'], v['preds']) for v in spk_data.values()]
    return {
        'num_speakers': len(spk_data),
        'mean_speaker_wer': float(np.mean(spk_wers)),
        'std_speaker_wer': float(np.std(spk_wers)),
        'median_speaker_wer': float(np.median(spk_wers)),
    }

# Execute PyTorch FP16 evaluations
wer_a, t_a, preds_a = evaluate_subset(subset_a_400_rand, "Subset A (400 Random, Seed 42 - Training Protocol)")
spk_stats_a = compute_speaker_summary(subset_a_400_rand, preds_a)
if spk_stats_a:
    print(f"  [Subset A Speaker Stats] Mean Speaker WER: {spk_stats_a['mean_speaker_wer']:.2f}%, Median: {spk_stats_a['median_speaker_wer']:.2f}%")

# Sanity-check check for Subset A baseline replication (~12.51%)
expected_baseline = 12.51
tolerance_pp = 2.0
if abs(wer_a - expected_baseline) > tolerance_pp:
    print("\\n" + "!"*80)
    print(f"[WARNING] Baseline Replica Sanity-Check Warning:")
    print(f"  Observed Subset A WER: {wer_a:.2f}% | Expected Baseline: {expected_baseline:.2f}% (Tolerance: +/-{tolerance_pp:.1f} pp)")
    print(f"  Delta: {wer_a - expected_baseline:+.2f} pp exceeds tolerance threshold.")
    print("  The 'replica' assumption may be broken. The attribution breakdown should NOT be trusted without investigation.")
    print("!"*80 + "\\n")
else:
    print(f"[OK] Subset A replicates training baseline: {wer_a:.2f}% (within +/-{tolerance_pp:.1f} pp of {expected_baseline:.2f}%).")

wer_b, t_b, preds_b = evaluate_subset(subset_b_400_seq,  "Subset B (400 Sequential - Scaled Downstream Protocol)")
spk_stats_b = compute_speaker_summary(subset_b_400_seq, preds_b)
if spk_stats_b:
    print(f"  [Subset B Speaker Stats] Mean Speaker WER: {spk_stats_b['mean_speaker_wer']:.2f}%, Median: {spk_stats_b['median_speaker_wer']:.2f}%")

wer_c, t_c, preds_c = evaluate_subset(subset_c_100_rand, "Subset C (100 Random, Seed 42 - Sample-Count Isolation)")
wer_d, t_d, preds_d = evaluate_subset(subset_d_100_seq,  "Subset D (100 Sequential - Downstream Protocol Replica)")

# ── 5b. 5 Additional Independent Random Draws of N=100 (Seeds != 42) ───────────
print("\\n" + "="*80)
print("4B. EVALUATING 5 INDEPENDENT RANDOM DRAWS (N=100) FOR SAMPLING VARIANCE")
print("="*80)

INDEP_SEEDS = [100, 2024, 777, 9999, 12345]
draw_results = []
for s in INDEP_SEEDS:
    df_draw = df_raw.sample(frac=1, random_state=s).reset_index(drop=True).iloc[:100].copy()
    w_draw, _, _ = evaluate_subset(df_draw, f"Random Draw N=100 (Seed {s})")
    draw_results.append({'seed': s, 'wer': w_draw})

draw_wers = [d['wer'] for d in draw_results]
mean_draw_wer = float(np.mean(draw_wers))
std_draw_wer = float(np.std(draw_wers))
print(f"\\n=== 5-Draw N=100 Random Sampling Stability ===")
for d in draw_results:
    print(f"  Seed {d['seed']:5d}: WER = {d['wer']:.2f}% (Delta vs A: {d['wer'] - wer_a:+.2f} pp)")
print(f"Mean WER: {mean_draw_wer:.2f}% +/- {std_draw_wer:.2f}% (Std: {std_draw_wer:.2f} pp)")
print(f"Observed Subset C (Seed 42) delta of {wer_c - wer_a:+.2f} pp vs 5-seed range: [{min(draw_wers) - wer_a:+.2f} pp, {max(draw_wers) - wer_a:+.2f} pp]")""")

    # 7. Attribution Matrix & Export
    add_code("""# ── 6. CT2 Reference Values & Empirical Attribution Matrix ───────────────────
print("="*80)
print("5. COMPUTING ATTRIBUTION MATRIX")
print("="*80)

downstream_res_paths = [
    '/kaggle/working/whisper_large_downstream/results/whisper_large_downstream_results.json',
    'Results/whisper_large_downstream_results.json',
    '../Results/whisper_large_downstream_results.json',
    '/kaggle/input/whisper_large_downstream_results.json',
]
downstream_json_path = None
for p in downstream_res_paths:
    if os.path.exists(p):
        downstream_json_path = p
        break

if not downstream_json_path:
    for root, _, files in os.walk('/kaggle/input'):
        if 'whisper_large_downstream_results.json' in files:
            downstream_json_path = os.path.join(root, 'whisper_large_downstream_results.json')
            break

wer_ct2_int8 = None
wer_ct2_fp16 = None

if downstream_json_path:
    with open(downstream_json_path, 'r', encoding='utf-8') as f:
        ds_data = json.load(f)
    for entry in ds_data.get('asr_quantization', []):
        prec = str(entry.get('Precision', '')).lower()
        model_name = str(entry.get('Model', '')).lower()
        if 'int8' in prec or 'int8' in model_name:
            wer_ct2_int8 = float(entry['WER (%)'])
        elif 'float16' in prec or 'float16' in model_name:
            if 'pytorch' not in model_name:
                wer_ct2_fp16 = float(entry['WER (%)'])

if wer_ct2_int8 is None or wer_ct2_fp16 is None:
    raise FileNotFoundError(
        f"Could not read CT2 reference values from {downstream_res_paths}. "
        "Checked keys in JSON: 'asr_quantization' -> 'Precision' / 'Model' / 'WER (%)'."
    )
print(f"Loaded CT2 references from {downstream_json_path}: CT2 INT8 = {wer_ct2_int8:.2f}%, CT2 FP16 = {wer_ct2_fp16:.2f}%")

# Detailed Attribution Calculation
total_gap = wer_ct2_int8 - wer_a  # Total gap from training eval to downstream CT2 INT8
engine_effect = wer_ct2_int8 - wer_d  # CT2 INT8 vs PyTorch FP16 on identical 100 sequential clips
sampling_method_effect = wer_b - wer_a  # 400 Sequential vs 400 Random holding engine and N fixed
sample_count_effect = wer_c - wer_a     # 100 Random vs 400 Random holding engine and Random method fixed
residual = total_gap - (engine_effect + sampling_method_effect + sample_count_effect)

print("\\n" + "="*90)
print("=== EMPIRICAL WER GAP ISOLATION & ATTRIBUTION MATRIX ===")
print("="*90)

table = [
    {
        'Evaluation Condition': 'A: 400 Random (Training Protocol)',
        'Sample Count': 400,
        'Sampling Method': 'Random (Seed 42)',
        'Engine': 'PyTorch FP16 Greedy',
        'WER (%)': f"{wer_a:.2f}%",
        'Delta vs Baseline': '0.00 pp (Ref)',
    },
    {
        'Evaluation Condition': 'B: 400 Sequential (Method Isolation)',
        'Sample Count': 400,
        'Sampling Method': 'First N Sequential',
        'Engine': 'PyTorch FP16 Greedy',
        'WER (%)': f"{wer_b:.2f}%",
        'Delta vs Baseline': f"{wer_b - wer_a:+.2f} pp",
    },
    {
        'Evaluation Condition': 'C: 100 Random (Count Isolation)',
        'Sample Count': 100,
        'Sampling Method': 'Random (Seed 42)',
        'Engine': 'PyTorch FP16 Greedy',
        'WER (%)': f"{wer_c:.2f}%",
        'Delta vs Baseline': f"{wer_c - wer_a:+.2f} pp",
    },
    {
        'Evaluation Condition': 'D: 100 Sequential (Downstream Replica)',
        'Sample Count': 100,
        'Sampling Method': 'First N Sequential',
        'Engine': 'PyTorch FP16 Greedy',
        'WER (%)': f"{wer_d:.2f}%",
        'Delta vs Baseline': f"{wer_d - wer_a:+.2f} pp",
    },
    {
        'Evaluation Condition': 'E: 100 Sequential (Downstream Production)',
        'Sample Count': 100,
        'Sampling Method': 'First N Sequential',
        'Engine': 'CT2 Static INT8',
        'WER (%)': f"{wer_ct2_int8:.2f}%",
        'Delta vs Baseline': f"{wer_ct2_int8 - wer_a:+.2f} pp",
    },
]

print(tabulate(table, headers='keys', tablefmt='github'))

print("\\n=== Clean Attribution of the Total WER Gap ===")
print(f"Total Observed Gap (E vs A)     : {total_gap:+.2f} pp (100.0%)")
print(f"1. Sampling Method Effect (B - A): {sampling_method_effect:+.2f} pp ({(sampling_method_effect/total_gap)*100:.1f}%)")
print(f"2. Sample Count Effect    (C - A): {sample_count_effect:+.2f} pp ({(sample_count_effect/total_gap)*100:.1f}%)")
print(f"3. Decoding Engine Effect (E - D): {engine_effect:+.2f} pp ({(engine_effect/total_gap)*100:.1f}%)")
print(f"4. Interaction / Residual Effect : {residual:+.2f} pp ({(residual/total_gap)*100:.1f}%)")

results_payload = {
    'subsets': {
        'subset_a_400_rand_wer': wer_a,
        'subset_b_400_seq_wer': wer_b,
        'subset_c_100_rand_wer': wer_c,
        'subset_d_100_seq_wer': wer_d,
        'subset_e_ct2_int8_wer': wer_ct2_int8,
    },
    'attribution_pp': {
        'total_gap': total_gap,
        'sampling_method_effect': sampling_method_effect,
        'sample_count_effect': sample_count_effect,
        'engine_effect': engine_effect,
        'residual': residual,
    },
    'multi_draw_n100': {
        'draws': [{'seed': d['seed'], 'wer': round(d['wer'], 2), 'delta_vs_a': round(d['wer'] - wer_a, 2)} for d in draw_results],
        'mean_wer': round(mean_draw_wer, 2),
        'std_wer': round(std_draw_wer, 2),
        'range_delta_vs_a': [round(min(draw_wers) - wer_a, 2), round(max(draw_wers) - wer_a, 2)],
        'observed_c_delta': round(wer_c - wer_a, 2),
    },
    'speaker_stats': {
        'subset_a_unique_speakers': int(subset_a_400_rand['client_id'].nunique()) if 'client_id' in subset_a_400_rand.columns else None,
        'subset_b_unique_speakers': int(subset_b_400_seq['client_id'].nunique()) if 'client_id' in subset_b_400_seq.columns else None,
        'subset_c_unique_speakers': int(subset_c_100_rand['client_id'].nunique()) if 'client_id' in subset_c_100_rand.columns else None,
        'subset_d_unique_speakers': int(subset_d_100_seq['client_id'].nunique()) if 'client_id' in subset_d_100_seq.columns else None,
    },
    'clip_ids': {
        'subset_a_400_rand': subset_a_400_rand[path_col].tolist(),
        'subset_b_400_seq': subset_b_400_seq[path_col].tolist(),
        'subset_c_100_rand': subset_c_100_rand[path_col].tolist(),
        'subset_d_100_seq': subset_d_100_seq[path_col].tolist(),
    }
}

os.makedirs('Results', exist_ok=True)
out_json = '/kaggle/working/cv_wer_gap_isolation_results.json' if os.path.exists('/kaggle/working') else 'Results/cv_wer_gap_isolation_results.json'
with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(results_payload, f, indent=2)

print(f"\\n[OK] Results exported to {out_json}")""")

    # Validate Python syntax of all code cells
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code_str = "".join(cell["source"])
            py_lines = [l for l in code_str.splitlines() if not l.strip().startswith("!") and not l.strip().startswith("%")]
            py_str = "\n".join(py_lines)
            try:
                ast.parse(py_str)
            except SyntaxError as e:
                print(f"Syntax error in cell {idx}: {e}")
                raise

    return nb

if __name__ == "__main__":
    notebook = build_audit_cv_notebook()
    nb_path = os.path.abspath("Notebooks/audit_cv_sampling_isolation.ipynb")
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"[OK] Generated notebook successfully at: {nb_path}")

    # Also sync copy to root if desired
    root_nb = os.path.abspath("audit_cv_sampling_isolation.ipynb")
    with open(root_nb, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"[OK] Synced copy to: {root_nb}")
