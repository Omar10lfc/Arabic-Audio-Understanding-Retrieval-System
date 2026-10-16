"""
MASTER GENERATOR BUILDER (scratch/generate_clean_builders.py)
Constructs the full scripts/build_all_final_notebooks.py file.
"""

import os
import sys

def main():
    parts = []
    
    # Header & Imports
    parts.append('''import os
import sys
import json
import hashlib
import nbformat as nbf

def create_notebook(nb_path, cells_source):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    }
    for idx, src in enumerate(cells_source):
        clean_src = src.strip()
        cell = nbf.v4.new_code_cell(clean_src)
        cell.id = f"cell_{idx:03d}"
        nb.cells.append(cell)
    
    os.makedirs(os.path.dirname(nb_path), exist_ok=True)
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Generated {nb_path} ({len(cells_source)} code cells)")

# Load evalkit-src manifest if available
evalkit_zip_sha = "4b9e455dce242f6dd21415a14b25f80729c03d883b2ce66d8f50f7c2b25f02e1"
if os.path.exists("evalkit-src.zip"):
    with open("evalkit-src.zip", "rb") as f:
        evalkit_zip_sha = hashlib.sha256(f.read()).hexdigest()

def build_all():
    os.makedirs("Notebooks", exist_ok=True)
''')

    # N0 BUILDER
    parts.append('''    # =========================================================================
    # N0: final_00_merge_models.ipynb (Internet ON)
    # =========================================================================
    n0_cells = [
        # Cell 1: Environment & evalkit verification
        f"""# ── CELL 1: ENVIRONMENT & EVALKIT VERIFICATION ─────────────────────────
import os, sys, subprocess, hashlib, json

subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
for pkg in ["ctranslate2", "faster-whisper", "peft", "transformers", "sentence-transformers"]:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

os.makedirs('Results/env', exist_ok=True)
os.makedirs('Results/final', exist_ok=True)

EXPECTED_EVALKIT_SHA = "{evalkit_zip_sha}"
print(f"Internet: ON | Expected evalkit-src SHA-256: {EXPECTED_EVALKIT_SHA}")

env_path = 'Results/env/final_00_merge_models_env.txt'
with open(env_path, 'w', encoding='utf-8') as f:
    f.write(f"Python Version: {sys.version}\\n")
    try:
        gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv'], text=True)
        f.write(f"GPU Info:\\n{gpu_info}\\n")
    except Exception as e:
        f.write(f"GPU Info: None ({e})\\n")
print(f"Environment logged to: {env_path}")
""",
        # Cell 2: Preflight check for inputs
        """# ── CELL 2: PREFLIGHT INPUTS CHECK ──────────────────────────────────────
import os

print("=== CHECKING /kaggle/input CONTENTS ===")
if os.path.exists("/kaggle/input"):
    for item in sorted(os.listdir("/kaggle/input")):
        item_path = os.path.join("/kaggle/input", item)
        if os.path.isdir(item_path):
            sub = os.listdir(item_path)[:6]
            print(f"  [DIR] {item}/ -> {sub}")
        else:
            print(f"  [FILE] {item}")
else:
    print("Notice: /kaggle/input does not exist (running outside Kaggle?)")
print("\\nPreflight check completed.")
""",
        # Cell 3: Process base weights, adapters, merge FP16 & build CT2 INT8
        """# ── CELL 3: SAVE BASE WEIGHTS, ADAPTERS, MERGE FP16 & BUILD CT2 INT8 ──────
import os, sys, shutil, subprocess, gc, torch, time, hashlib, json
from transformers import (
    WhisperForConditionalGeneration, WhisperProcessor,
    AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, AutoModelForSequenceClassification
)
import peft
from peft import PeftModel
import ctranslate2.converters

# Disable torchao check in PEFT
import peft.import_utils
peft.import_utils.is_torchao_available = lambda: False
try:
    import peft.tuners.lora.torchao as peft_torchao
    peft_torchao.is_torchao_available = lambda: False
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
out_dir = "/kaggle/working/final-merged-models"
os.makedirs(f"{out_dir}/base", exist_ok=True)
os.makedirs(f"{out_dir}/adapters", exist_ok=True)
os.makedirs(f"{out_dir}/merged", exist_ok=True)
os.makedirs(f"{out_dir}/ct2_int8", exist_ok=True)
os.makedirs(f"{out_dir}/downstream", exist_ok=True)

def get_git_commit_hash():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('utf-8').strip()
    except Exception:
        return "unknown"

def hash_directory_files(directory_path):
    hashes = {}
    if not os.path.exists(directory_path):
        return hashes
    for root, _, files in os.walk(directory_path):
        for f in sorted(files):
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, directory_path).replace('\\\\', '/')
            h = hashlib.sha256()
            with open(fp, "rb") as fd:
                while chk := fd.read(65536):
                    h.update(chk)
            hashes[rel] = h.hexdigest()
    return hashes

def find_explicit_adapter(keywords, exclude_keywords=[]):
    candidates = []
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if "adapter_config.json" in files or "adapter_model.safetensors" in files or "adapter_model.bin" in files:
                path_norm = root.lower().replace('\\\\', '/')
                if any(ex.lower() in path_norm for ex in exclude_keywords):
                    continue
                if any(kw.lower() in path_norm for kw in keywords):
                    candidates.append(root)
    if not candidates:
        raise FileNotFoundError(f"FATAL: Explicit adapter matching {keywords} not found in /kaggle/input! No silent fallback permitted.")
    def priority(p):
        p_low = p.lower()
        score = 0
        if "final" in p_low or "lora-adapter" in p_low:
            score += 10000
        if "stage2" in p_low or "stage-2" in p_low:
            score += 9000
        return score
    candidates.sort(key=priority, reverse=True)
    return candidates[0]

models = [
    {
        'key': 'whisper-large-v3',
        'base': 'openai/whisper-large-v3',
        'keywords': ['large-v3', 'large_v3', 'large', 'fine-tuning-whisper'],
        'exclude_keywords': ['turbo', 'medium', 'small'],
        'has_adapter': True
    },
    {
        'key': 'whisper-large-v3-turbo',
        'base': 'openai/whisper-large-v3-turbo',
        'keywords': ['turbo'],
        'exclude_keywords': ['medium', 'small'],
        'has_adapter': True
    },
    {
        'key': 'whisper-medium',
        'base': 'openai/whisper-medium',
        'keywords': ['stage-2', 'stage 2', 'stage2', 'medium'],
        'exclude_keywords': ['large', 'turbo', 'small'],
        'has_adapter': True
    },
    {
        'key': 'whisper-small',
        'base': 'openai/whisper-small',
        'fine_tuned': 'Omar10lfc/whisper-small-arabic',
        'has_adapter': False
    }
]

for m in models:
    k = m['key']
    print(f"\\n========================================")
    print(f"Processing model: {k}")
    print(f"========================================")
    base_save = f"{out_dir}/base/{k}"
    merged_save = f"{out_dir}/merged/{k}"
    ct2_save = f"{out_dir}/ct2_int8/{k}-ct2-int8"
    
    # 1. Base weights
    print(f"Downloading/Saving base weights for {m['base']}...")
    base_mdl = WhisperForConditionalGeneration.from_pretrained(m['base'], torch_dtype=torch.float16, low_cpu_mem_usage=True)
    base_proc = WhisperProcessor.from_pretrained(m['base'])
    base_mdl.save_pretrained(base_save)
    base_proc.save_pretrained(base_save)
    print(f"[OK] Saved base weights to {base_save}")

    adapter_shas = {}
    adapter_src = None
    if m['has_adapter']:
        adapter_src = find_explicit_adapter(m['keywords'], m.get('exclude_keywords', []))
        adapter_save = f"{out_dir}/adapters/{k}-adapter"
        shutil.copytree(adapter_src, adapter_save, dirs_exist_ok=True)
        adapter_shas = hash_directory_files(adapter_save)
        print(f"[OK] Explicit adapter matched: {adapter_src}")
        print(f"[OK] Adapter copied to {adapter_save} ({len(adapter_shas)} files indexed)")
        
        print(f"Merging LoRA weights into unquantized base model ({k})...")
        peft_m = PeftModel.from_pretrained(base_mdl, adapter_save)
        merged = peft_m.merge_and_unload()
        merged.generation_config.language = "arabic"
        merged.generation_config.task = "transcribe"
        merged.save_pretrained(merged_save)
        base_proc.save_pretrained(merged_save)
        print(f"[OK] Saved merged FP16 to {merged_save}")
        del peft_m, merged
    else:
        print(f"Downloading fine-tuned weights for {m['fine_tuned']}...")
        ft_mdl = WhisperForConditionalGeneration.from_pretrained(m['fine_tuned'], torch_dtype=torch.float16, low_cpu_mem_usage=True)
        ft_proc = WhisperProcessor.from_pretrained(m['fine_tuned'])
        ft_mdl.save_pretrained(merged_save)
        ft_proc.save_pretrained(merged_save)
        print(f"[OK] Saved fine-tuned FP16 to {merged_save}")
        del ft_mdl

    del base_mdl
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert to CT2 INT8 with --copy_files tokenizer.json preprocessor_config.json
    print(f"Converting {merged_save} to CT2 INT8 at {ct2_save}...")
    conv_cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", merged_save,
        "--output_dir", ct2_save,
        "--quantization", "int8",
        "--force",
        "--copy_files", "tokenizer.json", "preprocessor_config.json"
    ]
    subprocess.run(conv_cmd, check=True)
    base_proc.save_pretrained(ct2_save)
    print(f"[OK] Successfully built CT2 INT8 at {ct2_save}")

    # Write model_card.json for merged and CT2 models
    for sp, prec in [(merged_save, "PyTorch FP16"), (ct2_save, "CTranslate2 INT8")]:
        card = {
            "model_key": k,
            "base_model": m['base'],
            "adapter_source": adapter_src if m['has_adapter'] else m.get('fine_tuned'),
            "adapter_sha256": adapter_shas if m['has_adapter'] else {},
            "commit_sha": get_git_commit_hash(),
            "precision": prec,
            "export_timestamp": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
        }
        with open(os.path.join(sp, "model_card.json"), "w", encoding="utf-8") as f:
            json.dump(card, f, indent=2)
        print(f"[OK] Wrote model_card.json to {sp}")
    
    del base_proc
    gc.collect()

# 2. Package Downstream Models for Offline Execution
print("\\n========================================")
print("Packaging Downstream Models for Offline Execution")
print("========================================")
downstream_models = [
    ('Omar10lfc/arabart-xlsum-arabic', f"{out_dir}/downstream/arabart-xlsum-arabic", AutoModelForSeq2SeqLM, AutoTokenizer),
    ('CAMeL-Lab/bert-base-arabic-camelbert-msa', f"{out_dir}/downstream/camelbert-msa", AutoModel, AutoTokenizer),
    ('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1', f"{out_dir}/downstream/mmarco-reranker", AutoModelForSequenceClassification, AutoTokenizer)
]
for src_id, dst_path, mdl_cls, tok_cls in downstream_models:
    print(f"Downloading & saving downstream model: {src_id}...")
    m_obj = mdl_cls.from_pretrained(src_id)
    t_obj = tok_cls.from_pretrained(src_id)
    m_obj.save_pretrained(dst_path)
    t_obj.save_pretrained(dst_path)
    
    if mdl_cls == AutoModelForSequenceClassification:
        has_clf = any('classifier' in k for k in m_obj.state_dict().keys())
        assert has_clf, f"FATAL: Sequence classification weights missing in saved reranker at {dst_path}!"
        print(f"[OK] Verified classifier head weights present in {dst_path}")
        
    print(f"[OK] Saved downstream model {src_id} -> {dst_path}")
    del m_obj, t_obj
    gc.collect()

print("\\n[ALL MODELS READY] Base, Adapters, Merged FP16, CT2 INT8, and Downstream models saved successfully with model cards.")
""",
        # Cell 4: SHA256 Manifest generation
        """# ── CELL 4: SHA256 MANIFEST OF MERGED WEIGHTS & ADAPTERS ────────────────
import hashlib, os, json

manifest = {}
out_dir = "/kaggle/working/final-merged-models"
for root, _, files in os.walk(out_dir):
    for f in sorted(files):
        fp = os.path.join(root, f)
        rel = os.path.relpath(fp, out_dir).replace('\\\\', '/')
        with open(fp, 'rb') as fd:
            manifest[rel] = hashlib.sha256(fd.read()).hexdigest()

manifest_file = os.path.join(out_dir, 'sha256_manifest.json')
with open(manifest_file, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2)

print(f"[OK] Manifest generated: {manifest_file} ({len(manifest)} files indexed)")
"""
    ]
    create_notebook("Notebooks/final_00_merge_models.ipynb", n0_cells)
''')

    # Save to file
    with open("scratch/patch_clean_n0_builder.py", "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print("N0 builder generated.")

if __name__ == "__main__":
    main()
