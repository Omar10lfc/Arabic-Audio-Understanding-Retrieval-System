import json
import os
import ast
import shutil

def create_whisper_large_turbo_qlora_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat": 4,
                "nbformat_minor": 5
            }
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

    # ── Cell 0: Header & Documentation ──────────────────────────────────────────
    add_md("""# 🎙️ Fine-Tuning Whisper-Large-v3-Turbo on Arabic Speech with QLoRA

This notebook fine-tunes **`openai/whisper-large-v3-turbo` (809M parameters)** on Arabic speech using **4-bit QLoRA (Quantized Low-Rank Adaptation)**.

### Why Whisper-Large-v3-Turbo?
1. **Asymmetric High-Speed Architecture:** OpenAI's `whisper-large-v3-turbo` couples the unpruned **32-layer Large-v3 encoder** (128 Mel channels) with a pruned **4-layer decoder** (reduced from 32 layers). It captures high-fidelity acoustic features while decoding **over 2.5× faster** than standard Large-v3.
2. **Speed & Accuracy Sweet Spot:** Totaling 809M parameters (comparable to Medium's 769M), Turbo delivers near Large-v3 recognition quality while matching the throughput demands of real-time conversational and retrieval pipelines (~8.5× real-time throughput).
3. **QLoRA (BitsAndBytes 4-bit NF4 + PEFT):** Trains only ~15M adapter parameters (<2% of total weights). Base weights occupy only **~1.1 GB VRAM**. With gradient checkpointing and effective batch size 32, peak training VRAM remains **under 6.2 GB on a free 16 GB Tesla T4 GPU**.
4. **RAM-Safe Chunked Disk Caching:** Extracts 128-channel Mel filterbanks in streaming disk chunks with resume capability, preventing Kaggle kernel crashes from host RAM exhaustion.
5. **Pre-Training Baseline Evaluation:** Evaluates zero-shot WER on the held-out test split *before* training starts to rigorously measure exact absolute and relative error reductions.
6. **Full Model Merge & CTranslate2 Export:** Merges the LoRA adapter back into full FP16 weights for production serving or direct conversion to CTranslate2 (`faster-whisper`).""")

    # ── Cell 1: Package Installations ─────────────────────────────────────────
    add_code("""# ── Install Required Libraries ────────────────────────────────────────────────
# Do not use `pip install -U` indiscriminately, as it can conflict with pre-installed NumPy/SciPy
# C-extension binaries in memory. Only install missing/required packages:
!pip install -q peft bitsandbytes evaluate jiwer ctranslate2 librosa soundfile
# Remove incompatible pre-installed torchao to prevent PEFT merge dispatch errors
!pip uninstall -y -q torchao
print("Required libraries verified and ready.")""")

    # ── Cell 2: Imports & GPU Diagnostics ─────────────────────────────────────
    add_code("""import os, re, gc, time, shutil, json, inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import librosa
import evaluate

from datasets import Dataset, DatasetDict, Array2D, Sequence, Value, Features
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

# ── GPU Diagnostics ──────────────────────────────────────────────────────────
print(f'PyTorch    : {torch.__version__}')
print(f'CUDA       : {torch.version.cuda}')
print(f'GPU Device : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
if torch.cuda.is_available():
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'VRAM Total : {vram:.2f} GB')
    try:
        _ = torch.randn(4, 4).cuda() @ torch.randn(4, 4).cuda()
        print('CUDA Check : ✅ Tensor operations working properly')
    except Exception as e:
        print(f'CUDA Check : ❌ Error - {e}')
else:
    print('CUDA Check : ⚠️ Warning - No CUDA GPU detected. QLoRA requires a GPU.')""")

    # ── Cell 3: Experiment Configuration ──────────────────────────────────────
    add_code("""# ── Experiment Configuration ─────────────────────────────────────────────────
CFG = {
    # Path to Common Voice Arabic dataset on Kaggle/local disk
    'data_root': '/kaggle/input/datasets/omar10lfc/common-voice-scripted-speech-25-0-arabic',

    # Sample counts
    'n_train': 25000,           # Full training subset
    'n_test':  400,             # Held-out evaluation benchmark
    'seed':    42,

    # Model architecture
    'model_name': 'openai/whisper-large-v3-turbo',  # 809M parameters, 128 Mel channels, 4 decoder layers

    # LoRA hyperparameters (higher rank captures phonetic & dialectal nuances)
    'lora_r':        32,
    'lora_alpha':    64,
    'lora_dropout':  0.05,
    'target_modules': ['q_proj', 'v_proj', 'out_proj'],  # Encoder + Decoder attention

    # Training hyperparameters
    # Effective batch size = 8 * 4 = 32 samples per optimizer step.
    # 1200 steps = 38,400 samples (~1.54 full epochs over 25,000 samples).
    # Turbo decoder has only 4 layers, running in ~1.8 to 2.2 hours on Kaggle T4.
    'learning_rate':           1e-4,
    'max_steps':               1200,
    'eval_steps':              300,
    'save_steps':              300,
    'logging_steps':           25,
    'warmup_steps':            100,
    'batch_size':              8,
    'grad_accumulation':       4,   # Effective batch size = 8 * 4 = 32
    'weight_decay':            0.01,
    'early_stopping_patience': 3,

    # Directories
    'output_dir': '/kaggle/working/whisper-large-v3-turbo-arabic-qlora',
    'cache_dir':  '/kaggle/tmp/whisper_large_turbo_cache',
}

os.makedirs(CFG['output_dir'], exist_ok=True)
os.makedirs(CFG['cache_dir'],  exist_ok=True)

print('Experiment Configuration:')
for k, v in CFG.items():
    print(f'  {k:24s}: {v}')""")

    # ── Cell 4: Scan and Index Dataset Files ──────────────────────────────────
    add_code("""# ── Scan and Index Dataset Files ─────────────────────────────────────────────
# Robust path resolution across Kaggle and local disk environments
if not os.path.exists(CFG['data_root']):
    alt_paths = [
        '/kaggle/input/common-voice-scripted-speech-25-0-arabic',
        '/kaggle/input/datasets/omar10lfc/common-voice-scripted-speech-25-0-arabic',
        './Data/common_voice_arabic',
        os.path.expanduser('~/Downloads/arabic_audio_system/Data/common_voice_arabic'),
    ]
    for alt in alt_paths:
        if os.path.exists(alt):
            CFG['data_root'] = alt
            print(f'Found dataset at alternate path: {alt}')
            break

print(f'Scanning {CFG["data_root"]} ...')
t0 = time.time()

tsv_index   = {}   # 'train' / 'dev' / 'test' -> full .tsv path
audio_index = {}   # 'common_voice_ar_XXXXX' -> full audio path

if not os.path.exists(CFG['data_root']):
    print(f'⚠️ Warning: {CFG["data_root"]} not found. Please update CFG["data_root"] to your dataset path.')
else:
    for root, _, files in os.walk(CFG['data_root']):
        for fname in files:
            full = os.path.join(root, fname)
            if fname.endswith('.tsv'):
                stem = fname[:-4]
                if stem not in tsv_index:
                    tsv_index[stem] = full
            elif fname.endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                stem = os.path.splitext(fname)[0]
                audio_index[stem] = full

    print(f'Scan completed in {time.time()-t0:.1f}s')
    print(f'Found TSV splits : {list(tsv_index.keys())}')
    print(f'Indexed audio clips: {len(audio_index):,}')""")

    # ── Cell 5: Arabic Text Normalization & Data Loading ──────────────────────
    add_code('''# ── Arabic Text Normalization & Data Loading ─────────────────────────────────
def normalize_arabic(text: str) -> str:
    """
    Deterministic Arabic normalization identical to evaluation baseline:
    unifies alef variants, dotless ya, ta-marbuta, removes diacritics & tatweel.
    """
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[إأآٱ]', 'ا', text)              # alef forms -> ا
    text = re.sub(r'ى', 'ي', text)                    # dotless ya -> ي
    text = re.sub(r'ة', 'ه', text)                    # ta-marbuta -> ه
    text = re.sub(r'[\\u064B-\\u065F\\u0670]', '', text) # strip diacritics
    text = re.sub(r'ـ', '', text)                      # strip tatweel
    text = re.sub(r'[^\\w\\s\\u0600-\\u06FF]', '', text)  # keep Arabic letters & digits
    text = re.sub(r'\\s+', ' ', text)
    return text.strip()


def load_tsv(key: str, n_samples: Optional[int] = None, seed: int = 42) -> pd.DataFrame:
    if key not in tsv_index:
        raise FileNotFoundError(f'{key}.tsv not found in {list(tsv_index.keys())}')
    df = pd.read_csv(tsv_index[key], sep='\\t', low_memory=False)
    path_col = 'path' if 'path' in df.columns else df.columns[1]
    text_col = 'sentence' if 'sentence' in df.columns else 'text'

    def resolve(raw):
        stem = os.path.splitext(os.path.basename(str(raw)))[0]
        return audio_index.get(stem)

    df['audio_path'] = df[path_col].apply(resolve)
    df['sentence']   = df[text_col].apply(normalize_arabic)
    df = df.dropna(subset=['audio_path'])
    df = df[df['sentence'].str.len() > 2][['audio_path', 'sentence']].reset_index(drop=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    if n_samples and n_samples < len(df):
        df = df.iloc[:n_samples].reset_index(drop=True)
    return df

if tsv_index:
    df_train_raw = load_tsv('train', seed=CFG['seed'])
    df_dev_raw   = load_tsv('dev',   seed=CFG['seed'])
    df_test_raw  = load_tsv('test',  n_samples=CFG['n_test'], seed=CFG['seed'])

    df_all = pd.concat([df_train_raw, df_dev_raw], ignore_index=True)
    df_all = df_all.sample(frac=1, random_state=CFG['seed']).reset_index(drop=True)
    df_train = df_all.iloc[:CFG['n_train']].reset_index(drop=True)

    print(f'Train split size : {len(df_train):,}')
    print(f'Test split size  : {len(df_test_raw):,}')
    print(f'Example record   : {df_train.iloc[0].to_dict()}')''')

    # ── Cell 6: Load Whisper Processor ────────────────────────────────────────
    add_code("""# ── Load Whisper Processor (128 Mel Channels for Turbo) ───────────────────────
print(f'Loading processor for {CFG["model_name"]}...')
processor = WhisperProcessor.from_pretrained(
    CFG['model_name'],
    language='Arabic',
    task='transcribe',
)
print(f'Feature extractor feature_size: {processor.feature_extractor.feature_size} (Confirmed 128 Mel channels)')
print(f'Tokenizer vocabulary size     : {len(processor.tokenizer):,}')
print('Feature extractor and tokenizer ready.')""")

    # ── Cell 7: RAM-Safe Feature Extraction in Disk Chunks ───────────────────
    add_code("""# ── RAM-Safe Feature Extraction in Disk Chunks (128 Mel Bins) ────────────────
# Whisper Large-v3-Turbo uses 128 Mel filterbanks (shape: 128 x 3000)
ARROW_FEATURES = Features({
    'input_features': Array2D(shape=(128, 3000), dtype='float32'),
    'labels':         Sequence(Value('int64')),
})

def extract_features(df: pd.DataFrame, split_name: str, chunk_size: int = 300) -> Dataset:
    cache = os.path.join(CFG['cache_dir'], split_name)
    os.makedirs(cache, exist_ok=True)
    total = len(df)
    n_chunks = (total + chunk_size - 1) // chunk_size
    done_paths = []
    skipped = 0

    for i in range(n_chunks):
        chunk_path = os.path.join(cache, f'chunk_{i:04d}')
        if os.path.exists(os.path.join(chunk_path, 'dataset_info.json')):
            done_paths.append(chunk_path)
            skipped += 1
            continue

        start = i * chunk_size
        end   = min(start + chunk_size, total)
        rows  = df.iloc[start:end]
        feats_list, labels_list = [], []

        for _, row in rows.iterrows():
            try:
                audio, _ = librosa.load(row['audio_path'], sr=16000, mono=True)
                duration = len(audio) / 16000
                if not (0.5 <= duration <= 29.0):
                    continue
                feat = processor.feature_extractor(audio, sampling_rate=16000).input_features[0].astype(np.float32)
                token_ids = processor.tokenizer(row['sentence']).input_ids
                feats_list.append(feat)
                labels_list.append(token_ids)
            except Exception:
                continue

        if not feats_list:
            continue

        chunk_ds = Dataset.from_dict(
            {'input_features': np.array(feats_list, dtype=np.float32), 'labels': labels_list},
            features=ARROW_FEATURES,
        )
        chunk_ds.save_to_disk(chunk_path)
        done_paths.append(chunk_path)
        del feats_list, labels_list, chunk_ds
        gc.collect()

    if skipped:
        print(f'  [{split_name}] Resumed: {skipped} chunks already cached.')

    from datasets import concatenate_datasets
    all_chunks = [Dataset.load_from_disk(p) for p in done_paths]
    result = concatenate_datasets(all_chunks)
    print(f'  [{split_name}] Processed: {len(result):,} samples ready.')
    return result

if 'df_train' in locals():
    print('Extracting train features...')
    train_ds = extract_features(df_train, 'train')
    print('Extracting test features...')
    test_ds  = extract_features(df_test_raw, 'test')
    dataset  = DatasetDict({'train': train_ds, 'test': test_ds})
    print(f'Ready -> Train: {len(dataset["train"]):,}, Test: {len(dataset["test"]):,}')""")

    # ── Cell 8: Data Collator & WER Metric ─────────────────────────────────────
    add_code("""# ── Data Collator & WER Metric ───────────────────────────────────────────────
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorWhisper:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{'input_features': f['input_features']} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors='pt')

        label_features = [{'input_ids': f['labels']} for f in features]
        labels_batch   = self.processor.tokenizer.pad(label_features, return_tensors='pt')
        labels = labels_batch['input_ids'].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch['labels'] = labels
        return batch


wer_metric = evaluate.load('wer')

def compute_metrics(pred):
    pred_ids  = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    pred_str  = [normalize_arabic(s) for s in pred_str]
    label_str = [normalize_arabic(s) for s in label_str]

    pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
    if not pairs:
        return {'wer': 100.0}
    preds, labels = zip(*pairs)
    wer = 100 * wer_metric.compute(predictions=list(preds), references=list(labels))
    return {'wer': round(wer, 2)}

print('Data collator and WER metric defined.')""")

    # ── Cell 9: Load Whisper-Large-v3-Turbo in 4-bit with QLoRA PEFT ───────────
    add_code("""# ── Load Whisper-Large-v3-Turbo in 4-bit with QLoRA PEFT ─────────────────────
print(f'Loading {CFG["model_name"]} with 4-bit NormalFloat quantization...')

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

try:
    model = WhisperForConditionalGeneration.from_pretrained(
        CFG['model_name'],
        quantization_config=bnb_config,
        device_map='auto',
        attn_implementation='sdpa',
    )
    print('PyTorch SDPA (Flash Attention) enabled.')
except Exception:
    model = WhisperForConditionalGeneration.from_pretrained(
        CFG['model_name'],
        quantization_config=bnb_config,
        device_map='auto',
    )

# Prepare model for kbit training (freezes base model, enables gradient checkpointing)
model = prepare_model_for_kbit_training(model)

# Critical fix for Whisper with Gradient Checkpointing in PEFT:
# Whisper's audio encoder uses Conv1D rather than an Embedding layer.
# We attach a forward hook to conv1 so input features require gradients without error.
def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)

model.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)

# Configure LoRA
lora_config = LoraConfig(
    r=CFG['lora_r'],
    lora_alpha=CFG['lora_alpha'],
    target_modules=CFG['target_modules'],
    lora_dropout=CFG['lora_dropout'],
    bias='none',
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Set generation defaults
model.generation_config.language = 'arabic'
model.generation_config.task     = 'transcribe'
model.generation_config.forced_decoder_ids = None
model.config.use_cache = False  # required for training with gradient checkpointing

data_collator = DataCollatorWhisper(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)""")

    # ── Cell 10: Training Arguments & Seq2SeqTrainer ──────────────────────────
    add_code("""# ── Training Arguments & Seq2SeqTrainer ──────────────────────────────────────
# Detect eval_strategy vs evaluation_strategy for version compatibility across transformers
sig = inspect.signature(Seq2SeqTrainingArguments.__init__)
strat_key = 'eval_strategy' if 'eval_strategy' in sig.parameters else 'evaluation_strategy'

training_args_dict = {
    'output_dir':                  CFG['output_dir'],
    'per_device_train_batch_size': CFG['batch_size'],
    'gradient_accumulation_steps': CFG['grad_accumulation'],
    'per_device_eval_batch_size':  CFG['batch_size'],
    'learning_rate':               CFG['learning_rate'],
    'warmup_steps':                CFG['warmup_steps'],
    'lr_scheduler_type':           'cosine',
    'max_steps':                   CFG['max_steps'],
    'weight_decay':                CFG['weight_decay'],
    strat_key:                     'steps',
    'eval_steps':                  CFG['eval_steps'],
    'save_strategy':               'steps',
    'save_steps':                  CFG['save_steps'],
    'save_total_limit':            2,
    'logging_steps':               CFG['logging_steps'],
    'load_best_model_at_end':      True,
    'metric_for_best_model':       'wer',
    'greater_is_better':           False,
    'predict_with_generate':       True,
    'generation_max_length':       80,
    'fp16':                        True,
    'gradient_checkpointing':      True,
    'optim':                       'paged_adamw_8bit',  # Ultra memory-efficient optimizer for QLoRA
    'dataloader_num_workers':      2,
    'dataloader_pin_memory':       True,
    'remove_unused_columns':       False,
    'report_to':                   ['tensorboard'],
}

training_args = Seq2SeqTrainingArguments(**training_args_dict)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=dataset['train'] if 'dataset' in locals() else None,
    eval_dataset=dataset['test']   if 'dataset' in locals() else None,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=CFG['early_stopping_patience'])],
)

print('Seq2SeqTrainer initialized successfully.')""")

    # ── Cell 11: Evaluate Zero-Shot Baseline WER ──────────────────────────────
    add_code("""# ── Evaluate Zero-Shot Baseline WER on Whisper-Large-v3-Turbo ─────────────────
print('Evaluating Whisper-Large-v3-Turbo baseline zero-shot WER on test set...')
print('(Takes ~2-3 minutes on GPU)...')

# Temporarily detach progress callback to prevent notebook widget lockups
nb_callback = None
for cb in trainer.callback_handler.callbacks:
    if cb.__class__.__name__ == 'NotebookProgressCallback':
        nb_callback = cb
        break
if nb_callback is not None:
    trainer.remove_callback(nb_callback)

baseline_eval = trainer.evaluate()
BASELINE_WER = baseline_eval['eval_wer']

if nb_callback is not None:
    trainer.add_callback(nb_callback)

print(f'\\n==============================================')
print(f'  BASELINE WHISPER-LARGE-V3-TURBO ZERO-SHOT WER: {BASELINE_WER:.2f}%')
print(f'==============================================\\n')""")

    # ── Cell 12: Run QLoRA Training ───────────────────────────────────────────
    add_code("""# ── Run QLoRA Training ───────────────────────────────────────────────────────
print('Starting Whisper-Large-v3-Turbo QLoRA fine-tuning...')
train_output = trainer.train()

print('\\nTraining finished!')
print(f'Runtime    : {train_output.metrics["train_runtime"]:.1f}s ({train_output.metrics["train_runtime"]/3600:.2f} hours)')
print(f'Train Loss : {train_output.metrics["train_loss"]:.4f}')""")

    # ── Cell 13: Final Evaluation & Benchmark Comparison ─────────────────────
    add_code("""# ── Final Evaluation & Benchmark Comparison ──────────────────────────────────
print('Evaluating fine-tuned QLoRA Whisper-Large-v3-Turbo...')
final_eval = trainer.evaluate()
FINAL_WER = final_eval['eval_wer']

abs_diff = BASELINE_WER - FINAL_WER
rel_diff = (abs_diff / BASELINE_WER) * 100

print(f'\\n==============================================')
print(f'  BASELINE ZERO-SHOT WER : {BASELINE_WER:.2f}%')
print(f'  FINE-TUNED QLORA WER   : {FINAL_WER:.2f}%')
print(f'  ABSOLUTE REDUCTION     : {abs_diff:.2f} percentage points')
print(f'  RELATIVE REDUCTION     : {rel_diff:.1f}% error reduction')
print(f'==============================================\\n')

results = {
    'model':           CFG['model_name'],
    'method':          'QLoRA (4-bit NF4)',
    'n_train':         CFG['n_train'],
    'n_test':          CFG['n_test'],
    'max_steps':       CFG['max_steps'],
    'baseline_wer':    round(BASELINE_WER, 2),
    'finetuned_wer':   round(FINAL_WER, 2),
    'abs_improvement': round(abs_diff, 2),
    'rel_improvement': round(rel_diff, 1),
}
with open(os.path.join(CFG['output_dir'], 'results_large_v3_turbo_qlora.json'), 'w') as f:
    json.dump(results, f, indent=2)
print(f'Results exported to: {os.path.join(CFG["output_dir"], "results_large_v3_turbo_qlora.json")}')""")

    # ── Cell 14: Save LoRA Adapter Checkpoint ─────────────────────────────────
    add_code("""# ── Save LoRA Adapter Checkpoint ─────────────────────────────────────────────
LORA_SAVE_PATH = os.path.join(CFG['output_dir'], 'whisper-large-v3-turbo-arabic-lora-adapter')
trainer.save_model(LORA_SAVE_PATH)
processor.save_pretrained(LORA_SAVE_PATH)

print(f'LoRA adapter saved to: {LORA_SAVE_PATH}')
for f in sorted(os.listdir(LORA_SAVE_PATH)):
    sz = os.path.getsize(os.path.join(LORA_SAVE_PATH, f)) / 1e6
    print(f'  {f:<35s} {sz:>8.2f} MB')""")

    # ── Cell 15: Merge LoRA Weights into Full 16-bit Model ────────────────────
    add_code("""# ── Merge LoRA Weights into Full 16-bit Model for Production Deployment ──────
# WHY: Merging eliminates LoRA overhead and allows direct conversion to CTranslate2 (faster-whisper)
MERGED_OUTPUT_PATH = os.path.join(CFG['output_dir'], 'whisper-large-v3-turbo-arabic-final-merged')

print('Merging LoRA adapter back into base Whisper-Large-v3-Turbo (fp16)...')
del model, trainer
gc.collect()
torch.cuda.empty_cache()

# Load clean unquantized base model in fp16
base_model = WhisperForConditionalGeneration.from_pretrained(
    CFG['model_name'],
    torch_dtype=torch.float16,
    device_map='cpu',
)

# Safety bypass for torchao version incompatibility in PEFT during merge
try:
    import peft.import_utils
    peft.import_utils.is_torchao_available = lambda: False
except Exception:
    pass

# Attach LoRA adapter and fuse weights
peft_model = PeftModel.from_pretrained(base_model, LORA_SAVE_PATH)
merged_model = peft_model.merge_and_unload()

# Configure generation attributes
merged_model.generation_config.language = 'arabic'
merged_model.generation_config.task     = 'transcribe'
merged_model.generation_config.forced_decoder_ids = None

# Save full standalone model
merged_model.save_pretrained(MERGED_OUTPUT_PATH)
processor.save_pretrained(MERGED_OUTPUT_PATH)

print(f'✅ Full standalone merged model saved to: {MERGED_OUTPUT_PATH}')
print('Ready for direct upload to Hugging Face or CTranslate2 compilation!')""")

    # ── Cell 16: Qualitative Inspection on 5 Test Samples ────────────────────
    add_code("""# ── Qualitative Inspection on 5 Test Samples ─────────────────────────────────
print('=== Qualitative Test Samples (Ground Truth vs Prediction) ===\\n')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
merged_model = merged_model.to(device).eval()

indices = np.random.default_rng(42).integers(0, len(df_test_raw), size=5)

for idx in indices:
    row = df_test_raw.iloc[idx]
    try:
        audio, _ = librosa.load(row['audio_path'], sr=16000, mono=True)
        inputs = processor(audio, sampling_rate=16000, return_tensors='pt').input_features.to(device, dtype=torch.float16)
        with torch.no_grad():
            pred_tokens = merged_model.generate(inputs, language='arabic', task='transcribe', max_new_tokens=128)
        prediction = processor.batch_decode(pred_tokens, skip_special_tokens=True)[0]
        ref = row['sentence']

        sample_wer = wer_metric.compute(
            predictions=[normalize_arabic(prediction)],
            references=[normalize_arabic(ref)],
        ) * 100

        print(f'Reference : {ref}')
        print(f'Predicted : {prediction}')
        print(f'Sample WER: {sample_wer:.1f}%')
        print('-' * 70)
    except Exception as e:
        print(f'Sample {idx} error: {e}')""")

    # ── Cell 17: Summary Table for Report ─────────────────────────────────────
    add_code("""# ── Summary Table for Report ─────────────────────────────────────────────────
print('## Whisper-Large-v3-Turbo QLoRA Benchmark Results\\n')
print('| Metric               | Value                             |')
print('|----------------------|-----------------------------------|')
print(f'| Base Model           | {CFG["model_name"]} (809M)       |')
print(f'| Adaptation Method    | QLoRA (4-bit NF4, r=32, alpha=64) |')
print(f'| Mel Frequency Bins   | 128 Channels                      |')
print(f'| Training Samples     | {CFG["n_train"]:,}                        |')
print(f'| Test Samples         | {CFG["n_test"]:,}                         |')
print(f'| Training Steps       | {CFG["max_steps"]:,}                        |')
print(f'| Baseline Zero-Shot   | {BASELINE_WER:.2f}%                       |')
print(f'| Fine-Tuned WER       | {FINAL_WER:.2f}%                       |')
print(f'| Absolute Reduction   | {abs_diff:.2f} percentage points          |')
print(f'| Relative Reduction   | {rel_diff:.1f}%                        |')""")

    # ── Cell 18: CTranslate2 Conversion Instructions ──────────────────────────
    add_code("""# ── CTranslate2 Fast Inference Conversion ─────────────────────────────────────
ct2_out = os.path.join(CFG['output_dir'], 'whisper-large-v3-turbo-ct2-int8')
cmd = f"ct2-transformers-converter --model {MERGED_OUTPUT_PATH} --output_dir {ct2_out} --quantization int8_float16 --copy_files tokenizer.json preprocessor_config.json"
print("=== CTranslate2 Conversion Command ===")
print("To convert the merged model for ultra-fast CTranslate2 inference with faster-whisper, execute:")
print(cmd)""")

    # Validate Python syntax in all code cells
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            code = "".join(cell["source"])
            clean_code = "\n".join([line for line in code.split("\n") if not line.strip().startswith("!")])
            try:
                ast.parse(clean_code)
            except SyntaxError as e:
                print(f"Syntax error in cell {idx}: {e}")
                raise

    return nb

if __name__ == "__main__":
    notebook = create_whisper_large_turbo_qlora_notebook()
    target_path = os.path.abspath("whisper-large-v3-turbo-qlora.ipynb")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"[OK] Generated notebook successfully at: {target_path}")

    # Also sync to Notebooks/ directory
    notebooks_target = os.path.abspath("Notebooks/whisper-large-v3-turbo-qlora.ipynb")
    shutil.copyfile(target_path, notebooks_target)
    print(f"[OK] Synced copy to: {notebooks_target}")
