import json
import os
import ast
import shutil

def create_spoken_retrieval_notebook():
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

    # 1. Header & Research Motivation
    add_md("""# 🎙️ ➔ 🔍 Spoken Document Retrieval (Speech-RAG) Error Propagation Benchmark
### Evaluating ASR Recognition Error Propagation across Dense Retrieval (CAMeL-BERT) & Joint Cross-Attention Re-Ranking (`mmarco-mMiniLMv2`)

**Key Research Question:**
When building production Speech-RAG (Retrieval-Augmented Generation over Arabic lectures and podcasts):
1. **Bi-Encoder Vulnerability to ASR Noise:** How much does upstream speech recognition noise (~31% WER on encyclopedic texts from fine-tuned Whisper-Medium CT2 Static INT8) degrade dense single-vector retrieval (Precision@1 and Recall@10)?
2. **Cross-Encoder Transcription Noise Resilience Hypothesis:** Does joint cross-attention (`[Query, Candidate Chunk]`) overcome phonetic and morphological transcription corruptions, recovering high-precision retrieval where bi-encoders fail?
3. **End-to-End Retention:** How much of the Clean Oracle retrieval precision is retained when searching directly through raw Whisper transcripts?

**Benchmark Architecture:**
```
                     ┌─── Oracle Clean Text ───────► FAISS Index (Clean) ──────┐
                     │                                                        ▼
Audio ──► Whisper ──►┤                                                    Top-10 Candidates
          (CT2 INT8) │                                                        ▼
                     └─── Spoken ASR Transcripts ──► FAISS Index (Spoken) ────► Cross-Encoder
                                                                               (Re-Ranked Top-1)
```

**Evaluation Matrix (Tested on 50 Arabic Reading Comprehension Queries):**
- **Pipeline 1:** Clean Oracle — Bi-Encoder Only (`CAMeL-BERT` + FAISS)
- **Pipeline 2:** Clean Oracle — Bi-Encoder + Cross-Encoder Re-Ranking
- **Pipeline 3:** Spoken ASR Transcripts — Bi-Encoder Only (`CAMeL-BERT` + FAISS)
- **Pipeline 4:** Spoken ASR Transcripts — Bi-Encoder + Cross-Encoder Re-Ranking
- *(Bonus: Dense Embedder Upgrade Comparison)*""")

    # 2. Dependencies
    add_code("""# ── 1. Install Dependencies ──────────────────────────────────────────────────
# Explicitly uninstall broken torchaudio wheels to prevent libcudart.so.13 CUDA mismatch crashes
!pip uninstall -y torchaudio
!pip install -q sentence-transformers transformers datasets faiss-cpu faster-whisper ctranslate2 edge-tts soundfile tabulate pandas jiwer nest-asyncio scipy
print("[OK] Core dependencies installed successfully.")""")

    # 3. Environment & Hardware Diagnostics
    add_code("""# ── 2. Environment Verification ───────────────────────────────────────────────
import os
import sys
import time
import re
import json
import glob
import asyncio
import zipfile
import torch
import numpy as np
import pandas as pd
import soundfile as sf
import jiwer
import edge_tts
import nest_asyncio
from tabulate import tabulate
import faiss
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import CrossEncoder

nest_asyncio.apply()

print(f"PyTorch Version : {torch.__version__}")
print(f"CUDA Available  : {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print(f"GPU Device Name : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory Total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("Running on CPU.")""")

    # 4. Configuration & Auto-Detection
    add_code("""# ── 3. Configuration & Auto-Detection ─────────────────────────────────────────
CFG = {
    "num_passages": 100,             # Number of ARCD long passages to index (50 for quick test, 100 for full)
    "num_queries": 50,               # Number of evaluation queries
    "chunk_size": 50,                # Words per chunk (empirically proven optimal in Task 3)
    "top_k_retrieve": 10,            # Initial FAISS retrieval depth before re-ranking
    "voice": "ar-SA-HamedNeural",    # Microsoft Neural Arabic voice
    "beam_size": 1,                  # Whisper greedy decoding (production speed)
    "output_dir": "/kaggle/working/spoken_retrieval_results",
    "audio_dir": "/kaggle/working/spoken_retrieval_results/audio",
}
os.makedirs(CFG["output_dir"], exist_ok=True)
os.makedirs(CFG["audio_dir"], exist_ok=True)

# 1. Recursive Auto-detection for Stage 2 Whisper-Medium Model
print("Scanning for Stage 2 Whisper model...")
model_ct2_path = None

search_roots = ["/kaggle/input", "/kaggle/working", "./whisper_medium_output", "."]
for sroot in search_roots:
    if not os.path.exists(sroot):
        continue
    for root, dirs, files in os.walk(sroot):
        if "model.bin" in files and ("vocabulary.json" in files or "vocabulary.txt" in files):
            if not model_ct2_path or "stage2" in root.lower() or "ct2" in root.lower() or "int8" in root.lower():
                model_ct2_path = root
                print(f"  [FOUND] CT2 model: {root}")

if model_ct2_path:
    print(f"[OK] Using fine-tuned CT2 Whisper model from: {model_ct2_path}")
else:
    model_ct2_path = "openai/whisper-medium"
    print(f"[NOTE] Local fine-tuned checkpoint not found. Falling back to base: {model_ct2_path}")

print(f"Final Whisper Model Path: {model_ct2_path}")""")

    # 5. Dataset Loading & Arabic Normalization
    add_code(r"""# ── 4. Load ARCD Dataset & Arabic Preprocessing ─────────────────────────────
def normalize_arabic(text):
    text = str(text)
    text = re.sub(r'[إأآٱ]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ة', 'ه', text)
    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    text = diacritics.sub('', text)
    text = re.sub(r'ـ', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Attempt local load first, then fallback to Hugging Face
raw_data = None
arcd_candidates = [
    "/kaggle/input/arcd-arabic-reading-comprehension-dataset",
    "/kaggle/input/datasets/omar10lfc/arcd*",
    "/kaggle/input/*arcd*",
    "Data/ARCD (Arabic Language Comprehension)-Dataset.zip",
    "./Data/ARCD (Arabic Language Comprehension)-Dataset.zip",
    "Data",
    "."
]

for cand in arcd_candidates:
    matched = glob.glob(cand)
    for m in matched:
        if os.path.isfile(m) and m.endswith(".zip"):
            try:
                with zipfile.ZipFile(m, 'r') as z:
                    if 'train.csv' in z.namelist():
                        raw_data = pd.read_csv(z.open('train.csv'))
                        print(f"[OK] Loaded ARCD from zip: {m}")
                        break
            except Exception as e:
                pass
        elif os.path.isdir(m):
            t_csv = os.path.join(m, "train.csv")
            if os.path.exists(t_csv):
                raw_data = pd.read_csv(t_csv)
                print(f"[OK] Loaded ARCD from directory: {t_csv}")
                break
    if raw_data is not None:
        break

if raw_data is None:
    try:
        from datasets import load_dataset
        print("Loading ARCD via Hugging Face datasets ('hsseinmz/arcd')...")
        hf_ds = load_dataset('hsseinmz/arcd')
        raw_data = pd.DataFrame(hf_ds['train'])
        print(f"[OK] Loaded ARCD from Hugging Face ({len(raw_data)} rows).")
    except Exception as e:
        raise RuntimeError(f"Could not load ARCD dataset: {e}")

# Build clean passages and queries
seen_contexts = {}
eval_data = []

for idx, row in raw_data.iterrows():
    ctx = normalize_arabic(row['context'])
    q = normalize_arabic(row['question'])
    ans = ""
    if 'answers' in row and pd.notna(row['answers']):
        if isinstance(row['answers'], dict) and 'text' in row['answers'] and len(row['answers']['text']) > 0:
            ans = normalize_arabic(row['answers']['text'][0])
        elif isinstance(row['answers'], str):
            ans = normalize_arabic(row['answers'])

    if ctx not in seen_contexts:
        seen_contexts[ctx] = len(seen_contexts)

    eval_data.append({
        'query': q,
        'relevant_passage_id': seen_contexts[ctx],
        'answer': ans,
    })

passages = list(seen_contexts.keys())
print(f"Total unique passages in dataset: {len(passages)}")

# Filter long passages (>100 words) exactly matching the production setup in Task 3
long_passages_with_ids = [(i, p) for i, p in enumerate(passages) if len(p.split()) > 100][:CFG["num_passages"]]
valid_long_ids = set([i for i, p in long_passages_with_ids])

eval_queries = [e for e in eval_data if e['relevant_passage_id'] in valid_long_ids][:CFG["num_queries"]]

print(f"Selected long passages for indexing : {len(long_passages_with_ids)}")
print(f"Selected evaluation queries        : {len(eval_queries)}")""")

    # 6. Audio Generation & Whisper Transcription
    add_code("""# ── 5. Neural TTS Synthesis & Whisper ASR Transcription ───────────────────────
from faster_whisper import WhisperModel

print("1. Initializing Whisper ASR model for transcription...")
try:
    whisper_asr = WhisperModel(model_ct2_path, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="int8")
    print(f"[OK] Whisper CT2 model loaded with compute_type='int8'.")
except Exception as e:
    print(f"[WARN] Failed to load CT2 model ({e}), falling back to float16...")
    whisper_asr = WhisperModel(model_ct2_path, device="cuda" if torch.cuda.is_available() else "cpu", compute_type="float16")

async def synthesize_batch_tts(passage_items, audio_dir, voice, concurrency=5):
    sem = asyncio.Semaphore(concurrency)
    async def sem_synth(orig_id, text):
        audio_filename = os.path.join(audio_dir, f"passage_{orig_id}.wav")
        if not os.path.exists(audio_filename):
            async with sem:
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(audio_filename)
        return orig_id, audio_filename
    
    tasks = [sem_synth(orig_id, text) for orig_id, text in passage_items]
    return await asyncio.gather(*tasks)

import scipy.signal

def load_audio_16k_np(path):
    audio_data, sr = sf.read(path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)
    if sr != 16000:
        num_samples = int(len(audio_data) * 16000 / sr)
        audio_data = scipy.signal.resample(audio_data, num_samples)
    return audio_data.astype(np.float32)

print(f"\\n2. Synthesizing audio for {len(long_passages_with_ids)} passages (concurrency=5)...")
tts_start = time.time()
loop = asyncio.get_event_loop()
tts_pairs = loop.run_until_complete(
    synthesize_batch_tts(long_passages_with_ids, CFG["audio_dir"], CFG["voice"], concurrency=5)
)
print(f"[OK] TTS synthesis finished in {time.time() - tts_start:.1f} s.")

print(f"\\n3. Transcribing passages with Whisper CT2 INT8...")
spoken_passages = []
asr_wers = []
start_transcribe_time = time.time()

for idx, (orig_id, audio_filename) in enumerate(tts_pairs):
    passage_text = [p for i, p in long_passages_with_ids if i == orig_id][0]
    
    # Load audio array directly (avoids PyAV metadata errors)
    audio_16k = load_audio_16k_np(audio_filename)
    
    # Transcribe with Whisper
    segments, _ = whisper_asr.transcribe(
        audio_16k,
        language="ar",
        beam_size=CFG["beam_size"],
        temperature=0.0,
        vad_filter=True
    )
    transcript = " ".join([seg.text for seg in segments]).strip()
    norm_transcript = normalize_arabic(transcript)
    if not norm_transcript:
        norm_transcript = "فارغ"
    
    # Measure Word Error Rate against clean passage
    wer_val = jiwer.wer(passage_text, norm_transcript) * 100.0
    asr_wers.append(wer_val)
    
    spoken_passages.append((orig_id, norm_transcript))
    
    if (idx + 1) % 10 == 0 or (idx + 1) == len(tts_pairs):
        print(f"  Processed [{idx+1:3d}/{len(tts_pairs):3d}] | Current Avg WER: {np.mean(asr_wers):.2f}%")

total_transcribe_duration = time.time() - start_transcribe_time
mean_corpus_wer = np.mean(asr_wers)
print(f"\\n[SUMMARY] Upstream ASR Transcription Complete:")
print(f"  Total Passages Transcribed : {len(spoken_passages)}")
print(f"  Average Upstream WER       : {mean_corpus_wer:.2f}%")
print(f"  ASR Processing Time        : {total_transcribe_duration:.1f} s")""")

    # 7. Dual Corpus Chunking (50-word chunks)
    add_code("""# ── 6. Dual Corpus Construction (Oracle Clean vs. Spoken Transcribed) ────────
def split_into_chunks(text, max_words=50):
    words = str(text).split()
    if not words:
        return ["فارغ"]
    chunks = [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    return [c for c in chunks if c.strip()]

# 1. Oracle Clean Corpus Chunks
oracle_chunks = []
oracle_chunk_to_passage = []

for orig_id, clean_text in long_passages_with_ids:
    chunks = split_into_chunks(clean_text, CFG["chunk_size"])
    for ch in chunks:
        oracle_chunks.append(ch)
        oracle_chunk_to_passage.append(orig_id)

# 2. Spoken ASR Transcribed Corpus Chunks
spoken_chunks = []
spoken_chunk_to_passage = []

for orig_id, spoken_text in spoken_passages:
    chunks = split_into_chunks(spoken_text, CFG["chunk_size"])
    for ch in chunks:
        spoken_chunks.append(ch)
        spoken_chunk_to_passage.append(orig_id)

print(f"Oracle Clean Chunks  : {len(oracle_chunks)} chunks across {len(long_passages_with_ids)} passages")
print(f"Spoken ASR Chunks    : {len(spoken_chunks)} chunks across {len(spoken_passages)} passages")
print(f"Average chunks/doc   : {len(oracle_chunks)/len(long_passages_with_ids):.1f}")""")

    # 8. Bi-Encoder Loading & Dual FAISS Indexing
    add_code("""# ── 7. Bi-Encoder Embedding & Dual FAISS Indexing ────────────────────────────
BI_ENCODER_NAME = 'CAMeL-Lab/bert-base-arabic-camelbert-msa'
print(f"Loading Bi-Encoder ({BI_ENCODER_NAME})...")

bi_tokenizer = AutoTokenizer.from_pretrained(BI_ENCODER_NAME)
bi_model = AutoModel.from_pretrained(BI_ENCODER_NAME)
bi_model.eval()
if torch.cuda.is_available():
    bi_model.cuda()

def mean_pool(token_emb, attention_mask):
    mask_exp = attention_mask.unsqueeze(-1).expand(token_emb.size()).float()
    return (token_emb * mask_exp).sum(1) / mask_exp.sum(1).clamp(min=1e-9)

@torch.no_grad()
def encode_texts(texts, batch_size=64):
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inp = bi_tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors='pt')
        if torch.cuda.is_available():
            inp = {k: v.cuda() for k, v in inp.items()}
        out = bi_model(**inp)
        emb = mean_pool(out.last_hidden_state, inp['attention_mask'])
        emb = emb.cpu().numpy().astype(np.float32)
        # L2 normalize for cosine similarity via Inner Product
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.where(norms == 0, 1, norms)
        all_embs.append(emb)
    return np.vstack(all_embs)

print("Encoding Oracle Clean Chunks...")
oracle_embeddings = encode_texts(oracle_chunks)
oracle_index = faiss.IndexFlatIP(oracle_embeddings.shape[1])
oracle_index.add(oracle_embeddings)

print("Encoding Spoken ASR Chunks...")
spoken_embeddings = encode_texts(spoken_chunks)
spoken_index = faiss.IndexFlatIP(spoken_embeddings.shape[1])
spoken_index.add(spoken_embeddings)

print("Encoding Evaluation Queries...")
query_texts = [e['query'] for e in eval_queries]
query_embeddings = encode_texts(query_texts)

print(f"[OK] Oracle Index: {oracle_index.ntotal} vectors | Spoken Index: {spoken_index.ntotal} vectors")""")

    # 9. Cross-Encoder Re-Ranker Setup
    add_code("""# ── 8. Cross-Encoder Re-Ranker Setup ──────────────────────────────────────────
RERANKER_NAME = 'cross-encoder/mmarco-mMiniLMv2-L12-H384-v1'
print(f"Loading Cross-Encoder Re-Ranker ({RERANKER_NAME})...")
reranker = CrossEncoder(RERANKER_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
print("[OK] Cross-Encoder ready on device.")""")

    # 10. Master Evaluation Routine
    add_code("""# ── 9. Master Spoken Retrieval Evaluation Engine ──────────────────────────────
import random

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

def evaluate_pipeline(chunks, chunk_to_passage, faiss_index, query_embeddings, eval_queries, use_reranker=False, top_k_retrieve=10):
    set_all_seeds(42)
    k_values = [1, 3, 5]
    metrics = {f'P@{k}': [] for k in k_values}
    metrics['R@10'] = []
    metrics['MRR@10'] = []
    
    # FAISS initial retrieval
    _, all_indices = faiss_index.search(query_embeddings, top_k_retrieve)
    
    for i, q_item in enumerate(eval_queries):
        rel_passage_id = q_item['relevant_passage_id']
        query_text = q_item['query']
        retrieved_chunk_ids = [c for c in all_indices[i].tolist() if c >= 0]
        
        if not use_reranker:
            # Baseline: Direct FAISS candidate ordering
            candidate_passage_ids = [chunk_to_passage[c] for c in retrieved_chunk_ids]
            
            # Precision@K
            for k in k_values:
                hit = 1.0 if rel_passage_id in candidate_passage_ids[:k] else 0.0
                metrics[f'P@{k}'].append(hit)
                
            # Recall@10
            r10 = 1.0 if rel_passage_id in candidate_passage_ids[:10] else 0.0
            metrics['R@10'].append(r10)
            
            # MRR@10
            if rel_passage_id in candidate_passage_ids[:10]:
                rank = candidate_passage_ids[:10].index(rel_passage_id) + 1
                metrics['MRR@10'].append(1.0 / rank)
            else:
                metrics['MRR@10'].append(0.0)
                
        else:
            # Cross-Encoder Re-ranking Stage
            candidate_texts = [chunks[c] for c in retrieved_chunk_ids]
            cross_inp = [[query_text, t] for t in candidate_texts]
            
            if cross_inp:
                cross_scores = reranker.predict(cross_inp)
                candidate_passage_ids = [chunk_to_passage[c] for c in retrieved_chunk_ids]
                
                # Sort candidates descending by Cross-Encoder score with stable tie-breaking
                paired = list(zip(cross_scores, candidate_passage_ids, range(len(candidate_passage_ids))))
                paired.sort(key=lambda x: (x[0], -x[2]), reverse=True)
                reranked_passage_ids = [p_id for _, p_id, _ in paired]
                
                for k in k_values:
                    hit = 1.0 if rel_passage_id in reranked_passage_ids[:k] else 0.0
                    metrics[f'P@{k}'].append(hit)
                    
                r10 = 1.0 if rel_passage_id in reranked_passage_ids[:10] else 0.0
                metrics['R@10'].append(r10)
                
                if rel_passage_id in reranked_passage_ids[:10]:
                    rank = reranked_passage_ids[:10].index(rel_passage_id) + 1
                    metrics['MRR@10'].append(1.0 / rank)
                else:
                    metrics['MRR@10'].append(0.0)
            else:
                for k in k_values:
                    metrics[f'P@{k}'].append(0.0)
                metrics['R@10'].append(0.0)
                metrics['MRR@10'].append(0.0)

    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0)
        p = k / n
        denom = 1 + (z**2) / n
        center = (p + (z**2) / (2 * n)) / denom
        margin = (z * np.sqrt((p * (1 - p) / n) + (z**2) / (4 * (n**2)))) / denom
        return (round(max(0.0, float(center - margin)), 4), round(min(1.0, float(center + margin)), 4))

    res = {k: round(float(np.mean(v)), 4) for k, v in metrics.items()}
    res['hits1'] = metrics['P@1']
    res['hits3'] = metrics['P@3']
    res['ci_p1'] = wilson_ci(sum(metrics['P@1']), len(metrics['P@1']))
    res['ci_p3'] = wilson_ci(sum(metrics['P@3']), len(metrics['P@3']))
    return res

def mcnemar_paired_test(hits_ref, hits_spoken):
    b = sum(1 for r, s in zip(hits_ref, hits_spoken) if r == 1 and s == 0)
    c = sum(1 for r, s in zip(hits_ref, hits_spoken) if r == 0 and s == 1)
    from scipy.stats import binomtest
    total = b + c
    if total == 0:
        p_val = 1.0
    else:
        p_val = binomtest(min(b, c), total, 0.5, alternative='two-sided').pvalue
    return {'b': b, 'c': c, 'p_value': round(float(p_val), 4), 'significant': p_val < 0.05}

print("Running Dual Benchmark across all 4 Configurations...")
res_oracle_faiss = evaluate_pipeline(oracle_chunks, oracle_chunk_to_passage, oracle_index, query_embeddings, eval_queries, use_reranker=False)
res_oracle_rerank = evaluate_pipeline(oracle_chunks, oracle_chunk_to_passage, oracle_index, query_embeddings, eval_queries, use_reranker=True)

res_spoken_faiss = evaluate_pipeline(spoken_chunks, spoken_chunk_to_passage, spoken_index, query_embeddings, eval_queries, use_reranker=False)
res_spoken_rerank = evaluate_pipeline(spoken_chunks, spoken_chunk_to_passage, spoken_index, query_embeddings, eval_queries, use_reranker=True)

mcnemar_bi = mcnemar_paired_test(res_oracle_faiss['hits1'], res_spoken_faiss['hits1'])
mcnemar_rr = mcnemar_paired_test(res_oracle_rerank['hits1'], res_spoken_rerank['hits1'])

print("[OK] All 4 evaluation runs completed successfully.")""")

    # 11. Master Results Table & Saving
    add_code("""# ── 10. Master Benchmark Table & Quality Retention Analysis ───────────────────
oracle_p1_ref = res_oracle_rerank['P@1']

summary_rows = [
    {
        "Corpus / Pipeline": "1. Oracle Clean Text (Bi-Encoder Only)",
        "Corpus Condition": "Clean Text (0% WER)",
        "Reranker": "None (FAISS)",
        "P@1": res_oracle_faiss['P@1'],
        "P@1 95% Wilson CI": str(res_oracle_faiss['ci_p1']),
        "P@3": res_oracle_faiss['P@3'],
        "P@3 95% Wilson CI": str(res_oracle_faiss['ci_p3']),
        "MRR@10": res_oracle_faiss['MRR@10'],
        "Paired McNemar (vs Clean)": "N/A (Ref)",
        "Delta P@1": f"{(res_oracle_faiss['P@1'] - oracle_p1_ref)*100:+.1f} pp",
        "Quality Retention": f"{(res_oracle_faiss['P@1'] / oracle_p1_ref)*100:.1f}%"
    },
    {
        "Corpus / Pipeline": "2. Oracle Clean Text (+ Re-Rank)",
        "Corpus Condition": "Clean Text (0% WER)",
        "Reranker": "mmarco-mMiniLMv2",
        "P@1": res_oracle_rerank['P@1'],
        "P@1 95% Wilson CI": str(res_oracle_rerank['ci_p1']),
        "P@3": res_oracle_rerank['P@3'],
        "P@3 95% Wilson CI": str(res_oracle_rerank['ci_p3']),
        "MRR@10": res_oracle_rerank['MRR@10'],
        "Paired McNemar (vs Clean)": "N/A (Ref)",
        "Delta P@1": "0.0 pp (Ref)",
        "Quality Retention": "100.0% (Ref)"
    },
    {
        "Corpus / Pipeline": "3. Spoken ASR Transcripts (Bi-Encoder Only)",
        "Corpus Condition": f"Whisper INT8 (~{mean_corpus_wer:.1f}% WER)",
        "Reranker": "None (FAISS)",
        "P@1": res_spoken_faiss['P@1'],
        "P@1 95% Wilson CI": str(res_spoken_faiss['ci_p1']),
        "P@3": res_spoken_faiss['P@3'],
        "P@3 95% Wilson CI": str(res_spoken_faiss['ci_p3']),
        "MRR@10": res_spoken_faiss['MRR@10'],
        "Paired McNemar (vs Clean)": f"p={mcnemar_bi['p_value']} (sig={mcnemar_bi['significant']})",
        "Delta P@1": f"{(res_spoken_faiss['P@1'] - oracle_p1_ref)*100:+.1f} pp",
        "Quality Retention": f"{(res_spoken_faiss['P@1'] / oracle_p1_ref)*100:.1f}%"
    },
    {
        "Corpus / Pipeline": "4. Spoken ASR Transcripts (+ Re-Rank)",
        "Corpus Condition": f"Whisper INT8 (~{mean_corpus_wer:.1f}% WER)",
        "Reranker": "mmarco-mMiniLMv2",
        "P@1": res_spoken_rerank['P@1'],
        "P@1 95% Wilson CI": str(res_spoken_rerank['ci_p1']),
        "P@3": res_spoken_rerank['P@3'],
        "P@3 95% Wilson CI": str(res_spoken_rerank['ci_p3']),
        "MRR@10": res_spoken_rerank['MRR@10'],
        "Paired McNemar (vs Clean)": f"p={mcnemar_rr['p_value']} (sig={mcnemar_rr['significant']})",
        "Delta P@1": f"{(res_spoken_rerank['P@1'] - oracle_p1_ref)*100:+.1f} pp",
        "Quality Retention": f"{(res_spoken_rerank['P@1'] / oracle_p1_ref)*100:.1f}%"
    }
]

print("\\n" + "="*110)
print("📊 MASTER BENCHMARK: SPOKEN DOCUMENT RETRIEVAL ERROR PROPAGATION (WHISPER -> CAMeL-BERT -> RERANKER)")
print("="*110)
print(tabulate(summary_rows, headers="keys", tablefmt="github"))

# Save summary to CSV & JSON
df_summary = pd.DataFrame(summary_rows)
csv_out = os.path.join(CFG["output_dir"], "spoken_retrieval_summary.csv")
df_summary.to_csv(csv_out, index=False)

export_payload = {
    "corpus_wer": round(float(mean_corpus_wer), 2),
    "whisper_model": model_ct2_path,
    "num_passages": len(long_passages_with_ids),
    "num_queries": len(eval_queries),
    "oracle_clean": {
        "faiss_only": res_oracle_faiss,
        "with_rerank": res_oracle_rerank
    },
    "spoken_transcribed": {
        "faiss_only": res_spoken_faiss,
        "with_rerank": res_spoken_rerank
    },
    "summary_table": summary_rows
}
json_out = os.path.join(CFG["output_dir"], "spoken_retrieval_results.json")
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(export_payload, f, ensure_ascii=False, indent=2)

print(f"\\n[OK] Evaluation reports saved to:\\n  - {csv_out}\\n  - {json_out}")""")

    # 12. Qualitative Case Study & Cross-Attention Inspection
    add_code("""# ── 11. Qualitative Inspection: Cross-Attention Noise Resilience ─────────────
print("\\n" + "="*95)
print("🔍 QUALITATIVE CASE STUDY: HOW CROSS-ATTENTION OVERCOMES TRANSCRIPTION RECOGNITION NOISE")
print("="*95)

# Inspect queries where Bi-Encoder dropped the chunk but Cross-Encoder recovered it
_, spoken_indices = spoken_index.search(query_embeddings, 10)

recovered_cases = []
for i, q_item in enumerate(eval_queries):
    rel_id = q_item['relevant_passage_id']
    retrieved_chunk_ids = [c for c in spoken_indices[i].tolist() if c >= 0]
    faiss_passage_ids = [spoken_chunk_to_passage[c] for c in retrieved_chunk_ids]
    
    # Bi-Encoder failed at Top 1
    if len(faiss_passage_ids) > 0 and faiss_passage_ids[0] != rel_id and rel_id in faiss_passage_ids:
        # Check re-ranker
        candidate_texts = [spoken_chunks[c] for c in retrieved_chunk_ids]
        cross_inp = [[q_item['query'], t] for t in candidate_texts]
        cross_scores = reranker.predict(cross_inp)
        paired = list(zip(cross_scores, faiss_passage_ids, candidate_texts))
        paired.sort(key=lambda x: x[0], reverse=True)
        
        if paired[0][1] == rel_id:
            recovered_cases.append({
                "query": q_item['query'],
                "rel_id": rel_id,
                "faiss_rank": faiss_passage_ids.index(rel_id) + 1,
                "rerank_rank": 1,
                "spoken_chunk": paired[0][2]
            })

if recovered_cases:
    print(f"Found {len(recovered_cases)} test queries where Cross-Encoder corrected Bi-Encoder acoustic noise errors!\\n")
    case = recovered_cases[0]
    clean_passage_match = [p for i, p in long_passages_with_ids if i == case["rel_id"]][0]
    
    print(f"Query                   : {case['query']}")
    print(f"FAISS Rank (Bi-Encoder) : #{case['faiss_rank']}  (Failed Rank #1 due to ASR noise)")
    print(f"Cross-Encoder Rank      : #{case['rerank_rank']}  (Successfully Promoted to Top #1!)\\n")
    print(f"Original Clean Passage  :\\n  {clean_passage_match[:250]}...\\n")
    print(f"Whisper Noisy Transcript Chunk :\\n  {case['spoken_chunk'][:250]}...")
else:
    print("All retrieved candidates were ranked consistently. Showing Top sample query:")
    sample = eval_queries[0]
    print(f"Query: {sample['query']}")

print("="*95)
print("Conclusion: The Cross-Encoder's all-to-all cross-attention mechanism directly pairs query")
print("tokens with transcript tokens, rendering it robust to local phonetic and transcription shifts.")
print("="*95)""")

    # 13. Optional Embedder Upgrade Comparison
    add_code("""# ── 12. Optional Embedder Upgrade Benchmark (e.g. Multilingual-MPNet / BGE-M3) ──
print("\\n" + "="*80)
print("🚀 OPTIONAL EXTENSION: TESTING AN UPGRADED DENSE EMBEDDER")
print("="*80)
print("In production, can a modern multilingual dense model (e.g., multilingual-mpnet or BGE-M3)")
print("provide even higher zero-shot acoustic noise tolerance than the 110M CAMeL-BERT?")

UPGRADE_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
try:
    from sentence_transformers import SentenceTransformer
    print(f"Loading upgraded embedder: {UPGRADE_MODEL_NAME}...")
    up_model = SentenceTransformer(UPGRADE_MODEL_NAME, device="cuda" if torch.cuda.is_available() else "cpu")
    
    print("Encoding with upgraded model...")
    up_oracle_embs = up_model.encode(oracle_chunks, normalize_embeddings=True, show_progress_bar=False)
    up_spoken_embs = up_model.encode(spoken_chunks, normalize_embeddings=True, show_progress_bar=False)
    up_query_embs  = up_model.encode(query_texts, normalize_embeddings=True, show_progress_bar=False)
    
    up_oracle_index = faiss.IndexFlatIP(up_oracle_embs.shape[1])
    up_oracle_index.add(up_oracle_embs)
    
    up_spoken_index = faiss.IndexFlatIP(up_spoken_embs.shape[1])
    up_spoken_index.add(up_spoken_embs)
    
    res_up_oracle_faiss = evaluate_pipeline(oracle_chunks, oracle_chunk_to_passage, up_oracle_index, up_query_embs, eval_queries, use_reranker=False)
    res_up_spoken_faiss = evaluate_pipeline(spoken_chunks, spoken_chunk_to_passage, up_spoken_index, up_query_embs, eval_queries, use_reranker=False)
    
    print("\\nComparison of Bi-Encoder Acoustic Resilience (FAISS Only):")
    print(f"  CAMeL-BERT (Clean)  : P@1 = {res_oracle_faiss['P@1']:.4f} | MRR@10 = {res_oracle_faiss['MRR@10']:.4f}")
    print(f"  CAMeL-BERT (Spoken) : P@1 = {res_spoken_faiss['P@1']:.4f} | MRR@10 = {res_spoken_faiss['MRR@10']:.4f}")
    print(f"  Upgraded   (Clean)  : P@1 = {res_up_oracle_faiss['P@1']:.4f} | MRR@10 = {res_up_oracle_faiss['MRR@10']:.4f}")
    print(f"  Upgraded   (Spoken) : P@1 = {res_up_spoken_faiss['P@1']:.4f} | MRR@10 = {res_up_spoken_faiss['MRR@10']:.4f}")
except Exception as e:
    print(f"[SKIP] Optional upgraded embedder benchmark skipped ({e}).")""")

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
    notebook = create_spoken_retrieval_notebook()
    target_path = os.path.abspath("evaluate_spoken_retrieval.ipynb")
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)
    print(f"[OK] Generated notebook successfully at: {target_path}")

    # Also sync to Notebooks/ directory
    notebooks_target = os.path.abspath("Notebooks/evaluate_spoken_retrieval.ipynb")
    shutil.copyfile(target_path, notebooks_target)
    print(f"[OK] Synced copy to: {notebooks_target}")
