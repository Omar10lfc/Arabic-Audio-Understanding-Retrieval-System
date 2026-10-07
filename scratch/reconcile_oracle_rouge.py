import time, json, tarfile, torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

model_id = 'Omar10lfc/arabart-xlsum-arabic'
print("Loading AraBART model...", flush=True)
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).eval()
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, lang='arabic')

with tarfile.open('Data/arabic_XLSum_v2.0.tar.bz2', 'r:bz2') as tar:
    f = tar.extractfile('./arabic_test.jsonl')
    raw_lines = [line.decode('utf-8') for line in f]

# Set 1: Downstream protocol (first 25, no filter)
downstream_25 = [json.loads(l) for l in raw_lines[:25]]

# Set 2: Error propagation protocol (first 25 passing 80 <= wc <= 250 and summary >= 10)
err_prop_25 = []
for l in raw_lines:
    rec = json.loads(l)
    text = rec.get('text', '')
    summary = rec.get('summary', '')
    if 80 <= len(text.split()) <= 250 and len(summary.split()) >= 10:
        err_prop_25.append(rec)
        if len(err_prop_25) >= 25:
            break

def evaluate_set(articles, label, no_repeat_ngram_size=3, length_penalty=1.0):
    t0 = time.time()
    rLs, r1s, r2s = [], [], []
    for art in articles:
        inp = tok([art['text']], max_length=512, truncation=True, padding='longest', return_tensors='pt')
        with torch.no_grad():
            gen_kwargs = {
                'max_length': 64,
                'num_beams': 4,
                'no_repeat_ngram_size': no_repeat_ngram_size,
                'early_stopping': True,
            }
            if length_penalty != 1.0:
                gen_kwargs['length_penalty'] = length_penalty
            out = model.generate(**inp, **gen_kwargs)
        pred = tok.decode(out[0], skip_special_tokens=True).strip()
        score = scorer.score(art['summary'], pred)
        r1s.append(score['rouge1'].fmeasure * 100)
        r2s.append(score['rouge2'].fmeasure * 100)
        rLs.append(score['rougeL'].fmeasure * 100)
    print(f"=== {label} (N={len(articles)}, Time={time.time()-t0:.1f}s) ===")
    print(f"Params: no_repeat_ngram_size={no_repeat_ngram_size}, length_penalty={length_penalty}")
    print(f"ROUGE-1: {np.mean(r1s):.2f} | ROUGE-2: {np.mean(r2s):.2f} | ROUGE-L: {np.mean(rLs):.2f}")
    return np.mean(rLs)

print("\n--- Test 1: Downstream Articles with Downstream Params ---", flush=True)
evaluate_set(downstream_25, "Downstream Selection + Downstream Params", no_repeat_ngram_size=3, length_penalty=1.0)

print("\n--- Test 2: Error Prop Articles with Error Prop Params ---", flush=True)
evaluate_set(err_prop_25, "Error Prop Selection + Error Prop Params", no_repeat_ngram_size=2, length_penalty=0.6)

print("\n--- Test 3: Downstream Articles with Error Prop Params ---", flush=True)
evaluate_set(downstream_25, "Downstream Selection + Error Prop Params", no_repeat_ngram_size=2, length_penalty=0.6)

print("\n--- Test 4: Error Prop Articles with Downstream Params ---", flush=True)
evaluate_set(err_prop_25, "Error Prop Selection + Downstream Params", no_repeat_ngram_size=3, length_penalty=1.0)
