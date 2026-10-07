import os, sys, json, time, tarfile, re
import numpy as np
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

sys.stdout.reconfigure(encoding='utf-8')

# 1. Custom Hand-Rolled Scorer (Verbatim from build script)
class ArabicRougeScorer:
    def __init__(self, use_stemmer=True):
        self.use_stemmer = use_stemmer
        try:
            self.stemmer = SnowballStemmer('arabic') if use_stemmer else None
        except Exception:
            self.stemmer = None

    def _tokenize(self, text):
        if not text or not isinstance(text, str):
            return []
        cleaned = re.sub(r'[^\w\s\u0600-\u06FF]', ' ', text)
        words = cleaned.strip().split()
        if self.use_stemmer and self.stemmer:
            return [self.stemmer.stem(w) for w in words if w]
        return [w for w in words if w]

    @staticmethod
    def _ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    def _rouge_n(self, ref_tokens, pred_tokens, n):
        if len(ref_tokens) < n or len(pred_tokens) < n:
            return 0.0
        ref_ng = Counter(self._ngrams(ref_tokens, n))
        pred_ng = Counter(self._ngrams(pred_tokens, n))
        overlap = sum((ref_ng & pred_ng).values())
        if overlap == 0:
            return 0.0
        prec = overlap / len(pred_tokens)
        rec = overlap / len(ref_tokens)
        return 2 * prec * rec / (prec + rec)

    @staticmethod
    def _lcs(x, y):
        m, n = len(x), len(y)
        dp = [[0]*(n+1) for _ in range(m+1)]
        for i in range(m):
            for j in range(n):
                if x[i] == y[j]:
                    dp[i+1][j+1] = dp[i][j] + 1
                else:
                    dp[i+1][j+1] = max(dp[i+1][j], dp[i][j+1])
        return dp[m][n]

    def _rouge_l(self, ref_tokens, pred_tokens):
        if not ref_tokens or not pred_tokens:
            return 0.0
        match = self._lcs(ref_tokens, pred_tokens)
        if match == 0:
            return 0.0
        prec = match / len(pred_tokens)
        rec = match / len(ref_tokens)
        return 2 * prec * rec / (prec + rec)

    def score(self, ref, pred):
        ref_toks = self._tokenize(ref)
        pred_toks = self._tokenize(pred)
        return {
            'rouge1': self._rouge_n(ref_toks, pred_toks, 1),
            'rouge2': self._rouge_n(ref_toks, pred_toks, 2),
            'rougeL': self._rouge_l(ref_toks, pred_toks),
        }

# 2. Official XL-Sum Scorer
official_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, lang='arabic')
custom_scorer = ArabicRougeScorer(use_stemmer=True)

# 3. Load 25 articles from Data/arabic_XLSum_v2.0.tar.bz2
articles = []
with tarfile.open('Data/arabic_XLSum_v2.0.tar.bz2', 'r:bz2') as tar:
    f = tar.extractfile('./arabic_test.jsonl')
    for idx, line in enumerate(f):
        if idx >= 25:
            break
        record = json.loads(line.decode('utf-8'))
        articles.append({
            'id': record.get('id', str(idx)),
            'text': record.get('text', ''),
            'summary': record.get('summary', ''),
        })

print(f'Loaded {len(articles)} articles from XL-Sum archive.')

# 4. Load AraBART
print('Loading AraBART from Summarizer/ ...')
tokenizer = AutoTokenizer.from_pretrained('Summarizer')
model = AutoModelForSeq2SeqLM.from_pretrained('Summarizer')
model.eval()

def generate_summary(text):
    inputs = tokenizer([text], max_length=512, truncation=True, padding='longest', return_tensors='pt')
    with torch.no_grad():
        summary_ids = model.generate(
            **inputs,
            max_length=64,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print('Generating summaries for 25 articles (Oracle Clean Text)...')
oracle_summaries = []
for idx, art in enumerate(articles):
    t0 = time.time()
    s = generate_summary(art['text'])
    oracle_summaries.append(s)
    print(f'  Article {idx+1:02d}/25 generated in {time.time()-t0:.1f}s')

# Score Oracle Clean Text with both scorers
custom_r1, custom_r2, custom_rl = [], [], []
official_r1, official_r2, official_rl = [], [], []

for art, o_sum in zip(articles, oracle_summaries):
    ref = art['summary']
    c_s = custom_scorer.score(ref, o_sum)
    o_s = official_scorer.score(ref, o_sum)

    custom_r1.append(c_s['rouge1'] * 100)
    custom_r2.append(c_s['rouge2'] * 100)
    custom_rl.append(c_s['rougeL'] * 100)

    official_r1.append(o_s['rouge1'].fmeasure * 100)
    official_r2.append(o_s['rouge2'].fmeasure * 100)
    official_rl.append(o_s['rougeL'].fmeasure * 100)

print('\n=== Oracle Baseline Results (N=25) ===')
print(f'Custom   Scorer: ROUGE-1={np.mean(custom_r1):.2f}, ROUGE-2={np.mean(custom_r2):.2f}, ROUGE-L={np.mean(custom_rl):.2f}')
print(f'Official Scorer: ROUGE-1={np.mean(official_r1):.2f}, ROUGE-2={np.mean(official_r2):.2f}, ROUGE-L={np.mean(official_rl):.2f}')

# Save generated oracle summaries
with open('scratch/oracle_eval_cache.json', 'w', encoding='utf-8') as f:
    json.dump({
        'articles': articles,
        'oracle_summaries': oracle_summaries,
        'custom_scores': {'r1': custom_r1, 'r2': custom_r2, 'rl': custom_rl},
        'official_scores': {'r1': official_r1, 'r2': official_r2, 'rl': official_rl},
    }, f, indent=2, ensure_ascii=False)
print('Saved oracle cache to scratch/oracle_eval_cache.json')
