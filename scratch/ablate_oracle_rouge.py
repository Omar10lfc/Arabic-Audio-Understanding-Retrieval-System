import json, tarfile, os, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

articles = []
if os.path.exists('scratch/oracle_eval_cache.json'):
    with open('scratch/oracle_eval_cache.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        articles = data.get('articles', [])[:5]
elif os.path.exists('arabic_XLSum_v2.0/arabic_test.jsonl'):
    with open('arabic_XLSum_v2.0/arabic_test.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
            if len(articles) == 5:
                break

print(f"Loaded {len(articles)} articles.")

model_id = 'Omar10lfc/arabart-xlsum-arabic'
tokenizer = AutoTokenizer.from_pretrained(model_id)
device = 'cpu'
model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(device).eval()

scorer_official = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True, lang='arabic')
scorer_unstemmed = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)

def run_eval(padding='longest', max_len=100, min_len=0, num_beams=4, no_repeat=3, len_penalty=1.0, scorer=scorer_official):
    r_l_list = []
    for art in articles:
        inp = tokenizer([art['text']], max_length=512, truncation=True, padding=padding, return_tensors='pt').to(device)
        kwargs = dict(max_length=max_len, num_beams=num_beams, no_repeat_ngram_size=no_repeat, length_penalty=len_penalty, early_stopping=True)
        if min_len > 0:
            kwargs['min_length'] = min_len
        with torch.no_grad():
            out = model.generate(**inp, **kwargs)
        pred = tokenizer.decode(out[0], skip_special_tokens=True).strip()
        ref = art['summary'].strip()
        score = scorer.score(ref, pred)['rougeL'].fmeasure * 100.0
        r_l_list.append(score)
    return sum(r_l_list) / len(r_l_list)

# Controlled variations
base1 = run_eval()
base2 = run_eval()
print(f"1. Base run 1 (deterministic check) : {base1:.4f}")
print(f"   Base run 2 (deterministic check) : {base2:.4f}")
print(f"2. Change padding='max_length'      : {run_eval(padding='max_length'):.4f}")
print(f"3. Change max_length=128            : {run_eval(max_len=128):.4f}")
print(f"   Change max_length=84             : {run_eval(max_len=84):.4f}")
print(f"4. Change min_length=10             : {run_eval(min_len=10):.4f}")
print(f"   Change min_length=12             : {run_eval(min_len=12):.4f}")
print(f"5. Change length_penalty=0.6        : {run_eval(len_penalty=0.6):.4f}")
print(f"6. Change no_repeat_ngram_size=2    : {run_eval(no_repeat=2):.4f}")
print(f"7. Change scorer (unstemmed default): {run_eval(scorer=scorer_unstemmed):.4f}")
