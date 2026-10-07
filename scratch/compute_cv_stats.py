import pandas as pd
import numpy as np
import re

def normalize_arabic(text: str) -> str:
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[\u0625\u0623\u0622\u0671]', '\u0627', text)
    text = re.sub(r'\u0649', '\u064a', text)
    text = re.sub(r'\u0629', '\u0647', text)
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    text = re.sub(r'\u0640', '', text)
    text = re.sub(r'[^\w\s\u0600-\u06FF]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df = pd.read_csv('scratch/test.tsv', sep='\t', low_memory=False)
df['clean_ref'] = df['sentence'].apply(normalize_arabic)
df = df[df['clean_ref'].str.len() > 2].reset_index(drop=True)

# Subset A: 400 random clips (seed 42)
sub_a = df.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:400].copy()

# Subset B: 400 sequential clips
sub_b = df.iloc[:400].copy()

# Subset C: 100 random clips (seed 42)
sub_c = df.sample(frac=1, random_state=42).reset_index(drop=True).iloc[:100].copy()

# Subset D: 100 sequential clips
sub_d = df.iloc[:100].copy()

def get_stats(series, name):
    return {
        'name': name,
        'mean': float(np.mean(series)),
        'std': float(np.std(series)),
        'median': float(np.median(series)),
        'min': float(np.min(series)),
        'max': float(np.max(series)),
        'p25': float(np.percentile(series, 25)),
        'p50': float(np.percentile(series, 50)),
        'p75': float(np.percentile(series, 75)),
        'p90': float(np.percentile(series, 90)),
        'p95': float(np.percentile(series, 95)),
    }

sub_a['word_count'] = sub_a['clean_ref'].apply(lambda x: len(x.split()))
sub_b['word_count'] = sub_b['clean_ref'].apply(lambda x: len(x.split()))
sub_c['word_count'] = sub_c['clean_ref'].apply(lambda x: len(x.split()))
sub_d['word_count'] = sub_d['clean_ref'].apply(lambda x: len(x.split()))
df['word_count'] = df['clean_ref'].apply(lambda x: len(x.split()))

sub_a['char_count'] = sub_a['clean_ref'].apply(len)
sub_b['char_count'] = sub_b['clean_ref'].apply(len)
sub_c['char_count'] = sub_c['clean_ref'].apply(len)
sub_d['char_count'] = sub_d['clean_ref'].apply(len)
df['char_count'] = df['clean_ref'].apply(len)

print("=== Reference Word Count Stats ===")
for s, n in [(sub_a['word_count'], 'Subset A (400 Random)'),
            (sub_b['word_count'], 'Subset B (400 Sequential)'),
            (sub_c['word_count'], 'Subset C (100 Random)'),
            (sub_d['word_count'], 'Subset D (100 Sequential)'),
            (df['word_count'], 'Full Pool (10,506 clips)')]:
    st = get_stats(s, n)
    print(f"{n:28s} | Mean: {st['mean']:.2f} | Med: {st['median']:.1f} | Std: {st['std']:.2f} | P25: {st['p25']:.1f} | P75: {st['p75']:.1f} | P90: {st['p90']:.1f} | Max: {st['max']:.0f}")

print("\n=== Reference Character Count Stats ===")
for s, n in [(sub_a['char_count'], 'Subset A (400 Random)'),
            (sub_b['char_count'], 'Subset B (400 Sequential)'),
            (sub_c['char_count'], 'Subset C (100 Random)'),
            (sub_d['char_count'], 'Subset D (100 Sequential)'),
            (df['char_count'], 'Full Pool (10,506 clips)')]:
    st = get_stats(s, n)
    print(f"{n:28s} | Mean: {st['mean']:.2f} | Med: {st['median']:.1f} | Std: {st['std']:.2f} | P25: {st['p25']:.1f} | P75: {st['p75']:.1f} | P90: {st['p90']:.1f} | Max: {st['max']:.0f}")

# Histogram / Bins of word count
bins = [0, 3, 5, 7, 10, 15, 20, 50]
sub_a['wc_bin'] = pd.cut(sub_a['word_count'], bins=bins)
sub_b['wc_bin'] = pd.cut(sub_b['word_count'], bins=bins)

print("\n=== Word Count Distribution (Counts & Percentages) ===")
print("Bin          | Subset A (400 Rand)      | Subset B (400 Seq)")
print("-" * 55)
for b in sub_a['wc_bin'].cat.categories:
    cnt_a = (sub_a['wc_bin'] == b).sum()
    pct_a = cnt_a / len(sub_a) * 100
    cnt_b = (sub_b['wc_bin'] == b).sum()
    pct_b = cnt_b / len(sub_b) * 100
    print(f"{str(b):12s} | {cnt_a:3d} ({pct_a:5.1f}%)            | {cnt_b:3d} ({pct_b:5.1f}%)")
