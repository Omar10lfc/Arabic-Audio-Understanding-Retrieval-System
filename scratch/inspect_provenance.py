import json
import os
import re
import sys
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

print("=== 2d: COMMON VOICE FILTER DIFFERENCE INSPECTION ===")
df_tsv = pd.read_csv('scratch/test.tsv', sep='\t', low_memory=False)
print(f"Total raw rows in scratch/test.tsv: {len(df_tsv)}")

# Isolation notebook logic:
# df_raw['audio_path'] = df_raw[path_col].apply(get_audio)
# df_raw['clean_ref'] = df_raw[text_col].apply(normalize_arabic)
# df_raw = df_raw.dropna(subset=['audio_path'])
# df_raw = df_raw[df_raw['clean_ref'].str.len() > 2].reset_index(drop=True)

def normalize_arabic(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)  # remove diacritics
    text = re.sub(r'[إأآا]', 'ا', text)
    text = re.sub(r'ى', 'ي', text)
    text = re.sub(r'ؤ', 'ء', text)
    text = re.sub(r'ئ', 'ء', text)
    text = re.sub(r'ة', 'ه', text)
    text = re.sub(r'[^\u0621-\u064A\s]', ' ', text)  # punctuation/special chars
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_tsv['clean_ref'] = df_tsv['sentence'].apply(normalize_arabic)

valid_nonempty = df_tsv[df_tsv['clean_ref'].str.len() > 0]
valid_gt2 = df_tsv[df_tsv['clean_ref'].str.len() > 2]

print(f"Step 1 - Total rows in test.tsv: {len(df_tsv)}")
print(f"Step 2 - Rows with clean_ref.str.len() > 0: {len(valid_nonempty)}")
print(f"Step 3 - Rows with clean_ref.str.len() > 2 (isolation notebook filter): {len(valid_gt2)}")

diff_clips = df_tsv[df_tsv['clean_ref'].str.len() <= 2]
print(f"Differing clips count: {len(diff_clips)}")
for idx, r in diff_clips.iterrows():
    print(f"  Row {idx}: path={r['path']}, sentence={repr(r['sentence'])}, clean_ref={repr(r['clean_ref'])}, len={len(r['clean_ref'])}")

print("\n=== CELL 5 OF whisper-large-v3-qlora.ipynb ===")
with open('Notebooks/whisper-large-v3-qlora.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)
cell5_src = "".join(nb['cells'][5]['source'])
print(cell5_src)

print("\n=== CELL 7 OF whisper-large-v3-qlora.ipynb ===")
cell7_src = "".join(nb['cells'][7]['source'])
print(cell7_src)


