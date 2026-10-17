import shutil, os, json, pandas as pd

src_final = 'kaggle_outputs/final-02-tts-asr/Results/final'
dst_final = 'Results/final'
src_env = 'kaggle_outputs/final-02-tts-asr/Results/env'
dst_env = 'Results/env'

os.makedirs(dst_final, exist_ok=True)
os.makedirs(dst_env, exist_ok=True)

for f in os.listdir(src_env):
    shutil.copy2(os.path.join(src_env, f), os.path.join(dst_env, f))
    print(f'[COPIED] {f} -> {dst_env}')

for f in os.listdir(src_final):
    shutil.copy2(os.path.join(src_final, f), os.path.join(dst_final, f))
    print(f'[COPIED] {f} -> {dst_final}')

with open(os.path.join(dst_final, 'tts_manifest.json'), 'r', encoding='utf-8') as f:
    manifest = json.load(f)
print(f'TTS manifest entries: {len(manifest)}')

with open(os.path.join(dst_final, 'spoken_arcd_transcripts.json'), 'r', encoding='utf-8') as f:
    arcd = json.load(f)
print(f'Spoken ARCD passages: {len(arcd["passages"])}, queries: {len(arcd["queries"])}')

with open(os.path.join(dst_final, 'tts_asr_propagation_results.json'), 'r', encoding='utf-8') as f:
    prop = json.load(f)
print(f'Propagation models: {list(prop.keys())}, articles per model: {len(prop["large_v3"]["articles"])}')
