import json, sys
sys.stdout.reconfigure(encoding='utf-8')
nb_path = 'Notebooks/legacy/evaluate_spoken_retrieval.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx in [4, 5, 6, 7, 8]:
    src = ''.join(nb['cells'][idx].get('source', []))
    print(f"=== Cell {idx} ===")
    print(src[:1500])
