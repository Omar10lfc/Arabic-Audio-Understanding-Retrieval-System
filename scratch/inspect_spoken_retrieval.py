import os, json, sys
sys.stdout.reconfigure(encoding='utf-8')

nb_path = 'Notebooks/legacy/evaluate_spoken_retrieval.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    print(f"--- Cell {idx} ({cell.get('cell_type')}) ---")
    for line in src.splitlines():
        if any(w in line.lower() for w in ['condition', 'pipeline', 'set', 'p@1', 'mrr', 'cross', 'bi-encoder', 'whisper', 'spoken', 'clean', 'oracle']):
            print('  ', line[:120])
