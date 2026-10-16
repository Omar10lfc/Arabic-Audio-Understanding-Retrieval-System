import json, os, re, sys
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')

for nb_file in ['whisper-large-v3-qlora.ipynb', 'whisper-large-v3-turbo-qlora.ipynb', 'whisper-medium-qlora.ipynb', 'whisper-medium-qlora-stage2.ipynb']:
    path = os.path.join('Notebooks', nb_file)
    if not os.path.exists(path):
        continue
    print(f"==================== {nb_file} ====================")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for idx, c in enumerate(nb['cells']):
        src = "".join(c.get('source', []))
        if any(w in src for w in ['duration', '<= 30', '<=30', 'load_tsv', 'extract_features', 'n_test']):
            print(f"Cell {idx} ({c.get('cell_type')}):")
            for line in src.splitlines():
                if any(w in line for w in ['duration', '30', '29', 'load_tsv', 'n_test', 'sample', 'filter', 'len(']):
                    print("  ", line[:100])
