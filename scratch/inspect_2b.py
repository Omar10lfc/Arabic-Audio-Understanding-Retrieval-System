import json, os, re, sys
sys.stdout.reconfigure(encoding='utf-8')

files = [
    'Notebooks/evaluate_whisper_large_downstream.ipynb',
    'Notebooks/evaluate_whisper_large_turbo_downstream.ipynb',
    'Notebooks/evaluate_error_propagation.ipynb',
    'Notebooks/summarization.ipynb',
    'pipeline.py'
]

for fp in files:
    print(f"\n==================== {fp} ====================")
    if not os.path.exists(fp):
        print("FILE NOT FOUND")
        continue
    if fp.endswith('.py'):
        with open(fp, 'r', encoding='utf-8') as f:
            content = f.read()
        for line in content.splitlines():
            if any(k in line for k in ['generate(', 'num_beams', 'max_length', 'min_length', 'length_penalty', 'no_repeat_ngram_size', 'early_stopping']):
                print(line)
    elif fp.endswith('.ipynb'):
        with open(fp, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        for i, cell in enumerate(nb['cells']):
            src = "".join(cell.get('source', []))
            if any(k in src for k in ['generate(', 'summarizer', 'pipeline', 'min_length', 'max_summary_length', 'length_penalty']):
                lines = [l for l in src.splitlines() if any(k in l for k in ['generate(', 'min_length', 'max_length', 'max_summary_length', 'num_beams', 'length_penalty', 'no_repeat_ngram_size', 'early_stopping', 'padding'])]
                if lines:
                    print(f"Cell {i} ({cell.get('cell_type')}):")
                    for l in lines:
                        print("  ", l)
