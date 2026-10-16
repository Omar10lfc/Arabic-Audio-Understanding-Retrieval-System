import os, re, csv, sys

sys.stdout.reconfigure(encoding='utf-8')

targets = [
    '0.6200', '0.7017', '0.5855', '17.63', '17.84',
    '12.51', '13.03', '42.69', '20.61', '29.56', '13.48', '33.85'
]

rows = []

for root, dirs, files in os.walk('.'):
    if any(k in root for k in ['.git', '.venv']):
        continue
    for f in sorted(files):
        if f.endswith(('.py', '.ipynb', '.md', '.json', '.csv', '.txt')) and f != 'hardcode_audit.csv':
            p = os.path.normpath(os.path.join(root, f)).replace('\\', '/')
            try:
                with open(p, 'r', encoding='utf-8', errors='ignore') as fp:
                    for line_num, line in enumerate(fp, 1):
                        for tgt in targets:
                            if tgt in line:
                                snippet = line.strip().replace('"', "'")[:100]
                                # Classify
                                if 'legacy' in p:
                                    cls = 'superseded-legacy'
                                elif p.endswith(('.md', '.txt')):
                                    cls = 'doc-reporting'
                                elif 'Results/' in p:
                                    cls = 'read-from-file'
                                elif p.startswith(('Notebooks/', 'evalkit/', 'scripts/')):
                                    if 'test' in p:
                                        cls = 'test-verification'
                                    else:
                                        cls = 'computed'
                                else:
                                    cls = 'computed'
                                rows.append({
                                    'file': p,
                                    'line': line_num,
                                    'literal': tgt,
                                    'classification': cls,
                                    'snippet': snippet
                                })
            except Exception:
                pass

os.makedirs('Results', exist_ok=True)
csv_path = 'Results/hardcode_audit.csv'
with open(csv_path, 'w', encoding='utf-8', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['file', 'line', 'literal', 'classification', 'snippet'])
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {csv_path} with {len(rows)} entries.")
