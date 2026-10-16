import csv

hits = []
with open('Results/hardcode_audit.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        cls = row.get('classification', '')
        fpath = row.get('file', '')
        if cls != 'doc-reporting' or fpath.endswith(('.py', '.ipynb')):
            hits.append(row)

print(f'Total matches: {len(hits)}')
for h in hits:
    print(f"{h['file']}:{h['line']} [{h['classification']}] {h['literal']} -> {h['snippet'][:80]}")
