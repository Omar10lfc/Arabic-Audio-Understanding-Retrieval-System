import json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('scratch/hardcode_audit_matches.json', 'r', encoding='utf-8') as f:
    matches = json.load(f)

# Filter out .venv and external and tokenizer.json and vocabulary.json
relevant = []
for m in matches:
    p = m['file'].replace('\\', '/')
    if any(k in p for k in ['.venv/', 'external/', 'tokenizer.json', 'vocabulary.json', 'scratch/run_hardcode_audit.py', 'scratch/classify_matches.py']):
        continue
    relevant.append(m)

print(f"Total relevant matches: {len(relevant)}")

# Group by file
from collections import defaultdict
by_f = defaultdict(list)
for m in relevant:
    by_f[m['file']].append(m)

for f, lst in sorted(by_f.items()):
    print(f"\n=== {f} ({len(lst)} matches) ===")
    for m in lst:
        print(f"  Line {m['line']:4d} [{m['target']}]: {m['snippet']}")
