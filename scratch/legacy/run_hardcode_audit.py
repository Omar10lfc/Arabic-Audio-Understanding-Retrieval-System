import os, sys, re, json
sys.stdout.reconfigure(encoding='utf-8')

targets = [
    'in locals() else',
    '0.6200', '0.7017', '0.5855', '17.63', '17.84',
    '12.51', '13.03', '42.69', '20.61', '29.56', '13.48', '33.85'
]

exts = ('.py', '.ipynb', '.md', '.json')

matches = []

# Exclude .git
for root, dirs, files in os.walk('.'):
    if '.git' in root or '__pycache__' in root:
        continue
    for f in files:
        if f.endswith(exts):
            fpath = os.path.join(root, f)
            relpath = os.path.normpath(fpath)
            try:
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                    for line_num, line in enumerate(fp, 1):
                        for tgt in targets:
                            if tgt in line:
                                matches.append((relpath, line_num, tgt, line.strip()))
            except Exception as e:
                pass

print(f"Total matches found: {len(matches)}")
# Let's save all matches to a file for analysis
with open('scratch/hardcode_audit_matches.json', 'w', encoding='utf-8') as out:
    json.dump([{'file': m[0], 'line': m[1], 'target': m[2], 'snippet': m[3][:120]} for m in matches], out, indent=2)

# Print summary by file
from collections import defaultdict
by_file = defaultdict(list)
for m in matches:
    by_file[m[0]].append(m)

for f, lst in sorted(by_file.items()):
    print(f"{f}: {len(lst)} matches (targets: {set(m[2] for m in lst)})")
