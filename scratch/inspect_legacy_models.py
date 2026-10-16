import os, json, sys

sys.stdout.reconfigure(encoding='utf-8')
legacy_dir = 'Notebooks/legacy'
for f in sorted(os.listdir(legacy_dir)):
    if not f.endswith('.ipynb'):
        continue
    nb_path = os.path.join(legacy_dir, f)
    with open(nb_path, 'r', encoding='utf-8') as fd:
        nb = json.load(fd)
    hits = []
    for c in nb.get('cells', []):
        s = ''.join(c.get('source', []))
        for l in s.splitlines():
            if any(k in l.lower() for k in ['camelbert', 'arabart', 'mmarco', 'load_dataset', 'from_pretrained', 'crossencoder']):
                hits.append(l.strip())
    if hits:
        print(f"=== {f} ===")
        for h in hits:
            print(f"  {h}")
