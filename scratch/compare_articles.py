import tarfile, json

with tarfile.open('Data/arabic_XLSum_v2.0.tar.bz2', 'r:bz2') as tar:
    f = tar.extractfile('./arabic_test.jsonl')
    lines = [line.decode('utf-8') for line in f]

# 1. Downstream notebook selection (unfiltered first 50)
downstream_articles = []
for idx, line in enumerate(lines[:50]):
    rec = json.loads(line)
    downstream_articles.append({
        'idx': idx,
        'id': rec.get('id'),
        'words': len(rec.get('text', '').split()),
        'summary_words': len(rec.get('summary', '').split())
    })

# 2. Error propagation selection (filtered 80 <= words <= 250 and summary >= 10)
err_prop_articles = []
for idx, line in enumerate(lines):
    rec = json.loads(line)
    doc_text = rec.get('text', rec.get('maintext', ''))
    ref_summary = rec.get('summary', '')
    wc = len(doc_text.split())
    if 80 <= wc <= 250 and len(ref_summary.split()) >= 10:
        err_prop_articles.append({
            'idx': idx,
            'id': rec.get('id'),
            'words': wc,
            'summary_words': len(ref_summary.split())
        })
        if len(err_prop_articles) >= 50:
            break

ds_ids = set(a['id'] for a in downstream_articles)
ep_ids = set(a['id'] for a in err_prop_articles)
overlap = ds_ids.intersection(ep_ids)

print(f"Downstream N=50 max line index reached: {downstream_articles[-1]['idx']}")
print(f"Error Prop N=50 max line index reached: {err_prop_articles[-1]['idx']}")
print(f"Overlap between the two 50-article sets: {len(overlap)} / 50")
print(f"IDs in Downstream but not in Error Prop: {len(ds_ids - ep_ids)}")
print("\nFirst 10 article indices for each:")
print("Downstream:", [a['idx'] for a in downstream_articles[:10]])
print("Error Prop:", [a['idx'] for a in err_prop_articles[:10]])
