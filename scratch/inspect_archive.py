import tarfile, json

with tarfile.open('Data/arabic_XLSum_v2.0.tar.bz2', 'r:bz2') as tar:
    for m in tar.getmembers():
        print(m.name, m.size)
        if 'test.jsonl' in m.name:
            f = tar.extractfile(m)
            lines = [json.loads(f.readline()) for _ in range(5)]
            for i, line in enumerate(lines):
                text = line.get('text', '')
                summary = line.get('summary', '')
                print(f"Article {i}: words={len(text.split())}, summary_words={len(summary.split())}, id={line.get('id')}")
