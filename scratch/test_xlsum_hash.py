import tarfile, json, hashlib, os, sys
sys.stdout.reconfigure(encoding='utf-8')

articles = []
if os.path.exists('Data/arabic_XLSum_v2.0.tar.bz2'):
    with tarfile.open('Data/arabic_XLSum_v2.0.tar.bz2', 'r:bz2') as tar:
        for m in tar.getmembers():
            if 'test.jsonl' in m.name:
                print('Found member:', m.name)
                f = tar.extractfile(m)
                for line in f:
                    line_str = line.decode('utf-8').strip()
                    if line_str:
                        articles.append(json.loads(line_str))
                    if len(articles) == 50:
                        break
                break

print('Extracted articles:', len(articles))
concat_str = "".join([f"{a['id']}:{a['text']}" for a in articles])
h = hashlib.md5(concat_str.encode('utf-8')).hexdigest()
print('First 50 XL-Sum ids+texts MD5:', h)
