with open(r'C:\Users\omarm\.gemini\antigravity-ide\brain\1db03f01-c0a8-4fe4-b528-c33ffad17da6\.system_generated\logs\transcript_full.jsonl', 'rb') as f:
    data = f.read()

pos = 0
matches = []
while True:
    p = data.find(b'ci_p3_passage', pos)
    if p == -1:
        break
    matches.append(p)
    pos = p + 1

print(f"Total matches found: {len(matches)}")
for p in matches[-10:]:
    print(f"Pos {p}:")
    print(data[p-100:p+100].decode('utf-8', errors='ignore'))
    print("="*40)
