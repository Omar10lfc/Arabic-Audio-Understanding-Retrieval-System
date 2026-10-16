with open(r'C:\Users\omarm\.gemini\antigravity-ide\brain\1db03f01-c0a8-4fe4-b528-c33ffad17da6\.system_generated\logs\transcript_full.jsonl', 'rb') as f:
    data = f.read()

pos = data.rfind(b'ci_p3_passage')
print('Found binary pos:', pos)
if pos != -1:
    print('Context around pos:')
    print(data[pos-200:pos+200].decode('utf-8', errors='ignore'))
