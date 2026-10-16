import json

with open(r'C:\Users\omarm\.gemini\antigravity-ide\brain\1db03f01-c0a8-4fe4-b528-c33ffad17da6\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx == 8300:
            obj = json.loads(line)
            with open('scratch/line_8300.txt', 'w', encoding='utf-8') as out:
                out.write(obj.get('content', ''))
            print("Wrote line_8300.txt")
            break
