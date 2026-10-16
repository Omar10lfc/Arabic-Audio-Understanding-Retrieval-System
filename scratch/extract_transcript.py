import json

with open(r'C:\Users\omarm\.gemini\antigravity-ide\brain\1db03f01-c0a8-4fe4-b528-c33ffad17da6\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if 'Condition,Model,P@1' in line or 'retrieval' in line:
            obj = json.loads(line)
            content = obj.get('content', '')
            if 'Condition,Model,P@1' in content:
                print(f"Match found at line {idx}: len={len(content)}")
                with open('scratch/found_match.txt', 'w', encoding='utf-8') as out:
                    out.write(content)
print("Search complete.")
