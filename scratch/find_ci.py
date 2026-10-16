import json

with open(r'C:\Users\omarm\.gemini\antigravity-ide\brain\1db03f01-c0a8-4fe4-b528-c33ffad17da6\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if '"ci_p1":' in line or 'ci_p1' in line:
            obj = json.loads(line)
            c = obj.get('content', '')
            print(f"Match found at line {idx}: len={len(c)}")
            with open(f'scratch/match_ci_{idx}.txt', 'w', encoding='utf-8') as out:
                out.write(c)

print("Done searching.")
