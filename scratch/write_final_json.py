import json
import os

# We extract the JSON from the user's message in the transcript or reconstruct it from the exact text
# Let's inspect line 8239 or find where the JSON is stored in transcript_full.jsonl
with open(r'C:\Users\omarm\.gemini\antigravity-ide\brain\1db03f01-c0a8-4fe4-b528-c33ffad17da6\.system_generated\logs\transcript_full.jsonl', 'r', encoding='utf-8') as f:
    full_text = f.read()

# Find the json block
target_key = '"ci_p3_passage":'
idx = full_text.rfind(target_key)
assert idx != -1, "Target key not found!"

# Scan backward to find the start of the JSON object
# The user's prompt had:
# ```
# {
#   "retrieval": {
# ```
start_marker = '{\n  "retrieval": {'
json_start = full_text.rfind(start_marker, 0, idx)
if json_start == -1:
    start_marker = '{\n  "oracle": {'
    json_start = full_text.rfind(start_marker, 0, idx)

print("Found json_start at:", json_start)

# Scan forward to find the matching closing brace
depth = 0
in_string = False
escape = False
json_end = -1

for i in range(json_start, len(full_text)):
    char = full_text[i]
    if escape:
        escape = False
        continue
    if char == '\\':
        escape = True
        continue
    if char == '"':
        in_string = not in_string
        continue
    if not in_string:
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break

print("Found json_end at:", json_end)
json_str = full_text[json_start:json_end]

# Try parsing
try:
    data = json.loads(json_str)
    print("Successfully parsed JSON! Root keys:", list(data.keys()))
    
    # Save downstream_retrieval_summarization.json
    out_path = 'Results/final/downstream_retrieval_summarization.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path}")
    
    # Save spoken_retrieval_results.json
    retrieval_data = data.get('retrieval', data)
    out_path_ret = 'Results/final/spoken_retrieval_results.json'
    with open(out_path_ret, 'w', encoding='utf-8') as f:
        json.dump(retrieval_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_path_ret}")

except Exception as e:
    print("JSON parse error:", e)
    with open('scratch/failed_json.txt', 'w', encoding='utf-8') as f:
        f.write(json_str[:1000] + '\n...\n' + json_str[-1000:])
