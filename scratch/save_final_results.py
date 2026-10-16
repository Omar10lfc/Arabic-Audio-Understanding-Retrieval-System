import json
import os
import pandas as pd

# Load content from step_8239.txt
with open('scratch/step_8239.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Locate JSON string
marker = '{\n  "retrieval":'
pos1 = text.find(marker)
assert pos1 != -1, "Marker not found!"

# Find where the JSON ends (before the next block)
pos2 = text.rfind('\n}')
assert pos2 != -1, "Ending marker not found!"

raw_json = text[pos1:pos2+2]
data = json.loads(raw_json)
print("Parsed JSON successfully! Keys:", list(data.keys()))

os.makedirs('Results/final', exist_ok=True)

# 1. downstream_retrieval_summarization.json
with open('Results/final/downstream_retrieval_summarization.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
print("Saved: Results/final/downstream_retrieval_summarization.json")

# 2. spoken_retrieval_results.json
with open('Results/final/spoken_retrieval_results.json', 'w', encoding='utf-8') as f:
    json.dump(data['retrieval'], f, indent=2, ensure_ascii=False)
print("Saved: Results/final/spoken_retrieval_results.json")

# 3. spoken_retrieval_summary.csv
csv_marker = 'Condition,Model,P@1,P@3,MRR,NDCG@3'
csv_start = text.find(csv_marker)
csv_end = text.find(marker)
csv_content = text[csv_start:csv_end].strip()
with open('Results/final/spoken_retrieval_summary.csv', 'w', encoding='utf-8') as f:
    f.write(csv_content + '\n')
print("Saved: Results/final/spoken_retrieval_summary.csv")

# 4. downstream_summarization_summary.csv
sum_rows = [
    {
        'Pipeline': 'Oracle (Clean Text -> AraBART)',
        'ASR Model': 'None (0% WER)',
        'ROUGE-1': 39.03,
        'ROUGE-2': 18.64,
        'ROUGE-L': 33.85,
        '95% CI ROUGE-L': '[30.07, 37.70]',
        'Quality Retention (%)': 100.00
    },
    {
        'Pipeline': 'Spoken (large_v3 -> AraBART)',
        'ASR Model': 'large_v3',
        'ROUGE-1': 30.84,
        'ROUGE-2': 11.21,
        'ROUGE-L': 25.35,
        '95% CI ROUGE-L': '[22.54, 28.19]',
        'Quality Retention (%)': 74.87
    },
    {
        'Pipeline': 'Spoken (turbo -> AraBART)',
        'ASR Model': 'turbo',
        'ROUGE-1': 30.35,
        'ROUGE-2': 10.85,
        'ROUGE-L': 25.32,
        '95% CI ROUGE-L': '[22.26, 28.69]',
        'Quality Retention (%)': 74.80
    },
    {
        'Pipeline': 'Spoken (medium -> AraBART)',
        'ASR Model': 'medium',
        'ROUGE-1': 29.30,
        'ROUGE-2': 9.73,
        'ROUGE-L': 24.37,
        '95% CI ROUGE-L': '[21.53, 27.20]',
        'Quality Retention (%)': 71.97
    },
    {
        'Pipeline': 'Spoken (small -> AraBART)',
        'ASR Model': 'small',
        'ROUGE-1': 26.53,
        'ROUGE-2': 8.33,
        'ROUGE-L': 22.32,
        '95% CI ROUGE-L': '[19.23, 25.54]',
        'Quality Retention (%)': 65.92
    }
]
df_sum = pd.DataFrame(sum_rows)
df_sum.to_csv('Results/final/downstream_summarization_summary.csv', index=False)
print("Saved: Results/final/downstream_summarization_summary.csv")

# 5. tts_asr_summary.csv
asr_rows = [
    {
        'Model': 'Whisper-Large-v3',
        'XL-Sum WER (%)': 12.01,
        'ARCD Passages WER (%)': 20.17,
        'Throughput (xRT)': 12.22
    },
    {
        'Model': 'Whisper-Large-v3-Turbo',
        'XL-Sum WER (%)': 12.65,
        'ARCD Passages WER (%)': 24.27,
        'Throughput (xRT)': 34.84
    },
    {
        'Model': 'Whisper-Medium',
        'XL-Sum WER (%)': 16.58,
        'ARCD Passages WER (%)': 30.30,
        'Throughput (xRT)': 19.28
    },
    {
        'Model': 'Whisper-Small',
        'XL-Sum WER (%)': 23.99,
        'ARCD Passages WER (%)': 39.66,
        'Throughput (xRT)': 36.12
    }
]
df_asr = pd.DataFrame(asr_rows)
df_asr.to_csv('Results/final/tts_asr_summary.csv', index=False)
print("Saved: Results/final/tts_asr_summary.csv")
print("[ALL SAVED SUCCESSFULLY]")
