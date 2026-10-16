import json
import os
import pandas as pd

os.makedirs('Results/final', exist_ok=True)

# 1. spoken_retrieval_summary.csv
retrieval_csv_content = """Condition,Model,P@1,P@3,MRR,NDCG@3
Oracle (Clean -> Clean),Reference,0.64,0.78,0.7117,0.78
Set (i): Clean Q -> Spoken P,large_v3,0.6,0.8,0.7017,0.8
Set (ii): Spoken Q -> Clean P,large_v3,0.62,0.76,0.68,0.76
Set (iii): Spoken Q -> Spoken P,large_v3,0.7,0.88,0.79,0.88
Set (i): Clean Q -> Spoken P,turbo,0.62,0.82,0.733,0.82
Set (ii): Spoken Q -> Clean P,turbo,0.68,0.84,0.7617,0.84
Set (iii): Spoken Q -> Spoken P,turbo,0.64,0.88,0.7608,0.88
Set (i): Clean Q -> Spoken P,medium,0.64,0.8,0.7262,0.8
Set (ii): Spoken Q -> Clean P,medium,0.6,0.76,0.6817,0.76
Set (iii): Spoken Q -> Spoken P,medium,0.56,0.8,0.6842,0.8
Set (i): Clean Q -> Spoken P,small,0.62,0.8,0.721,0.8
Set (ii): Spoken Q -> Clean P,small,0.5,0.7,0.5967,0.7
Set (iii): Spoken Q -> Spoken P,small,0.56,0.78,0.679,0.78
"""

with open('Results/final/spoken_retrieval_summary.csv', 'w', encoding='utf-8') as f:
    f.write(retrieval_csv_content.strip() + '\n')
print("[OK] Results/final/spoken_retrieval_summary.csv written.")

# 2. downstream_summarization_summary.csv
sum_csv_content = """Pipeline,ASR Model,ROUGE-1,ROUGE-2,ROUGE-L,95% CI ROUGE-L,Quality Retention (%)
Oracle (Clean Text -> AraBART),None (0% WER),39.03,18.64,33.85,"[30.07, 37.70]",100.00
Spoken (large_v3 -> AraBART),large_v3,30.84,11.21,25.35,"[22.54, 28.19]",74.87
Spoken (turbo -> AraBART),turbo,30.35,10.85,25.32,"[22.26, 28.69]",74.80
Spoken (medium -> AraBART),medium,29.30,9.73,24.37,"[21.53, 27.20]",71.97
Spoken (small -> AraBART),small,26.53,8.33,22.32,"[19.23, 25.54]",65.92
"""

with open('Results/final/downstream_summarization_summary.csv', 'w', encoding='utf-8') as f:
    f.write(sum_csv_content.strip() + '\n')
print("[OK] Results/final/downstream_summarization_summary.csv written.")

# 3. tts_asr_summary.csv
asr_csv_content = """Model,XL-Sum WER (%),ARCD Passages WER (%),Throughput (xRT)
Whisper-Large-v3,12.01,20.17,12.22
Whisper-Large-v3-Turbo,12.65,24.27,34.84
Whisper-Medium,16.58,30.30,19.28
Whisper-Small,23.99,39.66,36.12
"""

with open('Results/final/tts_asr_summary.csv', 'w', encoding='utf-8') as f:
    f.write(asr_csv_content.strip() + '\n')
print("[OK] Results/final/tts_asr_summary.csv written.")
