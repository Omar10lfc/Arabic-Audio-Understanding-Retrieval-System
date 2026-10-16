import os, sys, time, json
import subprocess

sys.stdout.reconfigure(encoding='utf-8')

results_files = [
    'Results/cv_wer_gap_isolation_results.json',
    'Results/whisper_large_downstream_results.json',
    'Results/whisper_large_turbo_downstream_results.json',
    'Results/error_propagation_results.json',
    'Results/results-Whisper-large-v3-qlora.json',
    'Results/results-Whisper-large-v3-turbo-qlora.json',
    'Results/results-Whisper-medium-finetuned.json',
    'Results/results-Whisper-medium-stage2.json',
    'Results/summarization_results.csv',
]

print("=== RESULTS FILES MTIME AND COMMITS ===")
for rf in results_files:
    if os.path.exists(rf):
        mtime = os.path.getmtime(rf)
        mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))
        # Get last git commit affecting rf
        res = subprocess.run(['git', 'log', '-1', '--format=%h %ad %s', '--date=iso', '--', rf], capture_output=True, text=True)
        git_info = res.stdout.strip()
        print(f"File: {rf}")
        print(f"  mtime: {mtime_str}")
        print(f"  git:   {git_info}")
