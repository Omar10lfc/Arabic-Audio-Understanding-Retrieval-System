# ── CELL 5: ARTIFACT VALIDATION & DUAL WER CHECKPOINT ─────────────────────
import pandas as pd

print("=== N2 MULTI-MODEL ASR SUMMARY ===")
df_summary = pd.DataFrame(summary_records)
print(df_summary.to_string(index=False))

df_summary.to_csv(os.path.join(OUT_DIR, "tts_asr_summary.csv"), index=False)
print()
print("Artifacts verified:")
print(f" - {asr_out_file} ({os.path.getsize(asr_out_file) / 1024:.1f} KB)")
print(f" - {arcd_trans_file} ({os.path.getsize(arcd_trans_file) / 1024:.1f} KB)")
print(f" - {manifest_path} ({os.path.getsize(manifest_path) / 1024:.1f} KB)")
print()
print("[OK] Neural TTS synthesis and multi-model ASR transcription complete.")