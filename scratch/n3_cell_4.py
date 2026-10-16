# ── CELL 5: RESULTS CONSOLIDATION & HASH METADATA RECORDING ──────────────
summary_out = os.path.join(OUT_DIR, "downstream_retrieval_summarization.json")
consolidated = {
    "retrieval": full_retrieval_results,
    "summarization": full_summarization_results
}
with open(summary_out, "w", encoding="utf-8") as f:
    json.dump(consolidated, f, indent=2, ensure_ascii=False)

print(f"Consolidated results saved to: {summary_out}")
print("[OK] Downstream summarization and spoken document retrieval completed successfully!")