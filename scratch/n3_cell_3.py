# ── CELL 4: ABSTRACTIVE SUMMARIZATION WITH DYNAMIC BATCHING ──────────────
from evalkit.rouge import score_summary, multilingual_rouge_scoring
from evalkit.summarize import summarize_batch
from evalkit.stats import bootstrap_ci
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

print(f"Loading AraBART summarizer offline from: {arabart_path}...")
bart_tok = AutoTokenizer.from_pretrained(arabart_path)
bart_mdl = AutoModelForSeq2SeqLM.from_pretrained(arabart_path).to(device).eval()

with open(asr_out_file, "r", encoding="utf-8") as f:
    xl_results = json.load(f)

print()
print("--- Generating Oracle Summaries (Clean Text) ---")
clean_texts = [a['text'] for a in xlsum_articles]
reference_summaries = [a['summary'] for a in xlsum_articles]

oracle_summaries = summarize_batch(clean_texts, bart_mdl, bart_tok, device=device)
oracle_rl = [score_summary(ref, pred)['rougeL'] for ref, pred in zip(reference_summaries, oracle_summaries)]
oracle_r1 = [score_summary(ref, pred)['rouge1'] for ref, pred in zip(reference_summaries, oracle_summaries)]
oracle_r2 = [score_summary(ref, pred)['rouge2'] for ref, pred in zip(reference_summaries, oracle_summaries)]

oracle_boot = bootstrap_ci(oracle_rl, n_bootstrap=2000, seed=42)
print(f"Oracle ROUGE-L: {np.mean(oracle_rl):.2f}% (95% CI: [{oracle_boot['ci_lower']:.2f}, {oracle_boot['ci_upper']:.2f}])")

summarization_rows = [{
    "Pipeline": "Oracle (Clean Text -> AraBART)",
    "ASR Model": "None (0% WER)",
    "ROUGE-1": round(np.mean(oracle_r1), 2),
    "ROUGE-2": round(np.mean(oracle_r2), 2),
    "ROUGE-L": round(np.mean(oracle_rl), 2),
    "95% CI ROUGE-L": f"[{oracle_boot['ci_lower']:.2f}, {oracle_boot['ci_upper']:.2f}]",
    "Quality Retention (%)": 100.0
}]

full_summarization_results = {
    "oracle": {
        "rouge1": np.mean(oracle_r1),
        "rouge2": np.mean(oracle_r2),
        "rougeL": np.mean(oracle_rl),
        "ci_lower": oracle_boot['ci_lower'],
        "ci_upper": oracle_boot['ci_upper'],
        "summaries": oracle_summaries
    },
    "models": {}
}

for m_key in ["large_v3", "turbo", "medium", "small"]:
    print()
    print(f"--- Summarizing ASR transcripts for: {m_key} ---")
    spoken_texts = [a["norm_legacy_transcript"] for a in xl_results[m_key]["articles"]]
    spoken_summaries = summarize_batch(spoken_texts, bart_mdl, bart_tok, device=device)
    
    sp_rl = [score_summary(ref, pred)['rougeL'] for ref, pred in zip(reference_summaries, spoken_summaries)]
    sp_r1 = [score_summary(ref, pred)['rouge1'] for ref, pred in zip(reference_summaries, spoken_summaries)]
    sp_r2 = [score_summary(ref, pred)['rouge2'] for ref, pred in zip(reference_summaries, spoken_summaries)]
    
    sp_boot = bootstrap_ci(sp_rl, n_bootstrap=2000, seed=42)
    retention = (np.mean(sp_rl) / np.mean(oracle_rl) * 100.0) if np.mean(oracle_rl) > 0 else 0.0
    
    summarization_rows.append({
        "Pipeline": f"Spoken ({m_key} -> AraBART)",
        "ASR Model": m_key,
        "ROUGE-1": round(np.mean(sp_r1), 2),
        "ROUGE-2": round(np.mean(sp_r2), 2),
        "ROUGE-L": round(np.mean(sp_rl), 2),
        "95% CI ROUGE-L": f"[{sp_boot['ci_lower']:.2f}, {sp_boot['ci_upper']:.2f}]",
        "Quality Retention (%)": round(retention, 2)
    })
    
    full_summarization_results["models"][m_key] = {
        "rouge1": np.mean(sp_r1),
        "rouge2": np.mean(sp_r2),
        "rougeL": np.mean(sp_rl),
        "ci_lower": sp_boot['ci_lower'],
        "ci_upper": sp_boot['ci_upper'],
        "retention_pct": retention,
        "summaries": spoken_summaries
    }
    print(f"--> {m_key}: ROUGE-L = {np.mean(sp_rl):.2f}% | Retention = {retention:.1f}%")

df_sum = pd.DataFrame(summarization_rows)
print()
print("=== SUMMARIZATION CASCADE SUMMARY ===")
print(df_sum.to_string(index=False))