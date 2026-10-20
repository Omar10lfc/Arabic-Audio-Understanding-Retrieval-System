"""
scripts/reproduce_controls_audit.py
Item 5 Audit: Export per-article ROUGE for Oracle, compute paired bootstrap CIs,
report Precision/Recall breakdown, and audit normalized retrieval oracle.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import pandas as pd
from evalkit.stats import paired_bootstrap_ci

def audit_controls():
    print("=" * 80)
    print("AUDIT 5.a: EXPORT PER-ARTICLE ROUGE & PAIRED BOOTSTRAP CIs")
    print("=" * 80)
    
    with open("Results/final/downstream_retrieval_summarization.json", "r", encoding="utf-8") as f:
        data = json.load(f)["summarization"]
    
    oracle_arts = data["oracle"]["per_article_rouge"]
    models = ["large_v3", "turbo", "medium", "small"]
    
    rows = []
    oracle_rL = [a["rougeL"] for a in oracle_arts]
    
    for i, o_art in enumerate(oracle_arts):
        art_id = o_art["article_id"]
        row = {
            "article_id": art_id,
            "oracle_rouge1": o_art["rouge1"],
            "oracle_rouge2": o_art["rouge2"],
            "oracle_rougeL": o_art["rougeL"],
        }
        for m in models:
            m_art = data["models"][m]["per_article_rouge"][i]
            row[f"{m}_rouge1"] = m_art["rouge1"]
            row[f"{m}_rouge2"] = m_art["rouge2"]
            row[f"{m}_rougeL"] = m_art["rougeL"]
        rows.append(row)
    
    df_per_art = pd.DataFrame(rows)
    out_csv = "Results/final/downstream_summarization_oracle_per_article.csv"
    df_per_art.to_csv(out_csv, index=False)
    print(f"Exported per-article ROUGE to: {out_csv} (N={len(df_per_art)})")
    
    print("\n--- Paired Bootstrap CIs: Spoken Model vs Oracle (ROUGE-L, N=50, B=2000, seed=42) ---")
    for m in models:
        spoken_rL = [data["models"][m]["per_article_rouge"][i]["rougeL"] for i in range(len(oracle_arts))]
        
        diff_ci = paired_bootstrap_ci(spoken_rL, oracle_rL, stat="mean_diff", n_bootstrap=2000, seed=42)
        ratio_ci = paired_bootstrap_ci(spoken_rL, oracle_rL, stat="ratio_of_means", n_bootstrap=2000, seed=42)
        
        retention = data["models"][m].get("retention_pct", np.mean(spoken_rL) / np.mean(oracle_rL) * 100)
        
        print(f"Model: {m:10s} | Mean ROUGE-L = {np.mean(spoken_rL):.2f}% vs Oracle {np.mean(oracle_rL):.2f}%")
        print(f"  Mean Difference (Spoken - Oracle): {diff_ci['mean']:.2f} pp [95% CI: {diff_ci['ci_lower']:.2f}, {diff_ci['ci_upper']:.2f}] (SE={diff_ci['se']:.2f})")
        print(f"  Ratio of Means (Retention Rate)  : {ratio_ci['mean']*100:.2f}% [95% CI: {ratio_ci['ci_lower']*100:.2f}%, {ratio_ci['ci_upper']*100:.2f}%]")
    
    print("\n" + "=" * 80)
    print("AUDIT 5.b: ORACLE CONTROLS (NORMALIZATION SENSITIVITY)")
    print("=" * 80)
    controls = data["oracle"].get("controls", {})
    for k, v in controls.items():
        print(f"  Control: {k:30s} = {v}")

if __name__ == "__main__":
    audit_controls()
