import json
import os

def generate_report_tables():
    json_path = os.path.join("Results", "cv_wer_gap_isolation_results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sub = data["subsets"]
    attr = data["attribution_pp"]
    multi = data.get("multi_draw_n100", {})

    # 1. Main Isolation Table
    wer_a = sub["subset_a_400_rand_wer"]
    wer_b = sub["subset_b_400_seq_wer"]
    wer_c = sub["subset_c_100_rand_wer"]
    wer_d = sub["subset_d_100_seq_wer"]
    wer_e = sub["subset_e_ct2_int8_wer"]

    table_isolation = f"""| Evaluation Condition | Sample Count | Sampling Method | Decoding Engine | Empirical WER (%) | Delta vs. Baseline |
|---|---:|---|---|---:|---:|
| **A: Training Protocol Replica** | 400 | Random ($seed=42$) | PyTorch FP16 Greedy | **{wer_a:.2f}%** | 0.00 pp (Ref) |
| **B: Method Isolation** | 400 | First $N$ Sequential | PyTorch FP16 Greedy | **{wer_b:.2f}%** | {wer_b - wer_a:+.2f} pp |
| **C: Sample Count Isolation** | 100 | Random ($seed=42$) | PyTorch FP16 Greedy | **{wer_c:.2f}%** | {wer_c - wer_a:+.2f} pp |
| **D: Downstream Replica** | 100 | First $N$ Sequential | PyTorch FP16 Greedy | **{wer_d:.2f}%** | {wer_d - wer_a:+.2f} pp |
| **E: Downstream Production** | 100 | First $N$ Sequential | CT2 Static INT8 | **{wer_e:.2f}%** | {wer_e - wer_a:+.2f} pp |"""

    # 2. Multi-draw table
    draw_rows = []
    for idx, d in enumerate(multi.get("draws", []), 1):
        draw_rows.append(f"| **Draw {idx}** | {d['seed']} | 100 | {d['wer']:.2f}% | {d['delta_vs_a']:+.2f} pp |")
    draws_block = "\n".join(draw_rows)

    obs_c = multi.get("observed_c_delta", wer_c - wer_a)
    mean_wer = multi.get("mean_wer", 13.26)
    std_wer = multi.get("std_wer", 0.62)

    table_multidraw = f"""| Evaluation Draw | Random Seed | Clips ($N$) | Empirical WER (%) | Delta vs. 400-Clip Baseline ($A = {wer_a:.2f}\\%$) |
|---|:---:|:---:|:---:|:---:|
{draws_block}
| **Observed Subset C** | 42 | 100 | {wer_c:.2f}% | {obs_c:+.2f} pp |
| **5-Seed Summary** | — | 100 | **{mean_wer:.2f}% ± {std_wer:.2f}%** | **+{mean_wer - wer_a:.2f} pp** (Mean) |"""

    print("=== TABLE 1: EMPIRICAL WER GAP ISOLATION ===")
    print(table_isolation)
    print("\n=== TABLE 2: MULTI-DRAW STABILITY ===")
    print(table_multidraw)
    return table_isolation, table_multidraw

if __name__ == "__main__":
    generate_report_tables()
