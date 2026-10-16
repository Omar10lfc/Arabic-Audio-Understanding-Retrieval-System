import os, shutil, sys

sys.stdout.reconfigure(encoding='utf-8')

os.makedirs('legacy', exist_ok=True)
os.makedirs('Results/legacy', exist_ok=True)

# 1. Superseded results in Results/
results_to_move = [
    'cv_wer_gap_isolation_results.json',
    'cv_wer_gap_isolation_summary.csv',
    'error_propagation_results.json',
    'error_propagation_results.csv',
    'whisper_large_downstream_results.json',
    'whisper_large_turbo_downstream_results.json',
    'whisper_large_benchmark_results.json',
    'whisper_large_benchmark_summary.csv',
    'spoken_retrieval_results.json',
    'spoken_retrieval_summary.csv',
    'multi_seed_retrieval_table.csv',
    'retrieval_sweep_5seeds_output.txt',
    'quantization_benchmark_results.json',
    'quantization_benchmark_results.csv',
    'noise_sweep_results.json',
    'noise_sweep_summary.csv',
    'chunking_reranking_results.csv',
    'entity_error_breakdown.json',
    'entity_error_breakdown.csv',
]

for fname in results_to_move:
    src = os.path.join('Results', fname)
    dst = os.path.join('Results/legacy', fname)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {src} -> {dst}")

# 2. Superseded notebooks from root to legacy/
root_nbs_to_move = [
    'audit_cv_sampling_isolation.ipynb',
    'benchmark_whisper_large.ipynb',
    'benchmark_quantization.ipynb',
    'evaluate_whisper_large_downstream.ipynb',
    'evaluate_whisper_large_turbo_downstream.ipynb',
    'evaluate_error_propagation.ipynb',
    'evaluate_noise_sweep.ipynb',
    'whisper-large-v3-qlora.ipynb',
    'whisper-large-v3-turbo-qlora.ipynb',
    'whisper-medium-qlora.ipynb',
    'whisper-medium-qlora-stage2.ipynb',
]

for fname in root_nbs_to_move:
    src = fname
    dst = os.path.join('legacy', fname)
    if os.path.exists(src):
        shutil.move(src, dst)
        print(f"Moved {src} -> {dst}")

# 3. Create README in legacy/
legacy_readme = """# Legacy Notebooks Archive

This directory archives superseded experimental notebooks replaced by the unified three-stage evaluation suite:
1. `Notebooks/final_01_cv_full_pool.ipynb` (Full Common Voice test pool evaluation, PyTorch FP16 Greedy, CT2 INT8, and 4-bit unmerged training replica)
2. `Notebooks/final_02_tts_asr.ipynb` (Unified Neural TTS synthesis on XL-Sum test and ARCD corpus, followed by 4 CT2 INT8 models)
3. `Notebooks/final_03_summarization_retrieval.ipynb` (Cascading summarization and multi-set Speech-RAG retrieval benchmark under shared protocol)

### Superseded Files:
- `audit_cv_sampling_isolation.ipynb`: Early exploratory notebook for Common Voice slice composition. Superseded by `final_01_cv_full_pool.ipynb`.
- `evaluate_whisper_large_downstream.ipynb` / `evaluate_whisper_large_turbo_downstream.ipynb`: Previous downstream evaluation notebooks containing disparate generation configs and synthetic tie-breakers. Superseded by `final_02_tts_asr.ipynb` and `final_03_summarization_retrieval.ipynb`.
- `evaluate_error_propagation.ipynb`: Pre-unification cascading error evaluation. Superseded by `final_03_summarization_retrieval.ipynb`.
- `benchmark_whisper_large.ipynb` / `benchmark_quantization.ipynb`: Preliminary local/GPU benchmarks superseded by the final protocol.
"""
with open('legacy/README.md', 'w', encoding='utf-8') as f:
    f.write(legacy_readme)

# 4. Create README in Results/legacy/
results_legacy_readme = """# Legacy Results Archive

This directory preserves historical experiment output files that have been superseded by the final evaluation protocol:
- All headline numbers in the final report and paper draft are generated strictly from `Results/final/` using documented deterministic scripts.
- Previous result files contained non-standardized generation hyperparameters (e.g. varying length penalties and ngram blocking), synthetic word-corruption seeds, or partial sample slices.
- Preserved for full provenance and auditability.
"""
with open('Results/legacy/README.md', 'w', encoding='utf-8') as f:
    f.write(results_legacy_readme)

print("[OK] Superseded files archived and READMEs created successfully.")
