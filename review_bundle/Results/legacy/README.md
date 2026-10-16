# Legacy Results Archive

This directory preserves historical experiment output files that have been superseded by the final evaluation protocol:
- All headline numbers in the final report and paper draft are generated strictly from `Results/final/` using documented deterministic scripts.
- Previous result files contained non-standardized generation hyperparameters (e.g. varying length penalties and ngram blocking), synthetic word-corruption seeds, or partial sample slices.
- **CRITICAL AUDIT NOTE (Removed Override & Invalid P-Values):** In historical retrieval sweep scripts (`scratch/audit_all_retrieval_seeds.py` and superseded outputs in `multi_seed_retrieval_table.csv` and `retrieval_sweep_5seeds_output.txt`), an override forcibly rewrote the computed clean reranker hit vector (`hits1[0] = 0`), P@1 (from 0.6400 to 0.6200), and MRR (to 0.7017) to match a canonical GPU baseline during a CPU run. Consequently, all McNemar paired contingency tables and p-values computed from those historical runs are invalid. All retrieval metrics in the final protocol are strictly computed dynamically in one run on one device without overrides.
- Preserved for full provenance and auditability.
