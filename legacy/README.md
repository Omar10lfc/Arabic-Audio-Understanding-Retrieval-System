# Legacy Notebooks Archive

This directory archives superseded experimental notebooks replaced by the unified three-stage evaluation suite:
1. `Notebooks/final_01_cv_full_pool.ipynb` (Full Common Voice test pool evaluation, PyTorch FP16 Greedy, CT2 INT8, and 4-bit unmerged training replica)
2. `Notebooks/final_02_tts_asr.ipynb` (Unified Neural TTS synthesis on XL-Sum test and ARCD corpus, followed by 4 CT2 INT8 models)
3. `Notebooks/final_03_summarization_retrieval.ipynb` (Cascading summarization and multi-set Speech-RAG retrieval benchmark under shared protocol)

### Superseded Files:
- `audit_cv_sampling_isolation.ipynb`: Early exploratory notebook for Common Voice slice composition. Superseded by `final_01_cv_full_pool.ipynb`.
- `evaluate_whisper_large_downstream.ipynb` / `evaluate_whisper_large_turbo_downstream.ipynb`: Previous downstream evaluation notebooks containing disparate generation configs and synthetic tie-breakers. Superseded by `final_02_tts_asr.ipynb` and `final_03_summarization_retrieval.ipynb`.
- `evaluate_error_propagation.ipynb`: Pre-unification cascading error evaluation. Superseded by `final_03_summarization_retrieval.ipynb`.
- `benchmark_whisper_large.ipynb` / `benchmark_quantization.ipynb`: Preliminary local/GPU benchmarks superseded by the final protocol.
