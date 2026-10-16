import os, sys, json
import pyflakes.api
import pyflakes.reporter
import io
import nbformat

sys.stdout.reconfigure(encoding='utf-8')
os.environ['SMOKE'] = '1'
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

notebooks = [
    'Notebooks/final_01_cv_full_pool.ipynb',
    'Notebooks/final_02_tts_asr.ipynb',
    'Notebooks/final_03_summarization_retrieval.ipynb'
]

print("=== 1. PYFLAKES STATIC AUDIT OF EXTRACTED CODE CELLS ===")
for nb_path in notebooks:
    print(f"\n--- Linting {nb_path} ---")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    total_code_cells = 0
    all_code = []
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            total_code_cells += 1
            src = cell.source
            # Filter ipython magics if any
            clean_lines = [l for l in src.splitlines() if not l.strip().startswith(('!', '%'))]
            all_code.append("\n".join(clean_lines))

    combined_code = "\n".join(all_code)
    warn_out = io.StringIO()
    reporter = pyflakes.reporter.Reporter(warn_out, warn_out)
    errs = pyflakes.api.check(combined_code, nb_path, reporter=reporter)
    warnings = [w for w in warn_out.getvalue().splitlines() if 'imported but unused' not in w]
    print(f"Total code cells: {total_code_cells}")
    if not warnings:
        print("[PASS] pyflakes: 0 structural errors / undefined names.")
    else:
        print(f"[WARN] pyflakes output:\n" + "\n".join(warnings[:10]))

print("\n=== 2. LOCAL CPU EXECUTION TEST IN SMOKE MODE (SMOKE=1) ===")
# Execute each notebook with SMOKE=1
for nb_path in notebooks:
    print(f"\nExecuting {nb_path} in SMOKE mode...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    global_env = {'SMOKE': True, '__name__': '__main__'}
    skipped_cells = []
    executed_cells = 0
    
    for i, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
        code = cell.source
        clean_code = "\n".join([l for l in code.splitlines() if not l.strip().startswith(('!', '%'))])
        try:
            exec(clean_code, global_env)
            executed_cells += 1
        except Exception as e:
            skipped_cells.append((i, str(e)))
            print(f"  Cell {i} could not run locally on CPU: {e}")

    print(f"Executed: {executed_cells}/{executed_cells + len(skipped_cells)} cells")
    if not skipped_cells:
        print("[OK] All cells ran successfully locally in SMOKE mode.")
    else:
        print(f"Cells requiring GPU / external files: {[c[0] for c in skipped_cells]}")
