import json
import sys
import ast

sys.stdout.reconfigure(encoding="utf-8")

notebooks = [
    "whisper-medium-qlora.ipynb",
    "whisper-large-v3-qlora.ipynb",
    "whisper-large-v3-turbo-qlora.ipynb",
    "evaluate_whisper_large_downstream.ipynb",
    "evaluate_whisper_large_turbo_downstream.ipynb",
]

for nb_name in notebooks:
    with open(nb_name, "r", encoding="utf-8") as f:
        nb = json.load(f)
    print(f"\n==================== {nb_name} ====================")
    print(f"Total cells: {len(nb['cells'])}")
    for i, c in enumerate(nb["cells"]):
        ctype = c["cell_type"]
        if ctype == "code":
            src = "".join(c["source"])
            clean = "\n".join([line for line in src.split("\n") if not line.strip().startswith("!")])
            ast.parse(clean)
            first_line = clean.strip().split("\n")[0] if clean.strip() else "EMPTY"
            print(f"  Cell {i:2d} [code]: {first_line[:65]}")
        else:
            first_line = c["source"][0].strip() if c["source"] else "EMPTY"
            print(f"  Cell {i:2d} [md]  : {first_line[:65]}")

print("\n>>> ALL CELLS VALIDATED AND COMPILED WITH ZERO SYNTAX ERRORS! <<<")
