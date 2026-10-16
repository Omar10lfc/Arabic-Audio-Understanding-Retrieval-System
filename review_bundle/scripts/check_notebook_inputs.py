import os
import re
import sys
import json

def check_notebook_inputs(manifest_path="configs/notebook_manifest.json"):
    if not os.path.exists(manifest_path):
        print(f"[FAIL] Manifest not found: {manifest_path}")
        sys.exit(1)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    notebooks_manifest = manifest_data.get("notebooks", {})
    all_passed = True

    print("================================================================================")
    print("NOTEBOOK INPUTS, MOUNT PATHS, AND INTERNET DEPENDENCY CHECK")
    print("================================================================================")

    for nb_path, decl in notebooks_manifest.items():
        print(f"\n--- Checking: {nb_path} ---")
        if not os.path.exists(nb_path):
            print(f"  [ERROR] Notebook file missing: {nb_path}")
            all_passed = False
            continue

        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        internet_allowed = decl.get("internet", False)
        declared_inputs = set(decl.get("declared_inputs", []))
        declared_outputs = set(decl.get("declared_outputs", []))

        print(f"  Declared Internet Access : {'ON' if internet_allowed else 'OFF'}")
        print(f"  Declared Inputs Count    : {len(declared_inputs)}")
        print(f"  Declared Outputs Count   : {len(declared_outputs)}")

        discovered_from_pretrained = set()
        discovered_load_dataset = set()
        discovered_kaggle_paths = set()
        discovered_results_outputs = set()

        # Regex patterns
        fp_pattern = re.compile(r'from_pretrained\(\s*[\'"]([^\'"]+)[\'"]')
        ld_pattern = re.compile(r'load_dataset\(\s*[\'"]([^\'"]+)[\'"]')
        kp_pattern = re.compile(r'[\'"](/kaggle/input/[^\'"]+)[\'"]')
        ro_pattern = re.compile(r'[\'"](Results/[^\'"]+)[\'"]')
        kw_pattern = re.compile(r'[\'"](/kaggle/working/[^\'"]+)[\'"]')

        for c_idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            src = "".join(cell.get("source", []))

            for m in fp_pattern.finditer(src):
                discovered_from_pretrained.add((m.group(1), c_idx))
            for m in ld_pattern.finditer(src):
                discovered_load_dataset.add((m.group(1), c_idx))
            for m in kp_pattern.finditer(src):
                # Normalize base mounted dataset directory
                raw_path = m.group(1)
                parts = raw_path.split("/")
                if len(parts) >= 4:
                    base_dataset = "/".join(parts[:4])
                else:
                    base_dataset = raw_path
                discovered_kaggle_paths.add((base_dataset, raw_path, c_idx))
            for m in ro_pattern.finditer(src):
                discovered_results_outputs.add(m.group(1))
            for m in kw_pattern.finditer(src):
                discovered_results_outputs.add(m.group(1))

        # Check: from_pretrained targets
        for target, cell_no in discovered_from_pretrained:
            # If target is a variable reference or dynamic expression, ignore
            if target.startswith("m[") or target.startswith("{") or target.startswith("$"):
                continue

            # If internet is OFF, target MUST be a local mounted path (e.g. /kaggle/input/...)
            if not internet_allowed:
                if not (target.startswith("/kaggle/input/") or target.startswith(".") or target.startswith("models")):
                    print(f"  [VIOLATION] Cell {cell_no}: from_pretrained('{target}') requires Internet, but notebook has Internet OFF!")
                    all_passed = False
                else:
                    print(f"  [OK] Cell {cell_no}: from_pretrained('{target}') loads offline from mounted storage.")
            else:
                # Internet is ON: verify target is declared
                if target not in declared_inputs:
                    print(f"  [VIOLATION] Cell {cell_no}: Undeclared model ID '{target}' in from_pretrained!")
                    all_passed = False
                else:
                    print(f"  [OK] Cell {cell_no}: from_pretrained('{target}') matches declared online input.")

        # Check: load_dataset targets
        for target, cell_no in discovered_load_dataset:
            if not internet_allowed:
                print(f"  [VIOLATION] Cell {cell_no}: load_dataset('{target}') requires Internet, but notebook has Internet OFF!")
                all_passed = False
            elif target not in declared_inputs:
                print(f"  [VIOLATION] Cell {cell_no}: Undeclared dataset '{target}' in load_dataset!")
                all_passed = False

        # Check: /kaggle/input mount paths
        for base_mount, full_path, cell_no in discovered_kaggle_paths:
            # Check if base_mount or prefix is in declared inputs
            matched = any(full_path.startswith(d) or base_mount.startswith(d) for d in declared_inputs)
            if not matched:
                print(f"  [VIOLATION] Cell {cell_no}: Undeclared mounted path '{full_path}'!")
                all_passed = False
            else:
                print(f"  [OK] Cell {cell_no}: Mount path '{full_path}' verified.")

    print("\n================================================================================")
    if all_passed:
        print("[SUCCESS] All notebook inputs, mount paths, and offline constraints passed!")
        print("================================================================================")
        sys.exit(0)
    else:
        print("[FAILURE] Violations found in notebook dependencies!")
        print("================================================================================")
        sys.exit(1)

if __name__ == "__main__":
    check_notebook_inputs()
