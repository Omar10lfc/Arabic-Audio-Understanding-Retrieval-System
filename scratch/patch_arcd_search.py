with open('scratch/update_n2_n3_builders.py', 'r', encoding='utf-8') as f:
    text = f.read()

old_block = """for root, dirs, files in os.walk("/kaggle/input") if os.path.exists("/kaggle/input") else []:
    for f in sorted(files):
        if ("arcd" in root.lower() or "arcd" in f.lower()) and f.endswith(".csv"):
            raw_df = pd.read_csv(os.path.join(root, f))
            break
        elif ("arcd" in f.lower() or "arabic" in f.lower()) and f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(root, f), "r") as z:
                for zf in z.namelist():
                    if zf.endswith(".csv"):
                        raw_df = pd.read_csv(z.open(zf))
                        break
            if raw_df is not None:
                break
    if raw_df is not None:
        break"""

new_block = """for root, dirs, files in os.walk("/kaggle/input") if os.path.exists("/kaggle/input") else []:
    for f in sorted(files):
        cand_p = os.path.join(root, f)
        if f.lower() in ["train.csv", "arcd_train.csv"] or (f.endswith(".csv") and any(k in f.lower() or k in root.lower() for k in ["arcd", "comprehension"])):
            try:
                cand_df = pd.read_csv(cand_p)
                cols_low = [c.lower() for c in cand_df.columns]
                if any(c in cols_low for c in ["context", "text"]) and any(c in cols_low for c in ["question", "query"]):
                    raw_df = cand_df
                    print(f"[OK] Found ARCD dataset: {cand_p}")
                    break
            except Exception:
                pass
        elif f.endswith(".zip") and any(k in f.lower() for k in ["arcd", "comprehension", "arabic"]):
            try:
                with zipfile.ZipFile(cand_p, "r") as z:
                    for zf in z.namelist():
                        if zf.endswith(".csv") and "train" in zf.lower():
                            cand_df = pd.read_csv(z.open(zf))
                            cols_low = [c.lower() for c in cand_df.columns]
                            if any(c in cols_low for c in ["context", "text"]) and any(c in cols_low for c in ["question", "query"]):
                                raw_df = cand_df
                                print(f"[OK] Found ARCD dataset in archive: {cand_p} -> {zf}")
                                break
            except Exception:
                pass
        if raw_df is not None:
            break
    if raw_df is not None:
        break"""

count = text.count(old_block)
print(f'Occurrences of old_block: {count}')
assert count == 2, f'Expected 2 occurrences, got {count}'
updated = text.replace(old_block, new_block)
with open('scratch/update_n2_n3_builders.py', 'w', encoding='utf-8') as f:
    f.write(updated)
print('Successfully updated scratch/update_n2_n3_builders.py')
