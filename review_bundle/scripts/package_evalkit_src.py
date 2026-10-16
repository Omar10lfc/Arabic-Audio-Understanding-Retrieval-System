import os
import hashlib
import json
import zipfile

def package_evalkit_src():
    zip_path = "evalkit-src.zip"
    manifest_path = "evalkit-src-manifest.json"
    
    dirs_to_package = ["evalkit", "configs", "tests"]
    file_hashes = {}
    
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for d in dirs_to_package:
            for root, _, files in os.walk(d):
                for f in sorted(files):
                    if f.endswith((".py", ".json", ".ini")) and not f.startswith("."):
                        full_path = os.path.join(root, f)
                        rel_path = os.path.relpath(full_path, ".")
                        with open(full_path, "rb") as fp:
                            content = fp.read()
                            sha = hashlib.sha256(content).hexdigest()
                        file_hashes[rel_path.replace("\\", "/")] = sha
                        z.write(full_path, rel_path)

    # Compute overall zip sha256
    with open(zip_path, "rb") as fp:
        zip_sha256 = hashlib.sha256(fp.read()).hexdigest()

    manifest = {
        "dataset_name": "evalkit-src",
        "zip_sha256": zip_sha256,
        "files": file_hashes
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[OK] Packaged evalkit-src.zip (SHA-256: {zip_sha256})")
    print(f"[OK] Manifest saved to {manifest_path} ({len(file_hashes)} files tracked)")
    return zip_sha256, manifest

if __name__ == "__main__":
    package_evalkit_src()
