"""
Review Bundle Builder (scripts/review_bundle_builder.py)
Copies audited artifacts into review_bundle/, generates MANIFEST.json,
and creates review_bundle.zip (excluding model weights).
"""

import os
import sys
import shutil
import json
import hashlib
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
BUNDLE_DIR = ROOT_DIR / "review_bundle"

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def copy_tree_filtered(src_dir, dst_dir, exclude_exts=None):
    if exclude_exts is None:
        exclude_exts = {".safetensors", ".bin", ".pt", ".pth", ".pyc"}
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    if not src_path.exists():
        return
    for item in src_path.glob("**/*"):
        if item.is_file():
            if item.suffix in exclude_exts or "__pycache__" in item.parts:
                continue
            rel_p = item.relative_to(src_path)
            target = dst_path / rel_p
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)

def copy_file(src_file, dst_file):
    src = Path(src_file)
    dst = Path(dst_file)
    if src.exists() and src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

def build_bundle():
    print("Populating review_bundle/ directory...")
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Notebooks
    copy_tree_filtered(ROOT_DIR / "Notebooks", BUNDLE_DIR / "Notebooks")
    
    # 2. evalkit/
    copy_tree_filtered(ROOT_DIR / "evalkit", BUNDLE_DIR / "evalkit")
    
    # 3. configs/
    if (ROOT_DIR / "configs").exists():
        copy_tree_filtered(ROOT_DIR / "configs", BUNDLE_DIR / "configs")
        
    # 4. tests/
    copy_tree_filtered(ROOT_DIR / "tests", BUNDLE_DIR / "tests")
    
    # 5. scripts/
    copy_tree_filtered(ROOT_DIR / "scripts", BUNDLE_DIR / "scripts")
    
    # 6. Requirements & env snapshots
    copy_file(ROOT_DIR / "requirements.txt", BUNDLE_DIR / "requirements.txt")
    copy_file(ROOT_DIR / "packages.txt", BUNDLE_DIR / "packages.txt")
    copy_file(ROOT_DIR / "Results" / "pip_freeze.txt", BUNDLE_DIR / "Results" / "pip_freeze.txt")
    if (ROOT_DIR / "Results" / "env").exists():
        copy_tree_filtered(ROOT_DIR / "Results" / "env", BUNDLE_DIR / "Results" / "env")
        
    # 7. Results/final/**
    copy_tree_filtered(ROOT_DIR / "Results" / "final", BUNDLE_DIR / "Results" / "final")
    
    # 8. Results/legacy/README.md
    copy_file(ROOT_DIR / "Results" / "legacy" / "README.md", BUNDLE_DIR / "Results" / "legacy" / "README.md")
    
    # 9. Docs
    copy_file(ROOT_DIR / "README.md", BUNDLE_DIR / "README.md")
    copy_file(ROOT_DIR / "README_SPACE.md", BUNDLE_DIR / "README_SPACE.md")
    copy_file(ROOT_DIR / "REPORT_experiments_results.md", BUNDLE_DIR / "REPORT_experiments_results.md")
    copy_file(ROOT_DIR / "LICENSE", BUNDLE_DIR / "LICENSE")
    if (ROOT_DIR / "assets").exists():
        copy_tree_filtered(ROOT_DIR / "assets", BUNDLE_DIR / "assets")

    print("Files successfully staged in review_bundle/.")

def generate_manifest():
    print("Generating MANIFEST.json for review_bundle/...")
    manifest = []
    for item in sorted(BUNDLE_DIR.glob("**/*")):
        if item.is_file() and item.name != "MANIFEST.json" and not item.name.endswith(".zip"):
            stat = item.stat()
            rel_path = item.relative_to(BUNDLE_DIR).as_posix()
            sha256_val = compute_sha256(item)
            manifest.append({
                "path": rel_path,
                "sha256": sha256_val,
                "bytes": stat.st_size,
                "mtime": stat.st_mtime
            })
            
    manifest_path = BUNDLE_DIR / "MANIFEST.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"MANIFEST.json written with {len(manifest)} files.")

def create_zip():
    zip_path = ROOT_DIR / "review_bundle.zip"
    print(f"Creating {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in sorted(BUNDLE_DIR.glob("**/*")):
            if item.is_file():
                rel_path = item.relative_to(BUNDLE_DIR).as_posix()
                zf.write(item, arcname=f"review_bundle/{rel_path}")
    print(f"Successfully created {zip_path.name} ({zip_path.stat().st_size} bytes).")

if __name__ == "__main__":
    build_bundle()
    generate_manifest()
    create_zip()
