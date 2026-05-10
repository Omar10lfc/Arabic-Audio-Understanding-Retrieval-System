"""
push_whisper_to_hub.py — upload the fine-tuned Whisper model to the HF Hub.

One-time prep:
    pip install huggingface_hub
    huggingface-cli login   # paste a write token from huggingface.co/settings/tokens

Then:
    python push_whisper_to_hub.py --user your-username

What this does:
    1. Creates (or reuses) the repo  https://huggingface.co/{user}/{repo}
    2. Uploads the entire `Whisper-Fine-tuned-final-model/` folder (weights,
       processor, tokenizer, generation_config, README.md model card)
    3. Prints the public model URL
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


PROJECT_DIR  = Path(__file__).resolve().parent
DEFAULT_DIR  = PROJECT_DIR / "Whisper-Fine-tuned-final-model"
DEFAULT_REPO = "whisper-small-arabic"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", required=True, help="HF username or org")
    p.add_argument("--repo", default=DEFAULT_REPO,
                   help=f"Repo name on HF (default: {DEFAULT_REPO})")
    p.add_argument("--folder", default=str(DEFAULT_DIR),
                   help="Local folder to upload (default: ./Whisper-Fine-tuned-final-model)")
    p.add_argument("--private", action="store_true",
                   help="Create the repo as private")
    p.add_argument("--commit", default="Initial upload: whisper-small-arabic",
                   help="Commit message")
    args = p.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        sys.exit(f"❌ Folder not found: {folder}")

    required = ["config.json", "model.safetensors", "tokenizer.json",
                "generation_config.json"]
    missing = [f for f in required if not (folder / f).exists()]
    if missing:
        sys.exit(f"❌ Missing required files in {folder}: {missing}")
    if not (folder / "README.md").exists():
        print(f"⚠️  No README.md in {folder} — uploading without a model card.")

    repo_id = f"{args.user}/{args.repo}"
    print(f"→ Creating / verifying repo {repo_id} (private={args.private}) ...")
    create_repo(repo_id, private=args.private, exist_ok=True, repo_type="model")

    api = HfApi()
    print(f"→ Uploading {folder}  →  {repo_id}")
    api.upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type="model",
        commit_message=args.commit,
        # The training_args.bin file is huge and not needed for inference. Keep
        # everything else — including the README model card.
        ignore_patterns=["training_args.bin", "*.pyc", "__pycache__/*"],
    )

    print(f"\n✅ Done. https://huggingface.co/{repo_id}")
    print(f"   Use it from code:  AutoModelForSeq2SeqLM.from_pretrained('{repo_id}')")


if __name__ == "__main__":
    main()
