"""Assemble a self-contained Hugging Face Space folder in build/hf_space/.

    python scripts/prepare_space.py
    # then, after `hf auth login` (see docs/DEPLOYMENT.md):
    hf repos create <your-username>/news-credibility-analyzer --repo-type space --space-sdk gradio --public
    hf upload <your-username>/news-credibility-analyzer build/hf_space . --repo-type space

The Space folder contains only what the demo needs at runtime: app.py, the
`nca` package, pinned runtime requirements, the Space card, and the trained
model artifact (model.joblib + metadata.json).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_MODULES = ["__init__.py", "labels.py", "text.py", "evaluate.py", "predict.py", "content.py"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "model")
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "hf_space")
    args = ap.parse_args()

    model = args.model_dir / "model.joblib"
    meta = args.model_dir / "metadata.json"
    if not model.exists() or not meta.exists():
        sys.exit(f"Missing {model} or {meta}. Run `python -m nca.train` first.")

    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "src" / "nca").mkdir(parents=True)
    (args.out / "artifacts" / "model").mkdir(parents=True)

    shutil.copy2(ROOT / "app.py", args.out / "app.py")
    for name in RUNTIME_MODULES:
        shutil.copy2(ROOT / "src" / "nca" / name, args.out / "src" / "nca" / name)
    shutil.copy2(ROOT / "deploy" / "huggingface" / "requirements.txt", args.out / "requirements.txt")
    shutil.copy2(ROOT / "deploy" / "huggingface" / "README.md", args.out / "README.md")
    shutil.copy2(model, args.out / "artifacts" / "model" / "model.joblib")
    shutil.copy2(meta, args.out / "artifacts" / "model" / "metadata.json")
    (args.out / ".gitattributes").write_text(
        "*.joblib filter=lfs diff=lfs merge=lfs -text\n", encoding="utf-8"
    )

    sk = json.loads(meta.read_text(encoding="utf-8")).get("environment", {}).get("scikit_learn")
    req = (args.out / "requirements.txt").read_text(encoding="utf-8")
    if sk and f"scikit-learn=={sk}" not in req:
        sys.exit(f"Model was trained with scikit-learn {sk}; update deploy/huggingface/requirements.txt.")

    total = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file()) / 1e6
    print(f"Space folder ready: {args.out} ({total:.1f} MB)")
    for f in sorted(args.out.rglob("*")):
        if f.is_file():
            print("  ", f.relative_to(args.out))


if __name__ == "__main__":
    main()
