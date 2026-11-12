"""Write the SHA-256 of the trained model files into deploy/streamlit/model_source.json.

python scripts/model_checksums.py            # after retraining
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nca.remote import MODEL_FILES, sha256_of  # noqa: E402

model_dir = ROOT / "artifacts" / "model"
source_path = ROOT / "deploy" / "streamlit" / "model_source.json"
source = json.loads(source_path.read_text(encoding="utf-8"))
source["sha256"] = {name: sha256_of(model_dir / name) for name in MODEL_FILES}
source_path.write_text(json.dumps(source, indent=2) + "\n", encoding="utf-8", newline="\n")
for name, digest in source["sha256"].items():
    print(f"{name}: {digest}")
