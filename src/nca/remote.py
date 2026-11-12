"""Fetch the trained model from a URL (e.g. a GitHub Release) and verify it.

Hosted demos such as Streamlit Community Cloud deploy straight from the Git
repository, but model binaries are intentionally kept out of Git. Instead the
app downloads ``model.joblib`` and ``metadata.json`` once from ``base_url`` and
checks each file's SHA-256 against a value committed in the repository.

The checksum matters: ``joblib.load`` unpickles arbitrary Python objects, so an
unverified download would let whoever controls the URL run code in the app.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

MODEL_FILES = ("model.joblib", "metadata.json")
MAX_BYTES = 100 * 1024 * 1024  # refuse anything unexpectedly large


class ModelFetchError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def load_source(path: Path) -> dict:
    """Read a model-source file: {"base_url": ..., "sha256": {filename: hex}}."""
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [f for f in MODEL_FILES if f not in source.get("sha256", {})]
    if missing:
        raise ModelFetchError(f"{path} has no checksum for {missing}")
    return source


def model_is_present(model_dir: Path, checksums: dict[str, str]) -> bool:
    return all((model_dir / f).exists() and sha256_of(model_dir / f) == checksums[f] for f in MODEL_FILES)


def ensure_model(model_dir: Path, base_url: str, checksums: dict[str, str], timeout: float = 60.0) -> Path:
    """Make sure verified model files exist in ``model_dir``; download them if not."""
    model_dir = Path(model_dir)
    if model_is_present(model_dir, checksums):
        return model_dir
    if not base_url or "<" in base_url:
        raise ModelFetchError(
            "The model is not bundled with this deployment and no download URL is configured. "
            "Set NCA_MODEL_BASE_URL or edit deploy/streamlit/model_source.json (see docs/DEPLOYMENT.md)."
        )
    if not base_url.startswith("https://"):
        raise ModelFetchError("Model downloads must use https://")
    model_dir.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        url = f"{base_url.rstrip('/')}/{name}"
        fd, tmp = tempfile.mkstemp(dir=model_dir, prefix=f".{name}.", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, timeout=timeout) as resp:
                total = 0
                while block := resp.read(1 << 20):
                    total += len(block)
                    if total > MAX_BYTES:
                        raise ModelFetchError(f"{url} is larger than {MAX_BYTES // 2**20} MB")
                    out.write(block)
            digest = sha256_of(Path(tmp))
            if digest != checksums[name]:
                raise ModelFetchError(
                    f"Checksum mismatch for {name}: expected {checksums[name][:12]}…, got {digest[:12]}…. "
                    "The release file does not match the committed model_source.json."
                )
            os.replace(tmp, model_dir / name)
        except ModelFetchError:
            raise
        except Exception as exc:
            raise ModelFetchError(f"Could not download {url}: {type(exc).__name__}: {exc}") from exc
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return model_dir
