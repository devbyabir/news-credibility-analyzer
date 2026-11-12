import hashlib
import io
import json

import pytest

from nca import remote

FILES = {"model.joblib": b"model-bytes", "metadata.json": b'{"ok": true}'}
SUMS = {k: hashlib.sha256(v).hexdigest() for k, v in FILES.items()}
BASE = "https://example.org/releases/download/v1"


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _fake_urlopen(served):
    calls = []

    def urlopen(url, timeout=None):
        calls.append(url)
        return _Resp(served[url.rsplit("/", 1)[1]])

    return urlopen, calls


def test_downloads_and_verifies(tmp_path, monkeypatch):
    urlopen, calls = _fake_urlopen(FILES)
    monkeypatch.setattr(remote.urllib.request, "urlopen", urlopen)
    out = remote.ensure_model(tmp_path / "m", BASE, SUMS)
    assert (out / "model.joblib").read_bytes() == FILES["model.joblib"]
    assert calls == [f"{BASE}/model.joblib", f"{BASE}/metadata.json"]
    # second call finds verified files and does not download again
    remote.ensure_model(tmp_path / "m", BASE, SUMS)
    assert len(calls) == 2


def test_checksum_mismatch_is_rejected_and_nothing_is_kept(tmp_path, monkeypatch):
    tampered = dict(FILES, **{"model.joblib": b"malicious"})
    urlopen, _ = _fake_urlopen(tampered)
    monkeypatch.setattr(remote.urllib.request, "urlopen", urlopen)
    with pytest.raises(remote.ModelFetchError, match="Checksum mismatch"):
        remote.ensure_model(tmp_path / "m", BASE, SUMS)
    assert list((tmp_path / "m").iterdir()) == []


@pytest.mark.parametrize(
    "url", ["", "https://github.com/<your-github-username>/x", "http://insecure.example"]
)
def test_unconfigured_or_insecure_urls_are_refused(tmp_path, url):
    with pytest.raises(remote.ModelFetchError):
        remote.ensure_model(tmp_path / "m", url, SUMS)


def test_committed_model_source_is_well_formed():
    from pathlib import Path

    src = remote.load_source(Path(__file__).parents[1] / "deploy" / "streamlit" / "model_source.json")
    assert set(src["sha256"]) == set(remote.MODEL_FILES)
    assert all(len(h) == 64 for h in src["sha256"].values())
    json.dumps(src)
