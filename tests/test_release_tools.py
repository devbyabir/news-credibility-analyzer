import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _load():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "set_github_username", ROOT / "scripts" / "set_github_username.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_username_helper_replaces_only_publishing_files(tmp_path):
    mod = _load()
    for rel in mod.TARGETS:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, tmp_path / rel)
    dry = mod.replace_username("octo-cat", root=tmp_path)
    assert dry["deploy/streamlit/model_source.json"] >= 1 and dry["README.md"] >= 2
    assert mod.PLACEHOLDER in (tmp_path / "README.md").read_text(encoding="utf-8")  # dry run wrote nothing
    mod.replace_username("octo-cat", root=tmp_path, write=True)
    src = (tmp_path / "deploy/streamlit/model_source.json").read_text(encoding="utf-8")
    assert "https://github.com/octo-cat/news-credibility-analyzer/releases/download/v1.0.0" in src
    assert all(mod.PLACEHOLDER not in (tmp_path / rel).read_text(encoding="utf-8") for rel in mod.TARGETS)


@pytest.mark.parametrize("bad", ["", "-x", "a b", "x" * 40, "a--b", "../etc"])
def test_username_helper_rejects_invalid_names(bad):
    with pytest.raises(ValueError):
        _load().replace_username(bad)
