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
    ph = mod.PLACEHOLDER
    for rel in mod.TARGETS:  # synthetic copies, independent of whether the repo is already filled in
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(
            f"see https://github.com/{ph}/news-credibility-analyzer", encoding="utf-8"
        )
    (tmp_path / "docs" / "DEPLOYMENT.md").write_text(f"instructions keep {ph}", encoding="utf-8")
    dry = mod.replace_username("octo-cat", root=tmp_path)
    assert all(n == 1 for n in dry.values())
    assert ph in (tmp_path / "README.md").read_text(encoding="utf-8")  # dry run wrote nothing
    mod.replace_username("octo-cat", root=tmp_path, write=True)
    for rel in mod.TARGETS:
        assert "github.com/octo-cat/news-credibility-analyzer" in (tmp_path / rel).read_text(encoding="utf-8")
    assert ph in (tmp_path / "docs" / "DEPLOYMENT.md").read_text(encoding="utf-8")  # not a target


@pytest.mark.parametrize("bad", ["", "-x", "a b", "x" * 40, "a--b", "../etc"])
def test_username_helper_rejects_invalid_names(bad):
    with pytest.raises(ValueError):
        _load().replace_username(bad)
