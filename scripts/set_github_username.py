"""Replace the <your-github-username> placeholder in the files that publish links.

    python scripts/set_github_username.py YOUR_GITHUB_USERNAME            # dry run: show changes
    python scripts/set_github_username.py YOUR_GITHUB_USERNAME --write    # apply

Instruction documents (docs/DEPLOYMENT.md, docs/RELEASE_CHECKLIST.md) and tests keep
the placeholder on purpose.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = "<your-github-username>"
TARGETS = [
    "README.md",
    "deploy/streamlit/model_source.json",
    "deploy/huggingface/README.md",
    "docs/CV.md",
    "docs/FINETUNING.md",
]
# GitHub usernames: alphanumerics and single hyphens, max 39 characters.
VALID = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9]|-(?=[A-Za-z0-9])){0,38}$")


def replace_username(username: str, root: Path = ROOT, write: bool = False) -> dict[str, int]:
    if not VALID.match(username):
        raise ValueError(f"{username!r} is not a valid GitHub username")
    counts = {}
    for rel in TARGETS:
        path = root / rel
        text = path.read_text(encoding="utf-8")
        n = text.count(PLACEHOLDER)
        counts[rel] = n
        if n and write:
            path.write_text(text.replace(PLACEHOLDER, username), encoding="utf-8", newline="\n")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("username")
    ap.add_argument("--write", action="store_true", help="apply the changes (default: dry run)")
    args = ap.parse_args()
    counts = replace_username(args.username, write=args.write)
    for rel, n in counts.items():
        print(f"{'replaced' if args.write else 'would replace'} {n:>2} in {rel}")
    if not args.write:
        print("Dry run only. Re-run with --write to apply.")


if __name__ == "__main__":
    main()
