"""Dataset download, deterministic quality audit, de-duplication and splitting.

Usage:
    python -m nca.data download            # fetch the pinned CSV (~305 MB)
    python -m nca.data audit               # write reports/data_audit.json

Policy (documented in docs/DATA.md):
1. Rows with blank text or a label outside {0, 1} are dropped.
2. *Duplicate clusters*: rows with the same normalised text (exact
   duplicates) plus rows whose word-5-gram shingle cosine similarity is
   >= ``conflict_threshold`` (default 0.8: the same article with small edits),
   joined transitively.
3. A duplicate cluster whose rows carry *both* labels is a label conflict. We
   cannot tell which annotation is right, so every row of that cluster is
   excluded from training and evaluation (reported, never silently resolved).
4. Within a label-consistent cluster, exact duplicates collapse to the earliest
   row; distinct near-duplicates are kept.
5. *Split groups*: a looser similarity >= ``group_threshold`` (default 0.5:
   heavy shared wording, typically the same story from several outlets) links
   rows into ``group_id`` components. Train/test splitting is grouped by
   ``group_id``, so near-identical text never straddles the split. Mixed-label
   split groups are kept (a fake rewrite of a real event is a legitimate pair).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from nca.labels import AUTHENTIC, FAKE, VALID_LABELS
from nca.text import dedup_key, normalize_text, tokenize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "All60Kdataset.csv"

# Pinned source: Hugging Face mirror of BanFakeNews-2.0 (original: Mendeley Data).
HF_REPO_ID = "hrshihab/BanFakeNews-2.0"
HF_REVISION = "97a974186fc0ec52d3031a4a31ef8ca1f52192aa"
HF_FILENAME = "All60Kdataset.csv"
EXPECTED_SHA256 = "d9aed69b6019843a769bbcb045affd31c98cfa02f2b0bc821f551c54ba780c0a"


# --------------------------------------------------------------------------- IO


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def download_dataset(dest: Path = DEFAULT_RAW_PATH, force: bool = False) -> Path:
    """Download the pinned CSV revision and verify its SHA-256."""
    dest = Path(dest)
    if dest.exists() and not force:
        if sha256_file(dest) == EXPECTED_SHA256:
            print(f"Dataset already present and verified: {dest}")
            return dest
        raise RuntimeError(f"{dest} exists but its checksum differs; rerun with --force")

    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(
        repo_id=HF_REPO_ID, filename=HF_FILENAME, repo_type="dataset", revision=HF_REVISION
    )
    shutil.copyfile(cached, dest)
    digest = sha256_file(dest)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"Checksum mismatch for {dest}: {digest}")
    print(f"Downloaded and verified {dest}")
    return dest


def load_raw(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Load the CSV into columns: row_id, headline, text, label."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m nca.data download` first.")
    df = pd.read_csv(path, dtype={"Headline": "string", "Content": "string"})
    out = pd.DataFrame(
        {
            "row_id": np.arange(len(df), dtype=np.int64),
            "headline": df["Headline"],
            "text": df["Content"],
            "label": pd.to_numeric(df["Label"], errors="coerce"),
        }
    )
    return out


# ------------------------------------------------------------ near duplicates


def _shingles(text: str, n: int = 5) -> list[str]:
    toks = tokenize(text)
    if len(toks) < n:
        return [" ".join(toks)] if toks else []
    return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]


def near_duplicate_pairs(
    texts: list[str], threshold: float, chunk_size: int = 2000
) -> list[tuple[int, int, float]]:
    """Return (i, j, cosine) for i < j whose 5-gram shingle sets overlap >= threshold.

    Texts must already be normalised. Uses binary hashed shingles with L2
    normalisation, so the score is |A∩B| / sqrt(|A|·|B|).
    """
    vec = HashingVectorizer(
        analyzer=_shingles, n_features=2**24, binary=True, norm="l2", alternate_sign=False
    )
    X = vec.transform(texts).tocsr()
    XT = X.T.tocsc()
    pairs: list[tuple[int, int, float]] = []
    for start in range(0, X.shape[0], chunk_size):
        sims = (X[start : start + chunk_size] @ XT).tocoo()
        rows = sims.row + start
        mask = (sims.col > rows) & (sims.data >= threshold)
        pairs.extend(zip(rows[mask].tolist(), sims.col[mask].tolist(), sims.data[mask].tolist()))
    return pairs


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n)

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return int(root)

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


# ---------------------------------------------------------------------- audit


@dataclass
class CleanResult:
    data: pd.DataFrame  # row_id, text, label, group_id
    audit: dict
    excluded: pd.DataFrame  # row_id, reason


def audit_and_clean(
    df: pd.DataFrame, conflict_threshold: float = 0.8, group_threshold: float = 0.5
) -> CleanResult:
    """Apply the documented data policy and return clean rows plus an audit trail."""
    if group_threshold > conflict_threshold:
        raise ValueError("group_threshold must be <= conflict_threshold")
    audit: dict = {"raw_rows": int(len(df))}
    audit["raw_class_counts"] = _class_counts(df["label"])
    excluded: list[pd.DataFrame] = []

    text_str = df["text"].fillna("").astype(str)
    blank = text_str.str.strip() == ""
    bad_label = ~df["label"].isin(list(VALID_LABELS))
    audit["blank_text_rows"] = int(blank.sum())
    audit["invalid_label_rows"] = int((bad_label & ~blank).sum())
    excluded.append(_reasons(df.loc[blank, "row_id"], "blank_text"))
    excluded.append(_reasons(df.loc[bad_label & ~blank, "row_id"], "invalid_label"))

    # Raw duplicate statistics (informational; measured before any removal).
    valid = df[~blank & ~bad_label].copy()
    valid["label"] = valid["label"].astype(int)
    raw = valid["text"].astype(str)
    stripped = raw.str.strip()
    audit["raw_exact_duplicate_rows"] = int(raw.duplicated().sum())
    audit["raw_exact_duplicate_rows_label_conflict"] = _conflicting_repeat_rows(raw, valid["label"])
    audit["stripped_duplicate_rows"] = int(stripped.duplicated().sum())
    audit["stripped_duplicate_rows_label_conflict"] = _conflicting_repeat_rows(stripped, valid["label"])

    valid["norm"] = valid["text"].map(normalize_text)
    valid["key"] = valid["text"].map(dedup_key)
    no_tokens = valid["key"] == ""
    audit["no_usable_token_rows"] = int(no_tokens.sum())
    excluded.append(_reasons(valid.loc[no_tokens, "row_id"], "no_usable_tokens"))
    valid = valid[~no_tokens]
    audit["normalized_duplicate_rows"] = int(valid["key"].duplicated().sum())
    audit["normalized_duplicate_rows_label_conflict"] = _conflicting_repeat_rows(valid["key"], valid["label"])

    # Similarity between unique normalised texts, computed once at the looser threshold.
    uniq = valid.drop_duplicates("key")[["key", "norm"]].reset_index(drop=True)
    pairs = near_duplicate_pairs(uniq["norm"].tolist(), group_threshold)
    dup_uf, grp_uf = _UnionFind(len(uniq)), _UnionFind(len(uniq))
    n_dup_pairs = 0
    for i, j, sim in pairs:
        grp_uf.union(i, j)
        if sim >= conflict_threshold:
            dup_uf.union(i, j)
            n_dup_pairs += 1
    idx = {k: i for i, k in enumerate(uniq["key"])}
    pos = valid["key"].map(idx).to_numpy()
    valid["cluster"] = np.array([dup_uf.find(int(i)) for i in pos], dtype=np.int64)
    valid["group"] = np.array([grp_uf.find(int(i)) for i in pos], dtype=np.int64)
    audit["conflict_threshold"] = conflict_threshold
    audit["group_threshold"] = group_threshold
    audit["near_duplicate_pairs_at_conflict_threshold"] = n_dup_pairs
    audit["similar_pairs_at_group_threshold"] = len(pairs)

    texts_per_cluster = valid.groupby("cluster")["key"].nunique()
    audit["near_duplicate_clusters"] = int((texts_per_cluster > 1).sum())

    labels_per_cluster = valid.groupby("cluster")["label"].nunique()
    conflict_clusters = labels_per_cluster[labels_per_cluster > 1].index
    is_conflict = valid["cluster"].isin(conflict_clusters)
    audit["label_conflict_clusters"] = int(len(conflict_clusters))
    audit["label_conflict_rows_excluded"] = int(is_conflict.sum())
    audit["label_conflict_rows_excluded_by_label"] = _class_counts(valid.loc[is_conflict, "label"])
    excluded.append(_reasons(valid.loc[is_conflict, "row_id"], "label_conflict_cluster"))

    kept = valid[~is_conflict].sort_values("row_id")
    dup = kept["key"].duplicated(keep="first")
    audit["same_label_exact_duplicates_removed"] = int(dup.sum())
    excluded.append(_reasons(kept.loc[dup, "row_id"], "exact_duplicate_same_label"))
    kept = kept[~dup]

    clean = kept[["row_id", "text", "label"]].copy()
    clean["group_id"] = kept["group"].to_numpy()
    clean = clean.reset_index(drop=True)

    sizes = clean["group_id"].value_counts()
    audit["clean_rows"] = int(len(clean))
    audit["clean_class_counts"] = _class_counts(clean["label"])
    audit["split_groups"] = int(len(sizes))
    audit["split_groups_with_more_than_one_row"] = int((sizes > 1).sum())
    audit["rows_in_multi_row_split_groups"] = int(sizes[sizes > 1].sum())
    audit["largest_split_group_rows"] = int(sizes.max()) if len(sizes) else 0
    excl = pd.concat(excluded, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    audit["excluded_rows_total"] = int(len(excl))
    audit["excluded_by_reason"] = {k: int(v) for k, v in excl["reason"].value_counts().items()}
    if audit["clean_rows"] + audit["excluded_rows_total"] != audit["raw_rows"]:
        raise AssertionError("row accounting mismatch")
    return CleanResult(clean, audit, excl)


def _class_counts(labels: pd.Series) -> dict:
    s = pd.to_numeric(labels, errors="coerce")
    return {"fake (0)": int((s == FAKE).sum()), "authentic (1)": int((s == AUTHENTIC).sum())}


def _conflicting_repeat_rows(keys: pd.Series, labels: pd.Series) -> int:
    """Rows repeating an earlier key whose label differs from that key's first label."""
    first_label = labels.groupby(keys.to_numpy()).transform("first")
    return int((keys.duplicated() & (labels != first_label)).sum())


def _reasons(row_ids: pd.Series, reason: str) -> pd.DataFrame:
    return pd.DataFrame({"row_id": row_ids.to_numpy(dtype=np.int64), "reason": reason})


# ---------------------------------------------------------------------- split


def grouped_train_test_split(
    data: pd.DataFrame, test_fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Stratified, group-aware split. Returns positional (train_idx, test_idx).

    Uses one fold of ``StratifiedGroupKFold(n_splits=round(1/test_fraction))``:
    every duplicate cluster lands entirely in train or entirely in test, and the
    class ratio is approximately preserved.
    """
    n_splits = max(2, round(1 / test_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(data["text"], data["label"], groups=data["group_id"]))
    return train_idx, test_idx


def write_split_manifest(data: pd.DataFrame, test_idx: np.ndarray, path: Path) -> None:
    """Persist row_id -> split (+ text hash) so the exact split can be reproduced/checked."""
    split = np.full(len(data), "train", dtype=object)
    split[test_idx] = "test"
    manifest = pd.DataFrame(
        {
            "row_id": data["row_id"].to_numpy(),
            "group_id": data["group_id"].to_numpy(),
            "label": data["label"].to_numpy(),
            "split": split,
            "text_sha1": [hashlib.sha1(t.encode("utf-8")).hexdigest()[:12] for t in data["text"]],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False, compression="gzip")


# ------------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download the pinned dataset revision")
    d.add_argument("--dest", type=Path, default=DEFAULT_RAW_PATH)
    d.add_argument("--force", action="store_true")
    a = sub.add_parser("audit", help="run the data-quality audit")
    a.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    a.add_argument("--conflict-threshold", type=float, default=0.8)
    a.add_argument("--group-threshold", type=float, default=0.5)
    a.add_argument("--out", type=Path, default=PROJECT_ROOT / "reports" / "data_audit.json")
    args = parser.parse_args(argv)

    if args.cmd == "download":
        download_dataset(args.dest, args.force)
    else:
        result = audit_and_clean(load_raw(args.raw), args.conflict_threshold, args.group_threshold)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.audit, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(result.audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
