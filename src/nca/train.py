"""Leakage-free training and evaluation.

    python -m nca.train --config configs/baseline.json

Protocol
1. Audit + de-duplicate the raw CSV (nca.data.audit_and_clean).
2. Hold out ~20 % of duplicate *groups* as the test set (stratified, grouped,
   fixed seed) before any vectorizer or model is fit.
3. Model selection: grouped stratified K-fold CV on the training portion only.
   Each fold fits its own TF-IDF vocabulary on the fold's training rows.
4. The selected configuration (highest mean CV macro-F1 among models that
   output probabilities) is refit on the full training portion and evaluated
   once on the test set. The test set is never used for any choice.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from nca import __version__
from nca.data import (
    DEFAULT_RAW_PATH,
    EXPECTED_SHA256,
    HF_REPO_ID,
    HF_REVISION,
    PROJECT_ROOT,
    audit_and_clean,
    grouped_train_test_split,
    load_raw,
    write_split_manifest,
)
from nca.evaluate import band_stats, calibration_summary, classification_metrics, save_figures
from nca.labels import AUTHENTIC, FAKE, LABEL_NAMES
from nca.models import PROBABILISTIC_MODELS, make_classifier, make_pipeline, make_vectorizer


def expand_grid(params: dict) -> list[dict]:
    if not params:
        return [{}]
    keys = sorted(params)
    values = [v if isinstance(v, list) else [v] for v in (params[k] for k in keys)]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def fake_scores(clf, X) -> tuple[np.ndarray, bool]:
    """Return a score increasing with P(fake) and whether it is a probability."""
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        classes = list(clf.classes_)
        if FAKE in classes:
            return proba[:, classes.index(FAKE)], True
        return np.zeros(proba.shape[0]), True
    return -clf.decision_function(X), False  # decision > 0 means authentic (label 1)


def run_cv(train: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    seed, n_splits = cfg["seed"], cfg["cv_folds"]
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    texts, y, groups = train["text"].to_numpy(), train["label"].to_numpy(), train["group_id"].to_numpy()
    rows = []
    for fold, (tr, va) in enumerate(sgkf.split(texts, y, groups), start=1):
        by_features: dict[str | None, list[dict]] = {}
        for cand in cfg["candidates"]:
            by_features.setdefault(cand.get("features"), []).append(cand)
        for feat_name, cands in by_features.items():
            t0 = time.perf_counter()
            if feat_name is None:
                Xtr, Xva = np.zeros((len(tr), 1)), np.zeros((len(va), 1))
            else:
                vec = make_vectorizer(cfg["feature_sets"][feat_name])
                Xtr = vec.fit_transform(texts[tr])
                Xva = vec.transform(texts[va])
            vec_s = time.perf_counter() - t0
            for cand in cands:
                for params in expand_grid(cand.get("params", {})):
                    t1 = time.perf_counter()
                    clf = make_classifier(cand["model"], seed, **params).fit(Xtr, y[tr])
                    pred = clf.predict(Xva)
                    score, _ = fake_scores(clf, Xva)
                    m = classification_metrics(y[va], pred, score)
                    rows.append(
                        {
                            "candidate": cand["name"],
                            "params": json.dumps(params, sort_keys=True),
                            "fold": fold,
                            "macro_f1": m["macro_f1"],
                            "accuracy": m["accuracy"],
                            "fake_precision": m["per_class"]["fake"]["precision"],
                            "fake_recall": m["per_class"]["fake"]["recall"],
                            "fake_f1": m["per_class"]["fake"]["f1"],
                            "roc_auc": m.get("roc_auc", float("nan")),
                            "fit_seconds": time.perf_counter() - t1 + vec_s,
                        }
                    )
                    print(f"fold {fold} {cand['name']:<28} {params} macroF1={m['macro_f1']:.4f}")
    return pd.DataFrame(rows)


def summarize_cv(cv: pd.DataFrame) -> pd.DataFrame:
    metrics = ["macro_f1", "accuracy", "fake_precision", "fake_recall", "fake_f1", "roc_auc"]
    g = cv.groupby(["candidate", "params"], sort=False)
    summary = g[metrics].mean().add_suffix("_mean").join(g[["macro_f1"]].std().add_suffix("_std"))
    return summary.reset_index().sort_values("macro_f1_mean", ascending=False)


def select(summary: pd.DataFrame, cfg: dict) -> dict:
    kinds = {c["name"]: c["model"] for c in cfg["candidates"]}
    eligible = summary[
        summary["candidate"].map(lambda n: kinds[n] in PROBABILISTIC_MODELS and kinds[n] != "dummy")
    ]
    best = eligible.iloc[0]
    cand = next(c for c in cfg["candidates"] if c["name"] == best["candidate"])
    return {"candidate": cand, "params": json.loads(best["params"]), "cv": best.to_dict()}


def excerpt(text: str, n: int = 160) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[:n].rstrip() + "…"


def top_features(pipe, k: int = 25) -> dict:
    vec, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return {}
    names = vec.get_feature_names_out()
    coef = clf.coef_.ravel()  # positive -> class 1 (authentic)
    order = np.argsort(coef)
    return {
        "towards_fake": [(names[i], float(coef[i])) for i in order[:k]],
        "towards_authentic": [(names[i], float(coef[i])) for i in order[::-1][:k]],
    }


def measure_latency(pipe, texts: list[str], n: int = 200) -> dict:
    sample = texts[:n]
    pipe.predict_proba(sample[:5])  # warm-up
    times = []
    for t in sample:
        t0 = time.perf_counter()
        pipe.predict_proba([t])
        times.append((time.perf_counter() - t0) * 1000)
    return {"n": len(times), "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Train and evaluate the credibility classifier.")
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "robust.json")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    ap.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports" / "robust")
    ap.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "model")
    ap.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="debug only: deterministic subsample of raw rows (results not comparable)",
    )
    args = ap.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    rep, fig_dir = args.reports, args.reports / "figures"
    rep.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    # 1. Audit
    raw = load_raw(args.raw)
    if args.max_rows:
        raw = raw.sample(n=min(args.max_rows, len(raw)), random_state=cfg["seed"]).sort_values("row_id")
    res = audit_and_clean(raw, cfg["conflict_threshold"], cfg["group_threshold"])
    data = res.data
    (rep / "data_audit.json").write_text(json.dumps(res.audit, indent=2), encoding="utf-8")
    res.excluded.to_csv(rep / "excluded_rows.csv.gz", index=False, compression="gzip")

    # 2. Held-out split, before anything is fit
    tr_idx, te_idx = grouped_train_test_split(data, cfg["test_fraction"], cfg["seed"])
    train, test = data.iloc[tr_idx].reset_index(drop=True), data.iloc[te_idx].reset_index(drop=True)
    assert not set(train["group_id"]) & set(test["group_id"]), "group leakage"
    write_split_manifest(data, te_idx, rep / "split_manifest.csv.gz")
    split_info = {
        "train_rows": len(train),
        "test_rows": len(test),
        "train_class_counts": {
            LABEL_NAMES[k]: int(v) for k, v in train["label"].value_counts().sort_index().items()
        },
        "test_class_counts": {
            LABEL_NAMES[k]: int(v) for k, v in test["label"].value_counts().sort_index().items()
        },
        "shared_groups": 0,
    }
    print(json.dumps(split_info, indent=2))

    # 3. Cross-validated model selection on train only
    cv = run_cv(train, cfg)
    cv.to_csv(rep / "cv_folds.csv", index=False)
    summary = summarize_cv(cv)
    summary.to_csv(rep / "cv_summary.csv", index=False)
    chosen = select(summary, cfg)
    print("Selected:", chosen["candidate"]["name"], chosen["params"])

    # 4. Refit every candidate's best config on full train; evaluate once on test.
    feats = cfg["feature_sets"]
    test_table, final_pipe = [], None
    for cand in cfg["candidates"]:
        best = summary[summary["candidate"] == cand["name"]].iloc[0]
        params = json.loads(best["params"])
        spec = feats.get(cand.get("features")) if cand.get("features") else None
        t0 = time.perf_counter()
        pipe = make_pipeline(spec, cand["model"], cfg["seed"], **params).fit(train["text"], train["label"])
        fit_s = time.perf_counter() - t0
        pred = pipe.predict(test["text"])
        score, is_prob = fake_scores(pipe, test["text"])
        m = classification_metrics(test["label"], pred, score)
        test_table.append(
            {
                "candidate": cand["name"],
                "params": params,
                "selected": cand is chosen["candidate"],
                "cv_macro_f1_mean": float(best["macro_f1_mean"]),
                "cv_macro_f1_std": float(best["macro_f1_std"]),
                "test": m,
                "train_fit_seconds": fit_s,
                "scores_are_probabilities": is_prob,
            }
        )
        if cand is chosen["candidate"]:
            final_pipe, final_pred, final_score = pipe, pred, score

    # 5. Final-model diagnostics
    p_fake = final_score
    calib = calibration_summary(test["label"], p_fake)
    bands = band_stats(test["label"], p_fake)
    save_figures(test["label"].to_numpy(), final_pred, p_fake, calib, fig_dir)
    latency = measure_latency(final_pipe, test["text"].tolist())

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "model.joblib"
    joblib.dump(final_pipe, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1e6

    final_metrics = next(r for r in test_table if r["selected"])
    metadata = {
        "model_name": "news-credibility-analyzer",
        "package_version": __version__,
        "selected_candidate": chosen["candidate"]["name"],
        "params": chosen["params"],
        "feature_set": cfg["feature_sets"].get(chosen["candidate"].get("features")),
        "labels": {str(k): v for k, v in LABEL_NAMES.items()},
        "score": "P(fake) from predict_proba, column of class 0",
        "decision_rule": "argmax of predicted probabilities (equivalent to P(fake) >= 0.5)",
        "dataset": {
            "hf_repo": HF_REPO_ID,
            "hf_revision": HF_REVISION,
            "sha256": EXPECTED_SHA256,
            "original_doi": "10.17632/kjh887ct4j.1",
        },
        "config": cfg,
        "split": split_info,
        "test_metrics": final_metrics["test"],
        "calibration": {k: v for k, v in calib.items() if k != "bins"},
        "bands": bands,
        "latency_single_article_cpu": latency,
        "artifact_size_mb": round(size_mb, 2),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__},
    }
    (args.model_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    (rep / "test_results.json").write_text(
        json.dumps(
            {
                "note": "Selection used CV on the training split only; test was evaluated once per candidate.",
                "split": split_info,
                "candidates": test_table,
                "calibration": calib,
                "bands": bands,
                "latency_single_article_cpu": latency,
                "artifact_size_mb": round(size_mb, 2),
                "total_runtime_minutes": round((time.perf_counter() - t_start) / 60, 1),
                "environment": metadata["environment"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (rep / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _write_failure_cases(test, final_pred, p_fake, rep / "failure_cases.md")
    _write_top_features(top_features(final_pipe), rep / "top_features.md")
    print(json.dumps(final_metrics["test"], indent=2))
    print(f"Saved model to {model_path} ({size_mb:.1f} MB)")


def _write_failure_cases(test: pd.DataFrame, pred, p_fake, path: Path, k: int = 6) -> None:
    df = test.assign(pred=pred, p_fake=p_fake)
    fn = df[(df.label == FAKE) & (df.pred == AUTHENTIC)].sort_values("p_fake").head(k)
    fp = df[(df.label == AUTHENTIC) & (df.pred == FAKE)].sort_values("p_fake", ascending=False).head(k)
    lines = [
        "# Representative failure cases (held-out test set)",
        "",
        'Excerpts (first ~160 characters) from BanFakeNews-2.0 (CC BY 4.0). "Dataset label" is the',
        "dataset's annotation, not a verified fact. These are the most confident errors, chosen",
        "automatically by `nca.train`.",
        "",
        f"## Labelled fake (0), predicted authentic-like — {int(((df.label == FAKE) & (df.pred == AUTHENTIC)).sum())} total",
        "",
        "| row_id | P(fake) | excerpt |",
        "|---:|---:|---|",
    ]
    lines += [
        f"| {r.row_id} | {r.p_fake:.3f} | {excerpt(r.text).replace('|', '/')} |" for r in fn.itertuples()
    ]
    lines += [
        "",
        f"## Labelled authentic (1), predicted fake-like — {int(((df.label == AUTHENTIC) & (df.pred == FAKE)).sum())} total",
        "",
        "| row_id | P(fake) | excerpt |",
        "|---:|---:|---|",
    ]
    lines += [
        f"| {r.row_id} | {r.p_fake:.3f} | {excerpt(r.text).replace('|', '/')} |" for r in fp.itertuples()
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_top_features(tf: dict, path: Path) -> None:
    if not tf:
        return
    lines = [
        "# Most influential n-grams of the selected linear model",
        "",
        "Positive coefficient pushes towards authentic (1); negative towards fake (0).",
        "",
    ]
    for side in ("towards_fake", "towards_authentic"):
        lines += [f"## {side.replace('_', ' ')}", "", "| n-gram | coefficient |", "|---|---:|"]
        lines += [f"| {n} | {c:+.3f} |" for n, c in tf[side]]
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
