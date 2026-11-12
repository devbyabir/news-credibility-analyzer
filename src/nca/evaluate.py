"""Metrics, calibration checks and report figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from nca.labels import AUTHENTIC, FAKE, LABEL_NAMES

# Score bands shown in the UI. Scores are P(fake) from the model; the bands'
# empirical fake rates are measured on the held-out test set (see band_stats).
BANDS = [
    ("strongly_authentic_like", 0.00, 0.20),
    ("leans_authentic_like", 0.20, 0.40),
    ("uncertain", 0.40, 0.60),
    ("leans_fake_like", 0.60, 0.80),
    ("strongly_fake_like", 0.80, 1.0000001),
]


def classification_metrics(y_true, y_pred, fake_score=None) -> dict:
    """Full metric set. ``fake_score`` is any score that increases with P(fake)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = [FAKE, AUTHENTIC]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class": {
            LABEL_NAMES[lab]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i, lab in enumerate(labels)
        },
        "confusion_matrix": {
            "labels": ["fake (0)", "authentic (1)"],
            "rows_are": "true label",
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
    }
    if fake_score is not None and len(np.unique(y_true)) == 2:
        fake_score = np.asarray(fake_score, dtype=float)
        is_fake = (y_true == FAKE).astype(int)
        out["roc_auc"] = float(roc_auc_score(is_fake, fake_score))
        out["pr_auc_fake"] = float(average_precision_score(is_fake, fake_score))
        out["fake_prevalence"] = float(is_fake.mean())
    return out


def calibration_summary(y_true, p_fake, n_bins: int = 10) -> dict:
    y = (np.asarray(y_true) == FAKE).astype(int)
    p = np.clip(np.asarray(p_fake, dtype=float), 0, 1)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins, ece = [], 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += m.mean() * abs(conf - acc)
        bins.append(
            {
                "bin": [float(edges[b]), float(edges[b + 1])],
                "n": int(m.sum()),
                "mean_p_fake": conf,
                "observed_fake_rate": acc,
            }
        )
    return {"brier_fake": float(brier_score_loss(y, p)), "ece_10_bins": float(ece), "bins": bins}


def band_of(p_fake: float) -> str:
    for name, lo, hi in BANDS:
        if lo <= p_fake < hi:
            return name
    return BANDS[-1][0]


def band_stats(y_true, p_fake) -> list[dict]:
    """For each UI band: how many test articles fell in it and how many were labelled fake."""
    y = np.asarray(y_true)
    p = np.asarray(p_fake, dtype=float)
    rows = []
    for name, lo, hi in BANDS:
        m = (p >= lo) & (p < hi)
        n = int(m.sum())
        rows.append(
            {
                "band": name,
                "p_fake_range": [lo, min(hi, 1.0)],
                "n_test": n,
                "share_of_test": float(n / len(p)) if len(p) else 0.0,
                "observed_fake_rate": float((y[m] == FAKE).mean()) if n else None,
            }
        )
    return rows


# ------------------------------------------------------------------- figures


def save_figures(y_true, y_pred, p_fake, calib: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    cm = confusion_matrix(y_true, y_pred, labels=[FAKE, AUTHENTIC])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    names = ["Fake (0)", "Authentic (1)"]
    ax.set_xticks([0, 1], names)
    ax.set_yticks([0, 1], names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Dataset label")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12,
            )
    ax.set_title("Held-out test confusion matrix")
    fig.tight_layout()
    paths.append(out_dir / "confusion_matrix.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    is_fake = (np.asarray(y_true) == FAKE).astype(int)
    prec, rec, _ = precision_recall_curve(is_fake, p_fake)
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.plot(rec, prec, lw=2)
    ax.axhline(is_fake.mean(), ls="--", c="gray", lw=1, label=f"prevalence {is_fake.mean():.2f}")
    ax.set_xlabel("Recall (fake class)")
    ax.set_ylabel("Precision (fake class)")
    ax.set_title("Precision-recall, fake class (test)")
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    paths.append(out_dir / "pr_curve_fake.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    xs = [b["mean_p_fake"] for b in calib["bins"]]
    ys = [b["observed_fake_rate"] for b in calib["bins"]]
    ax.plot([0, 1], [0, 1], ls="--", c="gray", lw=1)
    ax.plot(xs, ys, marker="o", lw=2)
    ax.set_xlabel("Model score P(fake)")
    ax.set_ylabel("Observed fake rate")
    ax.set_title(f"Reliability (test), ECE={calib['ece_10_bins']:.3f}")
    fig.tight_layout()
    paths.append(out_dir / "reliability.png")
    fig.savefig(paths[-1], dpi=150)
    plt.close(fig)
    return paths
