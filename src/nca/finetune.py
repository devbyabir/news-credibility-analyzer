"""Fine-tune a pretrained Bangla transformer (default: csebuetnlp/banglabert).

Designed for a GPU runtime (Colab / Kaggle / any CUDA machine); it also runs on
CPU for smoke tests with --max-train-samples / --max-steps.

    pip install -e ".[finetune]"
    python -m nca.data download
    python -m nca.finetune --output-dir artifacts/banglabert

Protocol: identical data audit and *identical* held-out test split as
``nca.train`` (same config + seed). A grouped validation split is carved from
the training portion for early stopping / checkpoint selection; the test split
is scored exactly once, after training, with the selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from nca.data import DEFAULT_RAW_PATH, PROJECT_ROOT, audit_and_clean, grouped_train_test_split, load_raw
from nca.evaluate import classification_metrics
from nca.labels import AUTHENTIC, FAKE, LABEL_NAMES
from nca.text import PREPROCESSORS


def build_splits(cfg: dict, raw_path: Path, val_fraction: float, max_rows: int | None = None):
    raw = load_raw(raw_path)
    if max_rows:
        raw = raw.sample(n=min(max_rows, len(raw)), random_state=cfg["seed"]).sort_values("row_id")
    data = audit_and_clean(raw, cfg["conflict_threshold"], cfg["group_threshold"]).data
    tr_idx, te_idx = grouped_train_test_split(data, cfg["test_fraction"], cfg["seed"])
    train, test = data.iloc[tr_idx].reset_index(drop=True), data.iloc[te_idx].reset_index(drop=True)
    fit_idx, val_idx = grouped_train_test_split(train, val_fraction, cfg["seed"] + 1)
    fit, val = train.iloc[fit_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)
    return fit, val, test


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "robust.json")
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW_PATH)
    ap.add_argument("--model-name", default="csebuetnlp/banglabert")
    ap.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "banglabert")
    ap.add_argument(
        "--results", type=Path, default=None, help="JSON results path (default: <output-dir>/results.json)"
    )
    ap.add_argument(
        "--preprocessor",
        choices=sorted(PREPROCESSORS),
        default="robust",
        help="text preprocessing; robust = same shortcut masking as the deployed model",
    )
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--max-rows", type=int, default=None, help="debug: subsample raw rows")
    ap.add_argument("--max-train-samples", type=int, default=None, help="debug: cap training rows")
    ap.add_argument("--max-eval-samples", type=int, default=None, help="debug: cap val/test rows")
    ap.add_argument("--max-steps", type=int, default=-1, help="debug: stop after N optimiser steps")
    args = ap.parse_args(argv)

    import torch
    from torch import nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    normalize_text = PREPROCESSORS[args.preprocessor]
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    set_seed(cfg["seed"])
    fit, val, test = build_splits(cfg, args.raw, args.val_fraction, args.max_rows)
    debug = any(x is not None for x in (args.max_rows, args.max_train_samples, args.max_eval_samples)) or (
        args.max_steps > 0
    )
    if args.max_train_samples:
        fit = fit.sample(n=min(args.max_train_samples, len(fit)), random_state=cfg["seed"])
    if args.max_eval_samples:
        val = val.sample(n=min(args.max_eval_samples, len(val)), random_state=cfg["seed"])
        test = test.sample(n=min(args.max_eval_samples, len(test)), random_state=cfg["seed"])
    print(
        f"fit={len(fit)} val={len(val)} test={len(test)} device={'cuda' if torch.cuda.is_available() else 'cpu'}"
    )

    tok = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={k: v for k, v in LABEL_NAMES.items()},
        label2id={v: k for k, v in LABEL_NAMES.items()},
    )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.enc = tok(
                [normalize_text(t) for t in df["text"]], truncation=True, max_length=args.max_length
            )
            self.labels = df["label"].astype(int).tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.labels[i]
            return item

    counts = np.bincount(fit["label"].astype(int), minlength=2).astype(float)
    weights = torch.tensor(counts.sum() / (2 * np.maximum(counts, 1)), dtype=torch.float)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            loss = nn.functional.cross_entropy(
                outputs.logits, labels, weight=weights.to(outputs.logits.device)
            )
            return (loss, outputs) if return_outputs else loss

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        p_fake = torch.softmax(torch.tensor(logits), dim=-1)[:, FAKE].numpy()
        pred = np.where(p_fake >= 0.5, FAKE, AUTHENTIC)
        m = classification_metrics(labels, pred, p_fake)
        return {"macro_f1": m["macro_f1"], "fake_f1": m["per_class"]["fake"]["f1"], "accuracy": m["accuracy"]}

    targs = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_steps=0.06,  # float = ratio of total steps (transformers >= 5)
        eval_strategy="steps" if debug else "epoch",
        eval_steps=max(1, args.max_steps) if debug else None,
        save_strategy="steps" if debug else "epoch",
        save_steps=max(1, args.max_steps) if debug else None,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        seed=cfg["seed"],
        logging_steps=50,
        report_to="none",
    )
    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=TextDataset(fit),
        eval_dataset=TextDataset(val),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    t0 = time.perf_counter()
    trainer.train()
    train_minutes = (time.perf_counter() - t0) / 60

    # Single evaluation on the held-out test split with the selected checkpoint.
    out = trainer.predict(TextDataset(test))
    p_fake = torch.softmax(torch.tensor(out.predictions), dim=-1)[:, FAKE].numpy()
    pred = np.where(p_fake >= 0.5, FAKE, AUTHENTIC)
    test_metrics = classification_metrics(test["label"].to_numpy(), pred, p_fake)

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    tok.save_pretrained(str(final_dir))
    size_mb = sum(f.stat().st_size for f in final_dir.rglob("*") if f.is_file()) / 1e6

    model.eval()
    device = next(model.parameters()).device
    sample = [normalize_text(t) for t in test["text"].head(50)]
    times = []
    with torch.no_grad():
        for t in sample:
            enc = tok(t, truncation=True, max_length=args.max_length, return_tensors="pt").to(device)
            s = time.perf_counter()
            model(**enc)
            times.append((time.perf_counter() - s) * 1000)

    results = {
        "debug_run": debug,
        "note": "debug runs use tiny subsets and are NOT a benchmark" if debug else "full run",
        "model_name": args.model_name,
        "sizes": {"fit": len(fit), "val": len(val), "test": len(test)},
        "hyperparameters": {
            k: v for k, v in vars(args).items() if k not in {"config", "raw", "output_dir", "results"}
        },
        "best_val_macro_f1": trainer.state.best_metric,
        "test": test_metrics,
        "train_minutes": round(train_minutes, 2),
        "device": str(device),
        "latency_single_article_ms_median": float(np.median(times)) if times else None,
        "artifact_size_mb": round(size_mb, 1),
        "torch": torch.__version__,
    }
    path = args.results or (args.output_dir / "results.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        json.dumps({k: results[k] for k in ("debug_run", "best_val_macro_f1", "artifact_size_mb")}, indent=2)
    )
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}  -> {path}")


if __name__ == "__main__":
    main()
