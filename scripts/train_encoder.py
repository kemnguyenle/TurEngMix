#!/usr/bin/env python3
"""Fine-tune an encoder for token classification on TurEngMix.

    python scripts/train_encoder.py --task lid \
        --model dbmdz/bert-base-turkish-cased --out models/berturk_lid

One script covers both tasks and all three encoders; only --task, --model and
--out change between the runs.

Learning rate 2e-5, weight decay 0.01, at most 10 epochs,
and four random seeds. Pass --seeds to train the full set in one command; each
seed writes its own model directory, and `scripts/report.py` averages them
into the mean +/- SD.

Checkpoint selection uses the same metric the paper reports for that task —
flat macro-F1 over the six LID classes, or over the nineteen BIO labels for
NER. 

Sequence length is one value used for training and evaluation alike
(--max-length, default 512).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turengmix import scoring  # noqa: E402
from turengmix.data import DEFAULT_BENCHMARK, dump_json, load_benchmark, split_sentences  # noqa: E402
from turengmix.labels import id2label, label2id, labels_for  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


def tokenize_and_align(batch, tokenizer, l2i, max_length, label_all_tokens):
    enc = tokenizer(
        batch["tokens"], truncation=True, is_split_into_words=True,
        max_length=max_length,
    )
    all_labels, n_truncated = [], 0
    for i, words in enumerate(batch["labels"]):
        word_ids = enc.word_ids(batch_index=i)
        seen = {w for w in word_ids if w is not None}
        if len(seen) < len(words):
            n_truncated += len(words) - len(seen)
        previous, ids = None, []
        for w in word_ids:
            if w is None:
                ids.append(-100)                      # [CLS], [SEP], padding
            elif w != previous:
                ids.append(l2i[words[w]])             # first subword: real label
            else:
                # Continuation subwords are excluded from the loss. For NER,
                # labelling them with the word's own tag would put a second
                # `B-` inside one entity; `I-` would be the only correct
                # relabelling, so --label-all-tokens is off by default.
                ids.append(l2i[words[w]] if label_all_tokens else -100)
            previous = w
        all_labels.append(ids)
    enc["labels"] = all_labels
    if n_truncated:
        enc["n_truncated_words"] = [n_truncated] + [0] * (len(batch["labels"]) - 1)
    return enc


def build_compute_metrics(task: str):
    """Validation metric = the metric the paper reports for this task.

    Macro-averaged over the closed label set from `turengmix.labels`.
    """
    i2l = id2label(task)

    def compute(eval_pred):
        logits, gold = eval_pred
        preds = np.argmax(logits, axis=2)
        flat_true, flat_pred = [], []
        for p_row, g_row in zip(preds, gold):
            for p, g in zip(p_row, g_row):
                if g != -100:
                    flat_pred.append(i2l[int(p)])
                    flat_true.append(i2l[int(g)])
        m = scoring.token_metrics(flat_true, flat_pred, task)
        return {"f1": m["macro_f1"], "accuracy": m["accuracy"],
                "precision": m["macro_precision"], "recall": m["macro_recall"]}

    return compute


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("lid", "ner"))
    ap.add_argument("--model", required=True,
                    help="dbmdz/bert-base-turkish-cased | FacebookAI/xlm-roberta-base "
                         "| VRLLab/TurkishBERTweet")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    ap.add_argument("--learning-rate", type=float, default=2e-5, dest="learning_rate")
    ap.add_argument("--weight-decay", type=float, default=0.01, dest="weight_decay")
    ap.add_argument("--warmup-ratio", type=float, default=0.06, dest="warmup_ratio")
    ap.add_argument("--max-length", type=int, default=512, dest="max_length",
                    help="must match evaluate_encoder.py; the default covers "
                         "every sentence in this benchmark for all three encoders")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="train once per seed; the paper reports mean +/- SD "
                         "over four seeds (e.g. --seeds 1 2 3 4)")
    ap.add_argument("--label-all-tokens", action="store_true", dest="label_all_tokens")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--early-stopping-patience", type=int, default=3,
                    dest="patience", help="0 disables early stopping")
    args = ap.parse_args()

    from datasets import Dataset, DatasetDict
    from transformers import (
        AutoModelForTokenClassification, AutoTokenizer,
        DataCollatorForTokenClassification, EarlyStoppingCallback,
        Trainer, TrainingArguments, set_seed,
    )

    labels = labels_for(args.task)
    l2i, i2l = label2id(args.task), id2label(args.task)

    df = load_benchmark(args.benchmark)
    parts = split_sentences(df, args.task)
    ds = DatasetDict({
        name: Dataset.from_list([
            {"doc_id": s.doc_id, "sent_id": s.sent_id,
             "tokens": s.tokens, "labels": s.labels}
            for s in units
        ])
        for name, units in parts.items()
    })
    print({k: len(v) for k, v in ds.items()}, "sentences")

    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)

    # Report truncation.
    dropped = 0
    for units in parts.values():
        for s in units:
            n = len(tokenizer(s.tokens, is_split_into_words=True)["input_ids"])
            if n > args.max_length:
                enc = tokenizer(s.tokens, is_split_into_words=True,
                                truncation=True, max_length=args.max_length)
                dropped += len(s.tokens) - len({w for w in enc.word_ids() if w is not None})
    if dropped:
        print(f"WARNING: --max-length {args.max_length} truncates {dropped} words "
              f"({100 * dropped / len(df):.2f}% of the benchmark). These are "
              f"excluded from training and from every score.")

    tokenized = ds.map(
        lambda b: tokenize_and_align(b, tokenizer, l2i, args.max_length,
                                     args.label_all_tokens),
        batched=True, remove_columns=ds["train"].column_names,
    )

    summary = []
    for seed in args.seeds:
        out_dir = args.out if len(args.seeds) == 1 else Path(f"{args.out}_seed{seed}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== seed {seed} -> {out_dir} ===")
        set_seed(seed)

        model = AutoModelForTokenClassification.from_pretrained(
            args.model, num_labels=len(labels), id2label=i2l, label2id=l2i,
        )

        targs = TrainingArguments(
            output_dir=str(out_dir / "checkpoints"),
            eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
            load_best_model_at_end=True,
            metric_for_best_model="f1", greater_is_better=True,
            save_total_limit=2, seed=seed, data_seed=seed,
            fp16=args.fp16, report_to="none",
        )

        callbacks = ([EarlyStoppingCallback(early_stopping_patience=args.patience)]
                     if args.patience else [])

   
        trainer = Trainer(
            model=model, args=targs,
            train_dataset=tokenized["train"], eval_dataset=tokenized["validation"],
            processing_class=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
            compute_metrics=build_compute_metrics(args.task),
            callbacks=callbacks,
        )

        trainer.train()
        trainer.save_model(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        dump_json({
            "task": args.task, "model": args.model, "seed": seed,
            "metric_for_best_model": ("flat macro-F1 over the six LID classes"
                                      if args.task == "lid" else
                                      "flat macro-F1 over the nineteen BIO labels"),
            "max_length": args.max_length,
            "words_truncated": dropped,
            "hyperparameters": {k: v for k, v in vars(args).items()
                                if k not in ("out", "benchmark", "seeds")},
            "validation_history": [h for h in trainer.state.log_history
                                   if "eval_f1" in h],
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_f1": trainer.state.best_metric,
        }, out_dir / "train_config.json")

        summary.append((seed, out_dir, trainer.state.best_metric))
        print(f"seed {seed}: best validation macro-F1 {trainer.state.best_metric:.4f}")

    print("\n=== summary ===")
    for seed, out_dir, best in summary:
        print(f"  seed {seed:<4} {best:.4f}  {out_dir}")
    if len(summary) > 1:
        agg = scoring.aggregate_seeds([b for _, _, b in summary])
        print(f"  validation macro-F1 across seeds: "
              f"{agg['mean']:.4f} +/- {agg['std']:.4f}")

    print("\nEvaluate each with:")
    for _, out_dir, _ in summary:
        print(f"  python scripts/evaluate_encoder.py --task {args.task} "
              f"--model-dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
