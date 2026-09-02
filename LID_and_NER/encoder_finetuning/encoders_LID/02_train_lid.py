"""
02_train_lid.py

Generic token classification training script. The same script is used for
BERTurk, XLM-R, and TurkishBERTweet (and later, unchanged, for NER) — only
--model_name and --output_dir (and optionally --label_dir if you point it at
the NER-processed dataset) differ between runs.

This script does NOT create the train/val/test split; it loads the fixed
DatasetDict produced by 01_prepare_dataset.py so every model is trained and
evaluated on exactly the same data.

Usage:
    python 02_train_lid.py \
        --model_name dbmdz/bert-base-turkish-cased \
        --data_dir ../data/processed/lid \
        --output_dir ../models/berturk_lid \
        --epochs 10 --batch_size 16 --learning_rate 2e-5 \
        --weight_decay 0.01 --seed 42
"""

import argparse
import json
import os

import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)
import evaluate

seqeval_available = True
try:
    seqeval = evaluate.load("seqeval")
except Exception:
    seqeval_available = False


def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=False):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=512,
    )

    all_labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], padding)
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword piece of a new word: use the real label
                label_ids.append(label[word_idx])
            else:
                # Continuation subword piece: ignore in the loss
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def build_compute_metrics(id2label):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(pred, label) if l != -100]
            for pred, label in zip(predictions, labels)
        ]

        if seqeval_available:
            results = seqeval.compute(predictions=true_predictions, references=true_labels)
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
        else:
            # Fallback: flat accuracy only (seqeval expects BIO-style spans;
            # LID classes are flat, so this fallback is fine for LID and is
            # only a coarse signal during training either way — the detailed
            # per-class metrics are computed properly in 03_evaluate_lid.py).
            flat_preds = [p for seq in true_predictions for p in seq]
            flat_labels = [l for seq in true_labels for l in seq]
            acc = np.mean([p == l for p, l in zip(flat_preds, flat_labels)])
            return {"accuracy": float(acc)}

    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description="Train a token classification model")
    parser.add_argument("--model_name", type=str, required=True,
                         help="Pretrained checkpoint, e.g. dbmdz/bert-base-turkish-cased, "
                              "FacebookAI/xlm-roberta-base, VRLLab/TurkishBERTweet")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to the processed DatasetDict directory from 01_prepare_dataset.py")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--label_all_tokens", action="store_true",
                         help="If set, label all subword pieces instead of only the first "
                              "(default Hugging Face recommendation is off)")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Load fixed dataset and label maps (identical across all encoders) ---
    dataset = load_from_disk(args.data_dir)

    with open(os.path.join(args.data_dir, "label2id.json"), "r", encoding="utf-8") as f:
        label2id = json.load(f)
    with open(os.path.join(args.data_dir, "id2label.json"), "r", encoding="utf-8") as f:
        id2label_str_keys = json.load(f)
    id2label = {int(k): v for k, v in id2label_str_keys.items()}
    num_labels = len(label2id)

    # --- Tokenizer & label alignment ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples, tokenizer, label_all_tokens=args.label_all_tokens
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # --- Model ---
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=args.fp16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(id2label),
    )

    trainer.train()

    # Save the best checkpoint (final model + tokenizer) to output_dir
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    with open(os.path.join(args.output_dir, "label2id.json"), "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "id2label.json"), "w", encoding="utf-8") as f:
        json.dump(id2label_str_keys, f, ensure_ascii=False, indent=2)

    # Save run config for reproducibility
    run_config = vars(args)
    with open(os.path.join(args.output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2)

    print(f"Training complete. Best model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
