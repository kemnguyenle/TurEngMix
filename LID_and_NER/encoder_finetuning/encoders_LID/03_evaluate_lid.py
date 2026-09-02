"""
03_evaluate_lid.py

Loads a trained token classification model, runs it on the untouched test
split, and computes:
  - Accuracy
  - Macro F1 (primary metric)
  - Precision / Recall (macro)
  - Per-class precision / recall / F1
  - Confusion matrix
  - Full classification report

Saves everything to --output_dir/metrics.json (+ confusion_matrix.csv and
classification_report.txt), and optionally saves token-level predictions for
error analysis.

Only run this AFTER model selection (checkpoint choice, hyperparameters) is
finalized on the validation set, since the test set should be touched once.

Usage:
    python 03_evaluate_lid.py \
        --model_dir ../models/berturk_lid \
        --data_dir ../data/processed/lid \
        --output_dir ../results/berturk_lid \
        --save_predictions
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from transformers import AutoTokenizer, AutoModelForTokenClassification


def tokenize_and_align_labels(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )
    all_labels = []
    all_word_ids = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        all_labels.append(label_ids)
        all_word_ids.append(word_ids)
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained token classification model")
    parser.add_argument("--model_dir", type=str, required=True,
                         help="Directory of the trained model (output_dir from 02_train_lid.py)")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to the processed DatasetDict directory (same one used in training)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--save_predictions", action="store_true",
                         help="Save token-level predictions for error analysis")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model, tokenizer, label maps ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()

    with open(os.path.join(args.data_dir, "id2label.json"), "r", encoding="utf-8") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    label_names = [id2label[i] for i in range(len(id2label))]

    # --- Load untouched test set ---
    dataset = load_from_disk(args.data_dir)
    test_dataset = dataset["test"]

    tokenized_test = test_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    # --- Run inference in batches ---
    all_true_ids = []
    all_pred_ids = []
    per_example_records = [] if args.save_predictions else None

    n = len(tokenized_test)
    for start in range(0, n, args.batch_size):
        batch = tokenized_test[start:start + args.batch_size]
        input_ids = [torch.tensor(x) for x in batch["input_ids"]]
        attn_mask = [torch.tensor(x) for x in batch["attention_mask"]]
        labels_batch = batch["labels"]

        max_len = max(len(x) for x in input_ids)
        pad_id = tokenizer.pad_token_id or 0

        padded_input_ids = torch.full((len(input_ids), max_len), pad_id, dtype=torch.long)
        padded_attn = torch.zeros((len(input_ids), max_len), dtype=torch.long)
        for i, (ids, mask) in enumerate(zip(input_ids, attn_mask)):
            padded_input_ids[i, :len(ids)] = ids
            padded_attn[i, :len(mask)] = mask

        with torch.no_grad():
            outputs = model(
                input_ids=padded_input_ids.to(device),
                attention_mask=padded_attn.to(device),
            )
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        for i, labels in enumerate(labels_batch):
            example_true = []
            example_pred = []
            for j, lab in enumerate(labels):
                if lab == -100:
                    continue
                example_true.append(lab)
                example_pred.append(int(preds[i, j]))
            all_true_ids.extend(example_true)
            all_pred_ids.extend(example_pred)

            if args.save_predictions:
                doc_idx = start + i
                per_example_records.append({
                    "document_id": test_dataset[doc_idx]["document_id"],
                    "sentence_id": test_dataset[doc_idx]["sentence_id"],
                    "tokens": test_dataset[doc_idx]["tokens"],
                    "true_labels": [id2label[l] for l in example_true],
                    "pred_labels": [id2label[p] for p in example_pred],
                    "correct": [t == p for t, p in zip(example_true, example_pred)],
                })

    y_true = np.array(all_true_ids)
    y_pred = np.array(all_pred_ids)

    # --- Metrics ---
    present_labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    present_names = [id2label[i] for i in present_labels]

    accuracy = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=present_labels, average="macro", zero_division=0
    )
    per_class_p, per_class_r, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=present_labels, average=None, zero_division=0
    )

    per_class_metrics = {
        present_names[i]: {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f1[i]),
            "support": int(per_class_support[i]),
        }
        for i in range(len(present_labels))
    }

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    cm_df = pd.DataFrame(cm, index=present_names, columns=present_names)

    report_str = classification_report(
        y_true, y_pred, labels=present_labels, target_names=present_names,
        zero_division=0, digits=4,
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "per_class": per_class_metrics,
        "n_tokens_evaluated": int(len(y_true)),
    }

    # --- Save outputs ---
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm_df.to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_str)

    if args.save_predictions:
        with open(os.path.join(args.output_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
            for rec in per_example_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Evaluation complete.")
    print(f"Accuracy: {accuracy:.4f}  Macro F1: {macro_f1:.4f}  "
          f"Macro P: {macro_p:.4f}  Macro R: {macro_r:.4f}")
    print(f"Saved metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
