"""
03_evaluate_ner.py

Loads a trained NER model, runs it on the untouched test split, and computes
entity-level (span-based) metrics — the standard way to score BIO-tagged
NER, via seqeval:
  - Entity-level Precision / Recall / F1 (overall) — Macro F1 is primary
  - Per-entity-type Precision / Recall / F1 / Support
  - Full seqeval classification report

It also reports secondary, diagnostic token/tag-level metrics (flat
accuracy + confusion matrix over the raw B-/I-/O tags), which are useful for
error analysis (e.g. distinguishing boundary errors from type errors) but
are NOT the primary metric — entity-level F1 is.

Saves everything to --output_dir/metrics.json (+ confusion_matrix.csv and
classification_report.txt), and optionally saves token-level predictions for
error analysis.

Only run this AFTER model selection (checkpoint choice, hyperparameters) is
finalized on the validation set, since the test set should be touched once.

Usage:
    python 03_evaluate_ner.py \
        --model_dir ../models/berturk_ner \
        --data_dir ../data/processed/ner \
        --output_dir ../results/berturk_ner \
        --save_predictions
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import load_from_disk
from sklearn.metrics import confusion_matrix
from transformers import AutoTokenizer, AutoModelForTokenClassification

try:
    from seqeval.metrics import (
        precision_score as seq_precision_score,
        recall_score as seq_recall_score,
        f1_score as seq_f1_score,
        classification_report as seq_classification_report,
    )
except ImportError as e:
    raise ImportError(
        "seqeval is required for entity-level NER evaluation. "
        "Install it with `pip install seqeval`."
    ) from e


def tokenize_and_align_labels(examples, tokenizer, max_length):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )
    all_labels = []
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
    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained NER model (entity-level)")
    parser.add_argument("--model_dir", type=str, required=True,
                         help="Directory of the trained model (output_dir from 02_train_ner.py)")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="Path to the NER-processed DatasetDict directory (same one used in training)")
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

    # --- Load untouched test set ---
    dataset = load_from_disk(args.data_dir)
    test_dataset = dataset["test"]

    tokenized_test = test_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, args.max_length),
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    # --- Run inference in batches ---
    true_tag_sequences = []   # list of lists of BIO tag strings (per sentence) — for seqeval
    pred_tag_sequences = []
    all_true_flat = []        # flat lists of BIO tags — for the diagnostic confusion matrix
    all_pred_flat = []
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
            example_true_ids, example_pred_ids = [], []
            for j, lab in enumerate(labels):
                if lab == -100:
                    continue
                example_true_ids.append(lab)
                example_pred_ids.append(int(preds[i, j]))

            example_true_tags = [id2label[l] for l in example_true_ids]
            example_pred_tags = [id2label[p] for p in example_pred_ids]

            true_tag_sequences.append(example_true_tags)
            pred_tag_sequences.append(example_pred_tags)
            all_true_flat.extend(example_true_tags)
            all_pred_flat.extend(example_pred_tags)

            if args.save_predictions:
                doc_idx = start + i
                per_example_records.append({
                    "document_id": test_dataset[doc_idx]["document_id"],
                    "sentence_id": test_dataset[doc_idx]["sentence_id"],
                    "tokens": test_dataset[doc_idx]["tokens"],
                    "true_labels": example_true_tags,
                    "pred_labels": example_pred_tags,
                    "correct": [t == p for t, p in zip(example_true_tags, example_pred_tags)],
                })

    # --- Entity-level (primary) metrics via seqeval ---
    entity_precision = seq_precision_score(true_tag_sequences, pred_tag_sequences, average="micro")
    entity_recall = seq_recall_score(true_tag_sequences, pred_tag_sequences, average="micro")
    entity_f1_micro = seq_f1_score(true_tag_sequences, pred_tag_sequences, average="micro")
    entity_f1_macro = seq_f1_score(true_tag_sequences, pred_tag_sequences, average="macro")

    report_dict = seq_classification_report(
        true_tag_sequences, pred_tag_sequences, output_dict=True, zero_division=0
    )
    report_str = seq_classification_report(
        true_tag_sequences, pred_tag_sequences, output_dict=False, zero_division=0, digits=4
    )

    per_entity_metrics = {
        entity_type: {
            "precision": float(vals["precision"]),
            "recall": float(vals["recall"]),
            "f1": float(vals["f1-score"]),
            "support": int(vals["support"]),
        }
        for entity_type, vals in report_dict.items()
        if entity_type not in ("micro avg", "macro avg", "weighted avg")
    }

    # --- Secondary diagnostic metrics: flat tag-level accuracy + confusion matrix ---
    flat_accuracy = float(np.mean([t == p for t, p in zip(all_true_flat, all_pred_flat)]))
    tag_names = sorted(set(all_true_flat) | set(all_pred_flat))
    cm = confusion_matrix(all_true_flat, all_pred_flat, labels=tag_names)
    cm_df = pd.DataFrame(cm, index=tag_names, columns=tag_names)

    metrics = {
        "primary_metric": "entity_level_macro_f1",
        "entity_level": {
            "macro_f1": float(entity_f1_macro),
            "micro_f1": float(entity_f1_micro),
            "micro_precision": float(entity_precision),
            "micro_recall": float(entity_recall),
            "per_entity_type": per_entity_metrics,
        },
        "tag_level_diagnostic": {
            "flat_accuracy": flat_accuracy,
            "note": "Flat B-/I-/O tag accuracy is a secondary diagnostic only; "
                    "entity_level metrics above are the ones to report/compare.",
        },
        "n_sentences_evaluated": len(true_tag_sequences),
        "n_tokens_evaluated": len(all_true_flat),
    }

    # --- Save outputs ---
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    cm_df.to_csv(os.path.join(args.output_dir, "confusion_matrix.csv"))

    with open(os.path.join(args.output_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write("Entity-level (seqeval) classification report:\n\n")
        f.write(report_str)

    if args.save_predictions:
        with open(os.path.join(args.output_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
            for rec in per_example_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Evaluation complete.")
    print(f"Entity-level Macro F1: {entity_f1_macro:.4f}  Micro F1: {entity_f1_micro:.4f}  "
          f"Precision: {entity_precision:.4f}  Recall: {entity_recall:.4f}")
    print(f"(Secondary) flat tag accuracy: {flat_accuracy:.4f}")
    print(f"Saved metrics to: {args.output_dir}")


if __name__ == "__main__":
    main()
