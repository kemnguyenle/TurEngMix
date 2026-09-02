# Turkish-English Code-Switched LID Pipeline

Supervised token classification pipeline for Turkish-English code-switched
Language Identification (LID), built on Hugging Face Transformers. Compares
three pretrained encoders — **BERTurk**, **XLM-R**, and **TurkishBERTweet** —
on an identical dataset split, preprocessing, training procedure, and
evaluation, so that any performance differences are attributable only to the
pretrained checkpoint. The same pipeline is designed to be reused for NER
later by changing only the label column / label map / number of classes.

## Project structure

```
project/
  data/
    raw/            # original annotation spreadsheet(s) go here
    processed/       # output of 01_prepare_dataset.py (DatasetDict + label maps)
  scripts/
    01_prepare_dataset.py
    02_train_lid.py
    03_evaluate_lid.py
  models/            # trained checkpoints
  results/           # metrics, confusion matrices, classification reports
  requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Prepare the dataset

Input: a spreadsheet (CSV or XLSX) with one row per token, containing at
least `doc_id`, `sent_id`, `token`, `lid`, `ner`.

```bash
cd scripts
python 01_prepare_dataset.py \
    --input_path ../data/raw/annotations.xlsx \
    --output_dir ../data/processed/lid \
    --label_column lid \
    --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 \
    --seed 42
```

This groups tokens into sentences by `(doc_id, sent_id)`, splits **by
document** (all sentences from a given post stay in the same split, so no
document leaks across train/val/test), and saves:

- a Hugging Face `DatasetDict` (`train` / `validation` / `test`) to
  `data/processed/lid/`
- `label2id.json`, `id2label.json`
- `split_info.json` (which documents went to which split, and sentence counts)

Run this **once**. Reuse the same `--output_dir` for every model so all
three encoders see exactly the same split.

To prepare the (later) NER dataset with the same fixed split, run again with
`--label_column ner --output_dir ../data/processed/ner`.

## 2. Train

One generic script, only `--model_name` and `--output_dir` change between
the three encoder experiments:

```bash
python 02_train_lid.py \
    --model_name dbmdz/bert-base-turkish-cased \
    --data_dir ../data/processed/lid \
    --output_dir ../models/berturk_lid \
    --epochs 10 --batch_size 16 --learning_rate 2e-5 --weight_decay 0.01 --seed 42

python 02_train_lid.py \
    --model_name FacebookAI/xlm-roberta-base \
    --data_dir ../data/processed/lid \
    --output_dir ../models/xlmr_lid \
    --epochs 10 --batch_size 16 --learning_rate 2e-5 --weight_decay 0.01 --seed 42

python 02_train_lid.py \
    --model_name VRLLab/TurkishBERTweet \
    --data_dir ../data/processed/lid \
    --output_dir ../models/turkishbertweet_lid \
    --epochs 10 --batch_size 16 --learning_rate 2e-5 --weight_decay 0.01 --seed 42
```

The dataset stays word-level; the pretrained tokenizer for each model splits
words into subwords, and only the **first subword piece** of each word
carries the real label (`meeting` `'` `e` → `MIXED` `-100` `-100`), which is
the standard Hugging Face approach for token classification. The script
selects and saves the best checkpoint (by validation macro F1).

## 3. Evaluate

Run once per trained model, on the untouched test set, after model
selection is finalized:

```bash
python 03_evaluate_lid.py \
    --model_dir ../models/berturk_lid \
    --data_dir ../data/processed/lid \
    --output_dir ../results/berturk_lid \
    --save_predictions
```

Saves to `results/berturk_lid/`:

- `metrics.json` — accuracy, macro F1 (primary metric), macro precision/recall,
  per-class precision/recall/F1/support
- `confusion_matrix.csv`
- `classification_report.txt`
- `predictions.jsonl` (if `--save_predictions`) — token-level predictions for
  error analysis

Repeat for `xlmr_lid` and `turkishbertweet_lid` with matching `--output_dir`s,
then compare `results/*/metrics.json` across the three encoders (and later
against prompted LLMs).

## Label set (LID)

```
TR=0  EN=1  MIXED=2  OTHER=3  NE=4  AMBIGUOUS=5
```

`NE` is one of the six LID classes (not an auxiliary NER feature) — a named
entity token gets LID label `NE` regardless of its separate `ner` BIO tag
(e.g. `B-ORG`, `B-PER`), which lives in the `ner` column and is only used
when re-running `01_prepare_dataset.py` with `--label_column ner`.
