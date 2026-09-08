# TurEngMix

Corpus, benchmark and code for **TurEngMix: A Text Corpus and Benchmark for
Turkish–English Code-Mixed Language Identification and Named Entity
Recognition**.

- **Benchmark** — 15,012 expert-annotated tokens, 250 posts, with a published
  document-level split
- **Corpus** — 5,549 posts, 486,974 tokens, from Ekşi Sözlük
- **Baselines** — GPT-4o and Qwen3-8B (zero- and three-shot), plus fine-tuned
  XLM-RoBERTa, BERTurk and TurkishBERTweet

Dataset: [huggingface.co/datasets/ilydoa/TurEngMix](https://huggingface.co/datasets/ilydoa/TurEngMix)

---

## Quickstart

```bash
git clone https://github.com/ilydoa/TurEngMix && cd TurEngMix
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```
Rebuild the benchmark from the raw annotation export:

```bash
python scripts/normalize_annotations.py    # -> data/benchmark.csv
python scripts/make_splits.py              # -> data/splits/
```

## Reproducing the results

**[`docs/PAPER_MAPPING.md`](docs/PAPER_MAPPING.md) maps every table in the
paper to the script that produces it**, records the metric definitions, and
lists what is not reproducible and why. Start there.

The short version — produce predictions, then score them:

```bash
export OPENAI_API_KEY=...  OPENROUTER_API_KEY=...

python scripts/run_llm.py --task lid --model gpt-4o --shots 3
python scripts/run_llm.py --task ner --model qwen/qwen3-8b --shots 3 \
    --provider openrouter --disable-reasoning

python scripts/train_encoder.py --task lid \
    --model dbmdz/bert-base-turkish-cased --out models/berturk_lid \
    --seeds 1 2 3 4 --fp16
python scripts/evaluate_encoder.py --task lid --model-dir models/berturk_lid_seed1

python scripts/report.py
```

`report.py` writes `results/table08…table12` and `tableA3`/`tableA4` as
Markdown, plus `metrics.json`, a per-run classification report and a confusion
matrix. `--dry-run` on `run_llm.py` prints the exact prompt without spending
anything; `--limit 5` is a cheap smoke test.

Two flags matter. **`--disable-reasoning` for Qwen3** — it is a hybrid
reasoning model, and left on, `<think>` content competes with the label list
for the token budget so long posts come back truncated. And **`--seeds 1 2 3 4`** for encoders, because
§5.2 reports mean ± SD over four seeds.

Encoders were trained on one A6000, about six minutes per run.

## How the numbers are defined

Every metric is in `turengmix/scoring.py`, and each docstring names the table
it produces. 

**Macro averages use the closed label set** — six LID classes, nineteen BIO
labels — not the labels present in a given prediction file. A model that never
predicts `OTHER` is still averaged over it, so two models' macro-F1 stay on
the same scale.

**Evaluation scope differs by model family.** Decoders were run over the
complete benchmark (250 posts); encoders were evaluated on the test split.
`report.py` applies the right scope.

**Malformed output counts as incorrect**, as in §5.1. Unparseable predictions
become `UNK`, which is not in the label set, so it earns no credit and costs
the gold class its recall. The rate appears in every table, and the confusion
matrix carries a real `UNK` column so rows still sum to class support.

**Partial output is kept.** When a model labels 90 of 100 tokens, those 90 are
scored and the rest are `UNK`.

## Layout

```
turengmix/              one implementation of each thing
  labels.py             closed label sets for both tasks
  data.py               loading, invariant checks, split, prediction writing
  prompting.py          input format and output parser, shared by every model
  scoring.py            every metric, keyed to the table it produces
scripts/
  normalize_annotations.py   raw export -> data/benchmark.csv, changes logged
  make_splits.py             writes the committed split (verifies Table 7)
  run_llm.py                 prompted baselines, both tasks, both providers
  train_encoder.py           fine-tuning, both tasks, multi-seed
  evaluate_encoder.py        encoder predictions, same format as the LLM runs
  report.py                  predictions -> the paper's tables
  export_hf_splits.py        builds the Hugging Face upload
prompts/                {lid,ner}_{zero,few}_shot.txt  (Appendix A.3)
data/
  raw/annotations_raw.csv    original export, untouched
  benchmark.csv              canonical
  splits/                    committed doc_id lists
  normalization_report.json  every change, open issues, drift vs Tables 4-5
corpus_construction/    how the corpus was collected (Appendix A.1-A.2)
docs/PAPER_MAPPING.md   table-by-table mapping
```

Predictions are keyed by `(doc_id, sent_id, tok_id)` throughout, so a fresh
run, a resumed run and a partial run produce the same file layout. Encoder and LLM runs emit
the same prediction format, so one scorer handles both.

## Citation

See `CITATION.cff`.

## Licence

Code is MIT (`LICENSE`). The corpus and annotations are released under
`DATA_LICENSE`; see `data/README.md` for the schema, label distributions and
the PII note.

## Contact

idogan@umd.edu · nlpa@umd.edu
