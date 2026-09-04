# TurEngMix

Corpus, benchmark and code for **TurEngMix: A Text Corpus and Benchmark for
Turkish–English Code-Mixed Language Identification and Named Entity
Recognition**.

Turkish lets an English stem take a Turkish suffix, producing a single
mixed-language word — *influencerlarımız*, *frameworklerinin*, *link'teki*.
Models label monolingual tokens in code-mixed text reliably and these tokens
poorly. That gap is what the benchmark measures.

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

The benchmark and the split are committed, so this works with no API key and
no GPU. Among the 28 tests, two check the artifact against the paper directly:
the split reproduces **Table 7** exactly (200/25/25 posts, 254/26/41
sentences, 12,466/1,173/1,373 tokens), and the majority-class baselines
reproduce **Table 9** and **Table 10** exactly (0.1414 and 0.0502).

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
for the token budget so long posts come back truncated. That scores as a model
error when it is a budget error; `run_llm.py` logs `finish_reason` per call so
the two stay distinguishable. And **`--seeds 1 2 3 4`** for encoders, because
§5.2 reports mean ± SD over four seeds and a single-seed number is not
comparable to the published table.

Encoders were trained on one A6000, about six minutes per run. There are no
cluster job files here — they encoded one account's paths and nobody else
could run them. These commands are what those jobs wrapped.

## How the numbers are defined

Every metric is in `turengmix/scoring.py`, and each docstring names the table
it produces. The definitions were confirmed by reproducing the published
baselines, which pins down the evaluation scope and the macro denominator at
once. Four decisions are load-bearing:

**Macro averages use the closed label set** — six LID classes, nineteen BIO
labels — not the labels present in a given prediction file. A model that never
predicts `OTHER` is still averaged over it, so two models' macro-F1 stay on
the same scale.

**Evaluation scope differs by model family.** Decoders were run over the
complete benchmark (250 posts); encoders were evaluated on the test split.
`report.py` applies the right scope per run rather than one flag globally.

**Malformed output counts as incorrect**, as in §5.1. Unparseable predictions
become `UNK`, which is not in the label set, so it earns no credit and costs
the gold class its recall. The rate appears in every table, and the confusion
matrix carries a real `UNK` column so rows still sum to class support.

**Partial output is kept.** When a model labels 90 of 100 tokens, those 90 are
scored and the rest are `UNK` — one policy for every model, so error rates are
comparable across them.

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
tests/                  28 tests
```

Predictions are keyed by `(doc_id, sent_id, tok_id)` throughout, so a fresh
run, a resumed run and a partial run produce the same file layout, and a
misaligned prediction cannot be written silently. Encoder and LLM runs emit
the same prediction format, so one scorer handles both — which is what lets
their numbers share a table.

## Notes for anyone building on this

**Prediction files are not committed.** Regenerating them needs an API key, a
GPU and about a day. If you have the CSVs from the paper's runs, dropping them
in `results/predictions/` makes `python scripts/report.py` reproduce every
results table in seconds on any laptop. See the last section of
`docs/PAPER_MAPPING.md`.

**The released file differs from Tables 4 and 5 by fourteen tokens.**
`normalize_annotations.py` prints the deltas and records them in its report.
Table 4's `NE = 793` is the 986 NE tokens minus the 193 that are also
morphologically integrated, which is what its caption alludes to; the rest is
`TR` +2, `EN` +1, `TITLE` −7, `PER` +4, `O` +3.

**Sixty-nine tokens have open annotation questions** — three ill-formed BIO
transitions and 66 places where the LID and NER layers disagree about whether
a token is part of a named entity, against the rule in §4.1.1 that entity
tokens take `NE`. They are listed in `data/normalization_report.json` and
deliberately not auto-repaired.

**The test set is 25 documents** (1,373 tokens, 25 of them morphologically
integrated), and the paper's Limitations section already says several
categories have too few instances for reliable per-category evaluation. The
four-seed spread in Table 9 is the right thing to read alongside any encoder
comparison.

**`OTHER` has 4 tokens and `I-TIME` has none** in the whole benchmark.
Per-class figures for the rarest labels should not be read as estimates.

## Citation

See `CITATION.cff`.

## Licence

Code is MIT (`LICENSE`). The corpus and annotations are released under
`DATA_LICENSE`; see `data/README.md` for the schema, label distributions and
the PII note.

## Contact

idogan@umd.edu · nlpa@umd.edu
