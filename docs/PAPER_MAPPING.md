# Paper → code

Every table and reported figure, and what produces it.

## Tables

| Paper | What it reports | Produced by | Output |
|:--|:--|:--|:--|
| Table 2 | Corpus statistics | `corpus_construction/` | not reproducible — see below |
| Table 4 | LID label distribution | `scripts/normalize_annotations.py` | `data/normalization_report.json` → `lid_distribution` |
| Table 5 | NER label distribution | `scripts/normalize_annotations.py` | `data/normalization_report.json` → `ner_by_entity_type` |
| Table 6 | Code-mixing pattern counts | — | not in the released columns; see note below |
| Table 7 | Split sizes | `scripts/make_splits.py` | `data/splits/*.txt` |
| Table 8 | Per-class F1, LID | `scripts/report.py` | `results/table08_lid_per_class.md` |
| Table 9 | Macro F1, LID | `scripts/report.py` | `results/table09_lid_macro.md` |
| Table 10 | BIO-level macro F1, NER | `scripts/report.py` | `results/table10_ner_bio_macro.md` |
| Table 11 | Binary NE / Non-NE F1 | `scripts/report.py` | `results/table11_ner_binary.md` |
| Table 12 | Error rates on integrated tokens | `scripts/report.py` | `results/table12_integration.md` |
| Table A1–A2 | Inter-annotator agreement | — | computed during annotation; not in this repo |
| Table A3 | Per-type NER F1, decoders | `scripts/report.py` | `results/tableA3_ner_per_type.md` |
| Table A4 | Per-type NER F1, encoders | `scripts/report.py` | `results/tableA4_ner_per_type.md` |
| Table A5 | Example errors | `results/*_report.txt`, prediction CSVs | qualitative |
| A.1.1 | Topic tags | `corpus_construction/scrape_eksisozluk.py` | `TOPICS` |
| A.2.1–A.2.2 | Filtering prompts | `corpus_construction/filter_code_mixed.py` | `PROMPT_BASIC`, `PROMPT_STRICT` |
| A.3.1–A.3.2 | LID / NER prompts | `prompts/` | four files |

## Metric definitions

Confirmed by reproducing the published majority-class baselines exactly —
**LID 0.1414** and **NER 0.0502** (Tables 9 and 10). Those two numbers pin
down the evaluation scope and the macro denominator simultaneously; if either
were defined differently they would not match. `pytest` asserts both.

| Decision | Value | Where |
|:--|:--|:--|
| LID macro denominator | 6 classes, closed set | `turengmix/labels.py` |
| NER macro denominator | 19 BIO labels, closed set | `turengmix/labels.py` |
| Baseline | majority class, test split | `scoring.majority_baseline` |
| Decoder eval scope | complete benchmark, 250 posts | §5.1 |
| Encoder eval scope | test split | §5.2 |
| Malformed output | counted as incorrect | §5.1 |
| Encoder aggregation | mean ± SD over 4 seeds | §5.2 |
| Table A3/A4 classes | 10, with B-/I- collapsed | `scoring.ner_type_metrics` |

`report.py` picks the scope per run: filenames containing `berturk`, `xlm`,
`roberta`, `bertweet` or `encoder` are scored on the test split, everything
else on the full benchmark. Override it with `"scope": "test"` or `"all"` in a
run's `.meta.json`.

## Reproducing the numbers

```bash
# LLM baselines — 4 conditions per model
for task in lid ner; do for shots in 0 3; do
  python scripts/run_llm.py --task $task --model gpt-4o --shots $shots
  python scripts/run_llm.py --task $task --model qwen/qwen3-8b --shots $shots \
      --provider openrouter --disable-reasoning
done; done

# Encoders — 3 models x 2 tasks x 4 seeds
for m in dbmdz/bert-base-turkish-cased FacebookAI/xlm-roberta-base VRLLab/TurkishBERTweet; do
  for task in lid ner; do
    name=$(basename $m)_$task
    python scripts/train_encoder.py --task $task --model $m \
        --out models/$name --seeds 1 2 3 4 --fp16
    for s in 1 2 3 4; do
      python scripts/evaluate_encoder.py --task $task --model-dir models/${name}_seed$s
    done
  done
done

python scripts/report.py
```

## Known differences between the released file and the published tables

`normalize_annotations.py` prints these and records them under
`comparison_with_published_tables` in its report. They are small, and they
are stated here rather than left for a reader to discover.

**Table 4 (LID).** The published `NE = 793` is the benchmark's 986 NE tokens
minus the 193 that are also morphologically integrated — which is what the
caption's "NE tokens included additional morphologically mixed examples"
refers to. That is why the published table sums to 14,816 rather than its
stated total of 15,012. Beyond that, `TR` differs by +2 and `EN` by +1.

**Table 5 (NER).** `TITLE` −7, `PER` +4, `O` +3 against the released file.
The deltas net to zero, which is consistent with a small re-annotation
between computing the table and exporting the file.

Fourteen tokens in total. Deciding what to do about them is an author call:
note it, re-derive the tables from the released file, or locate the exact
export the tables were computed from. Nothing in the code depends on the
resolution — the counts are recomputed from whatever file is present.

**Table 6** counts posts by code-mixing pattern (embedded mixed token 165,
isolated English token 152, embedded English phrase 154). Those are
post-level annotations that are not among the released token-level columns.
If the post-level labels exist, adding them to the benchmark or shipping them
as a second file would make Table 6 reproducible too.

## What is not reproducible, and why

**The corpus.** `corpus_construction/` reads a live website whose contents
change, so a run today returns different posts. It documents the method. The
released annotations are the reproducible artifact.

**Sentence boundaries.** `sent_id` was assigned by the annotators.
`tokenize_posts.py` splits on `.!?`, which does not recover those boundaries
in noisy social media text. Use it for new data.

**Inter-annotator agreement** (Tables A1–A2) was computed during annotation
on a 203-token sample; the second annotator's labels are not in the repo.
Shipping that sample would make the κ figures checkable.

## The one thing that would most improve this artifact

**Commit the prediction files.** `results/predictions/*.csv` is gitignored by
default because the files are regenerable — but regenerating them costs an
API key, a GPU and about a day. If the prediction CSVs from the runs behind
the paper are committed instead, then `python scripts/report.py` reproduces
every results table in seconds, on any laptop, with no key and no GPU. That
is the difference between an artifact a reviewer can check and one they have
to take on trust.

To do it: drop the CSVs into `results/predictions/`, remove the matching
`results/predictions/*.csv` line from `.gitignore`, and commit.
