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
