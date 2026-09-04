---
license: cc-by-nc-sa-4.0
language:
  - tr
  - en
tags:
  - code-mixing
  - code-switching
  - turkish
  - english
  - LID
  - NER
  - token-classification
task_categories:
  - token-classification
size_categories:
  - 10K<n<100K
configs:
  - config_name: benchmark
    default: true
    data_files:
      - split: train
        path: splits/train.csv
      - split: validation
        path: splits/validation.csv
      - split: test
        path: splits/test.csv
  - config_name: corpus
    data_files:
      - split: train
        path: TurEngMix_Corpus.csv
---

<!--
REPLACES the current dataset card. Two changes need a decision before upload;
both are explained in the repository, not on this page.

1. `configs:` above. The repository currently holds two CSVs with different
   schemas at the root and declares no configs, so Hugging Face's default
   pattern globs both into one `train` split and the load fails on the schema
   mismatch. That is also why the dataset viewer is off. The block above
   declares them as separate configs and publishes the split.

   It expects `splits/{train,validation,test}.csv` — produce them with:
       python scripts/export_hf_splits.py

2. `license:` above is cc-by-nc-sa-4.0, not the cc-by-nc-nd-4.0 currently
   declared. ND forbids distributing adapted material, which blocks the
   re-annotation, re-tokenisation and format conversion that benchmarking
   produces. See DATA_LICENSE. Confirm before uploading.

Delete this comment before publishing.
-->

# TurEngMix

A corpus and benchmark for Turkish–English code-mixed **language
identification** and **named entity recognition**.

Turkish permits an English stem to take a Turkish suffix, producing a single
mixed-language word — *entry'nin*, *link'teki*, *developer'lar*. Models handle
monolingual tokens in code-mixed text reliably and these tokens poorly, which
is what this benchmark measures.

- **`benchmark`** — 15,012 expert-annotated tokens, 250 posts, with a published
  document-level split
- **`corpus`** — ~5,500 code-mixed posts (486,974 tokens), unannotated
- Code: [github.com/ilydoa/TurEngMix](https://github.com/ilydoa/TurEngMix)

## Usage

```python
from datasets import load_dataset

bench = load_dataset("ilydoa/TurEngMix", "benchmark")
corpus = load_dataset("ilydoa/TurEngMix", "corpus")

print(bench["test"][0])
# {'doc_id': 'post_0007', 'sent_id': 1, 'tok_id': 0, 'token': 'yabancı',
#  'lid': 'TR', 'borrowed_suffix': '', 'ner': 'O'}
```

Reading the CSV directly? Pass `keep_default_na=False` — tokens such as `nan`,
`null` and `NA` occur in the text and pandas would otherwise read them as
missing values.

## Schema

One row per token.

| Column | Description |
|:--|:--|
| `doc_id` | `post_NNNN`. One social media post. |
| `sent_id` | Sentence within the post, 1-based. Annotator-assigned. |
| `tok_id` | Token within the sentence, 0-based, dense. |
| `token` | Word form. |
| `lid` | `TR` `EN` `NE` `MIXED` `AMBIGUOUS` `OTHER` |
| `borrowed_suffix` | `MIXED` if the stem and suffix come from different languages, else empty. |
| `ner` | BIO tag over 9 entity types, or `O`. |

`NE` is a language class in its own right: a token belonging to a named entity
takes `lid = NE` regardless of its BIO tag. The two layers are independent.

## Label distribution

**LID**

| TR | EN | NE | MIXED | AMBIGUOUS | OTHER |
|--:|--:|--:|--:|--:|--:|
| 11,751 | 2,016 | 986 | 193 | 62 | 4 |

**NER** — 1,028 entity tokens over `O` = 13,984.

| B-PROD | B-TITLE | I-TITLE | B-ORG | B-PER | B-LOC | I-PER | B-OTHER | I-PROD | I-ORG |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 304 | 122 | 117 | 100 | 74 | 66 | 49 | 49 | 46 | 29 |

| I-LOC | B-GROUP | I-EVENT | I-GROUP | I-OTHER | B-EVENT | B-TIME | I-TIME |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 16 | 15 | 14 | 11 | 8 | 5 | 3 | 0 |

`B-EVENT` (5), `B-TIME` (3) and `I-TIME` (0) are too rare to support a
per-class score. `OTHER` has 4 tokens in the whole benchmark.

## Split

Split by document, so no post contributes tokens to two splits. Use this split
when reporting on TurEngMix — results computed on a different partition are
not comparable to the paper's.

| Split | Documents | Sentences | Tokens |
|:--|--:|--:|--:|
| train | 200 | 254 | 12,466 |
| validation | 25 | 26 | 1,173 |
| test | 25 | 41 | 1,373 |

The test set is 25 documents. Differences of a point or two between systems
are inside the noise of a set this size.

## How it was built

Posts were collected from [Ekşi Sözlük](https://eksisozluk.com), a Turkish
social platform, across ~100 topics spanning technology, entertainment,
sports, lifestyle and daily life. Candidates were filtered in two stages:
`langdetect` applied per word as a recall-oriented pre-filter, then two GPT-4o
prompts whose agreement decided inclusion. Surviving posts were tokenized at
the word level and annotated by expert annotators for both layers.

Collection and filtering code is in `corpus_construction/` in the GitHub
repository. It reads a live website and cannot reproduce this corpus; it
documents the method.

## Known annotation issues

Three ill-formed BIO transitions (an `I-` tag with no matching `B-`) and 66
tokens where the LID and NER layers disagree about whether the token is part
of a named entity. Both are enumerated in `normalization_report.json` in the
repository. They are documented rather than silently repaired, because a
scorer that quietly fixes them changes the gold entity count.

## Intended use

Non-commercial research: benchmarking LID and NER on code-mixed text,
computational and sociolinguistic study of Turkish–English code-mixing.

**Not for**: commercial use; training models for commercial deployment;
attempting to identify the authors of the source posts.

## Ethical considerations

**PII.** No personally identifying information was deliberately collected or
annotated, and no attempt was made to identify authors. The text is verbatim
public social media, so usernames, @handles and self-disclosed personal
details may appear in it, and it has not been scrubbed. Treat it as
un-redacted public social media text rather than a de-identified corpus.

**Bias.** Ekşi Sözlük skews young, urban and technically literate. Code-mixing
patterns here are not representative of Turkish–English contact generally.
Models trained on it inherit the platform's demographic and topical skew.

## Citation

See `CITATION.cff` in the repository. BibTeX to follow on acceptance.

## Contact

idogan@umd.edu · nlpa@umd.edu
