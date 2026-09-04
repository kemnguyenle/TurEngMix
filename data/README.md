# TurEngMix benchmark

15,012 tokens · 250 documents · 321 sentences · one row per token.

| File | What it is |
|:--|:--|
| `raw/annotations_raw.csv` | the annotators' export, untouched |
| `benchmark.csv` | canonical version — **use this one** |
| `splits/{train,validation,test}.txt` | committed `doc_id` lists |
| `normalization_report.json` | every change from raw to canonical, plus open issues |

`benchmark.csv` is produced by `scripts/normalize_annotations.py`. The raw
export is kept so the two can be diffed.

## Schema

| Column | Type | Description |
|:--|:--|:--|
| `doc_id` | str | One social media post. Left exactly as exported, matching the identifiers already on the Hub; the export uses two numbering schemes (`post_0001_1`, `post_007_47`). |
| `doc_num` | int | The trailing number, which is unique across all 250 documents. Use this to sort or join. |
| `sent_id` | int | Sentence within the post, 1-based, contiguous. Annotator-assigned. |
| `tok_id` | int | Token within the sentence, 0-based, dense. A position, not a stable identifier across releases. |
| `token` | str | The word form. |
| `lid` | str | Language label. One of the six below. |
| `borrowed_suffix` | str | `MIXED` if the token carries a suffix from the other language on its stem; otherwise empty. |
| `ner` | str | BIO tag. `O` where the token is outside every entity. |
| `source_tok_id` | int | The export's original `tok_id`, before gaps were closed. |

Read it with `keep_default_na=False`. Tokens such as `nan`, `null` and `NA`
occur in the text and pandas would otherwise turn them into missing values.
`turengmix.data.load_benchmark()` does this and checks the invariants.

## Labels

**LID** — flat, six classes. `NE` is a language class in its own right: a token
belonging to a named entity takes `NE` regardless of its BIO tag.

| TR | EN | NE | MIXED | AMBIGUOUS | OTHER |
|--:|--:|--:|--:|--:|--:|
| 11,751 | 2,016 | 986 | 193 | 62 | 4 |

`MIXED` marks a token whose stem and suffix come from different languages —
*entry'nin*, *link'teki*. `AMBIGUOUS` is a token that could belong to either
language in context. `OTHER` has 4 tokens in the entire benchmark; a per-class
score for it is not meaningful.

**NER** — BIO over nine entity types.

| O | B-PROD | B-TITLE | I-TITLE | B-ORG | B-PER | B-LOC | I-PER | B-OTHER | I-PROD |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 13,984 | 304 | 122 | 117 | 100 | 74 | 66 | 49 | 49 | 46 |

| I-ORG | I-LOC | B-GROUP | I-EVENT | I-GROUP | I-OTHER | B-EVENT | B-TIME | I-TIME |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 29 | 16 | 15 | 14 | 11 | 8 | 5 | 3 | 0 |

`B-EVENT` (5), `B-TIME` (3) and `I-TIME` (0) are too rare to support a
per-class figure. `I-TIME` is in the label set but never observed.

## Split

Split by document, so no post contributes tokens to two splits. Committed
rather than regenerated: `random.shuffle` is not stable across Python
versions. This exact partition reproduces Table 7 of the paper, and
`scripts/make_splits.py` verifies that on every run.

| Split | Documents | Sentences | Tokens | Integrated tokens |
|:--|--:|--:|--:|--:|
| train | 200 | 254 | 12,466 | 317 |
| validation | 25 | 26 | 1,173 | 44 |
| test | 25 | 41 | 1,373 | 25 |

**The test set is small.** 25 documents, and 25 morphologically integrated
tokens in it — the class the paper is about. Differences of a point or two
between models are inside the noise. See the caveat in the top-level README.

Sentence length: median 26 tokens, 90th percentile 106, maximum 373. The long
tail matters: at 512 subwords, four to five sentences truncate under BERTurk
and XLM-R. `train_encoder.py` and `evaluate_encoder.py` report truncation
rather than letting it vanish into the score.

## Known issues, open

Listed in full in `normalization_report.json`. These need an annotator's
decision and were deliberately **not** auto-repaired.

**Three ill-formed BIO transitions** — an `I-` tag with no matching `B-`:
`fm` → `I-ORG` sentence-initially (twice, same token), and `halloween'i` →
`I-EVENT` after an `O`. Span-based scorers repair these silently, so the gold
entity count shifts depending on which scorer a reader runs. The metrics in
this repository are token-level, matching the paper, so they are unaffected.

**66 LID/NER disagreements.** 54 tokens carry an entity tag but a `lid` other
than `NE` (26 `TR`, 24 `EN`, 4 `MIXED`); 12 have `lid == NE` with no entity
tag. Some may be deliberate — a morphologically integrated entity is arguably
`MIXED` — but the documented rule says entity tokens take `NE`, so either the
rule or these rows should change.

**`borrowed_suffix` and `lid` are near-redundant.** All 386 integrated tokens
split exactly 193 `MIXED` / 193 `NE`, and every `lid == MIXED` token is
integrated. The exact 193/193 balance is worth confirming as a real annotation
pattern rather than an artifact of how the column was filled.

## Differences from Tables 4 and 5 of the paper

`scripts/normalize_annotations.py` compares the file against the published
counts on every run and records the result under
`comparison_with_published_tables`. Fourteen tokens differ.

**Table 4 (LID).** The published `NE = 793` is this file's 986 NE tokens minus
the 193 that are also morphologically integrated — which the caption's "NE
tokens included additional morphologically mixed examples" refers to. That is
why the published table sums to 14,816 rather than its stated total of 15,012.
Beyond that: `TR` +2, `EN` +1.

**Table 5 (NER).** `TITLE` −7, `PER` +4, `O` +3. The deltas net to zero,
consistent with a small re-annotation between computing the table and
exporting the file.

Nothing in the code depends on how this is resolved — every count is
recomputed from whatever file is present.

## What was fixed

From `normalization_report.json`:

- 1,048 blank `ner` cells set to `O` — the raw export could not distinguish
  "outside an entity" from "not annotated"
- one `borrowed_suffix` cell held `" MIXED"` with a leading space, invisible to
  any `== "MIXED"` filter
- two rows shared a `(doc_id, sent_id, tok_id)` key
- 1,404 missing `tok_id` values from gaps left by cleaning; renumbered densely
- the exported row-number column was dropped

`doc_id` was deliberately **not** renumbered, despite the two schemes in the
export, so that the GitHub and Hugging Face releases stay identifier-
compatible. `doc_num` carries the unique trailing number.

## Licence

See `../DATA_LICENSE`. The corpus is derived from public Ekşi Sözlük posts.

**PII.** No personally identifying information was deliberately collected or
annotated, and no attempt was made to identify authors. The source text is
verbatim public social media, so usernames, @handles and self-disclosed
personal details may appear in it. It has not been scrubbed. Anyone making an
IRB or GDPR determination should treat it as un-redacted public social media
text, not as a de-identified corpus.
