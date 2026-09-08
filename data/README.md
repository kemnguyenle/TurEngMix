# TurEngMix benchmark

15,012 tokens · 250 documents · 321 sentences · one row per token.

| File | What it is |
|:--|:--|
`raw/annotations_raw.csv`  | original annotation export
`benchmark.csv`            | canonical released benchmark
`splits/{train,validation,test}.txt` | committed document-level split

## Schema

| Column | Type | Description |
|:--|:--|:--|
| `doc_id` | str | One social media post.. |
| `doc_num` | int | The trailing number, which is unique across all 250 documents. Use this to sort or join. |
| `sent_id` | int | Sentence within the post, 1-based, contiguous. |
| `tok_id` | int | Token within the sentence, 0-based, dense. |
| `token` | str | The word form. |
| `lid` | str | Language label. One of the six below. |
| `borrowed_suffix` | str | `MIXED` if the token carries a suffix from the other language on its stem; otherwise empty. |
| `ner` | str | BIO tag. `O` where the token is outside every entity. |
| `source_tok_id` | int | The export's original `tok_id`. |

Read it with `keep_default_na=False`. Tokens such as `nan`, `null` and `NA`
occur in the text and pandas would otherwise turn them into missing values.
`turengmix.data.load_benchmark()` does this and checks the invariants.

## Labels

**LID** — flat, six classes. `NE` is a language class in its own right: a token
belonging to a named entity takes `NE` regardless of its BIO tag.

| TR | EN | NE | MIXED | AMBIGUOUS | OTHER |
|--:|--:|--:|--:|--:|--:|
| 11,723 | 1,992 | 1,042 | 189 | 62 | 4 |

`MIXED` marks a token whose stem and suffix come from different languages —
*entry'nin*, *link'teki*. `AMBIGUOUS` is a token that could belong to either
language in context. 

**NER** — BIO over nine entity types.

| O | B-PROD | B-TITLE | I-TITLE | B-ORG | B-PER | B-LOC | I-PER | B-OTHER | I-PROD |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 12,923 | 313 | 124 | 122 | 100 | 73 | 70 | 46 | 46 | 46 |

| I-ORG | I-LOC | B-GROUP | I-EVENT | I-GROUP | I-OTHER | B-EVENT | B-TIME | I-TIME |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 29 | 18 | 15 | 14 | 11 | 6 | 5 | 3 | 0 |

## Split

Split by document, so no post contributes tokens to two splits. 

| Split | Documents | Sentences | Tokens | Integrated tokens |
|:--|--:|--:|--:|--:|
| train | 200 | 254 | 12,466 | 317 |
| validation | 25 | 26 | 1,173 | 44 |
| test | 25 | 41 | 1,373 | 25 |


## Licence

See `../DATA_LICENSE`. The corpus is derived from public Ekşi Sözlük posts.

**PII.** No personally identifying information was deliberately collected or
annotated, and no attempt was made to identify authors. The source text is
verbatim public social media, so usernames, @handles and self-disclosed
personal details may appear in it.
