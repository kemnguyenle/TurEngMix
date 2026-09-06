# Corpus construction

How the TurEngMix corpus was collected. **These scripts are documentation, not
a reproduction path.** They read a live website whose contents change, so a run
today returns different posts from ours. 

```bash
pip install -r ../requirements-corpus.txt
```

## Pipeline

```bash
# 1. Collect entries for the topic list. Deduplicated on content; the output
#    is rewritten, not appended, so a restarted run does not double the corpus.
python scrape_eksisozluk.py --out entries.csv

# 2. Filter to code-mixed posts: langdetect, then two GPT-4o prompts whose
#    agreement decides inclusion. Checkpoints per batch.
export OPENAI_API_KEY=...
python filter_code_mixed.py --input entries.csv --output entries_code_mixed.csv

# 3. Tokenize into the one-row-per-token annotation sheet.
python tokenize_posts.py --input entries_code_mixed.csv \
    --output posts_for_annotation.tsv
```

Stage 2 writes a decision for every entry, not only the survivors, so the
selection is auditable afterwards.

## Tokenizer

Word-level. URLs, @handles, hashtags and emoji are kept whole; trailing
punctuation is split off every token.

Turkish apostrophe suffixes stay attached to their stem — `Google'e` is one
token, because an English stem plus a Turkish suffix being a single word is a
key subject of the benchmark. 

`doc_id` is `post_NNNN` from the input row number, zero-padded to the width the
corpus needs, so skipped (empty) posts leave a gap and every `doc_id` traces
back to its source row.
