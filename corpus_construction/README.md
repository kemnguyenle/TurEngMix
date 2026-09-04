# Corpus construction

How the TurEngMix corpus was collected. **These scripts are documentation, not
a reproduction path.** They read a live website whose contents change, so a run
today returns different posts from ours. To reproduce the paper's numbers, use
the released annotations in `../data/` — nothing here needs to run.

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

## Why this is not reproducible, precisely

**The source changes.** Ekşi Sözlük entries are added, edited and deleted.

**`langdetect` is applied per word.** That is close to the least reliable way
to use a profile-based detector. It is kept because it is what produced the
released corpus, and it is why stage 1 is treated as a recall-oriented
pre-filter with stage 2 doing the real work. It is now seeded
(`DetectorFactory.seed = 0`); it was not before, so the original filtering was
non-deterministic even given identical input.

**`sent_id` is not reproduced.** Sentence boundaries in the released benchmark
were assigned by the annotators. `tokenize_posts.py` splits on `.!?`, which
does not recover those boundaries in noisy social media text. Use it for new
data, not to rebuild the benchmark.

To make a future corpus reconstructible, record the retained entry IDs
alongside the text — `scrape_eksisozluk.py` writes an `entry_id` column for
this purpose.

## Tokenizer

Word-level. URLs, @handles, hashtags and emoji are kept whole; trailing
punctuation is split off every token.

Turkish apostrophe suffixes stay attached to their stem — `Google'e` is one
token, because an English stem plus a Turkish suffix being a single word is the
entire subject of this benchmark. The earlier version suppressed punctuation
splitting for tokens containing an apostrophe and ending in `e`, so `Google'e.`
kept its period while `Spotify'da.` was split in two.

`doc_id` is `post_NNNN` from the input row number, zero-padded to the width the
corpus needs, so skipped (empty) posts leave a gap and every `doc_id` traces
back to its source row.
