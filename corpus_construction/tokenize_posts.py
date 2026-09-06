#!/usr/bin/env python3
"""Tokenize collected posts into the one-row-per-token annotation sheet.

    python corpus_construction/tokenize_posts.py \
        --input posts.csv --output posts_for_annotation.tsv

Word-level tokenizer for Turkish-English social media text. URLs, @handles,
hashtags and emoji are kept whole; sentence-final punctuation is split off.

Turkish apostrophe suffixes stay attached to their stem. Trailing punctuation is
stripped from *every* token, including those. 

"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

URL_RE = re.compile(r"(?:https?://|www\.)[^\s]+", re.I)
HANDLE_RE = re.compile(r"@[A-Za-z0-9_]+")
HASHTAG_RE = re.compile(r"#\w+", re.UNICODE)
# Covers the ranges the previous pattern missed: regional-indicator flags
# (U+1F1E6-1F1FF) sit below the old U+1F300 floor, as do the misc-symbols and
# dingbat blocks.
EMOJI_RE = re.compile(
    "[" "\U0001F1E6-\U0001F1FF" "\U0001F300-\U0001FAFF" "\U0001F000-\U0001F0FF"
    "☀-➿" "⬀-⯿" "️" "]+",
    re.UNICODE,
)
# Trailing punctuation that ends a token. Apostrophes are deliberately absent.
TRAILING_PUNCT_RE = re.compile(r"^(.*?)([.!?,;:\"»)\]]+)$")
SENTENCE_END_RE = re.compile(r"^[.!?]+$")

ATOMIC = (URL_RE, HANDLE_RE, HASHTAG_RE, EMOJI_RE)


def tokenize_text(text: str) -> list[str]:
    """Split one post into word tokens."""
    text = unicodedata.normalize("NFC", text)
    for rx in ATOMIC:
        text = rx.sub(lambda m: f" {m.group(0)} ", text)

    tokens: list[str] = []
    for raw in text.split():
        # An atomic unit is kept exactly as it is, except that a URL or
        # hashtag greedily swallows trailing punctuation, so peel that back.
        atomic = next((rx for rx in ATOMIC if rx.fullmatch(raw)), None)
        if atomic is EMOJI_RE or atomic is HANDLE_RE:
            tokens.append(raw)
            continue
        if atomic is not None:
            m = re.match(r"^(.*?)([.!?,;:]+)$", raw)
            if m and m.group(1):
                tokens.append(m.group(1))
                tokens.append(m.group(2))
            else:
                tokens.append(raw)
            continue

        m = TRAILING_PUNCT_RE.match(raw)
        if m and m.group(1):
            tokens.append(m.group(1))
            tokens.append(m.group(2))
        else:
            tokens.append(raw)
    return tokens


def split_sentences(tokens: list[str]) -> list[list[str]]:
    # Break a token list at sentence-final punctuation.
    out, current = [], []
    for tok in tokens:
        current.append(tok)
        if SENTENCE_END_RE.match(tok):
            out.append(current)
            current = []
    if current:
        out.append(current)
    return out or [[]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True,
                    help="CSV with an `entry` column, one post per row")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--text-column", default="entry", dest="text_column")
    ap.add_argument("--no-sentence-split", action="store_true",
                    dest="no_sentence_split",
                    help="put every token of a post in sent_id 1")
    args = ap.parse_args()

    # keep_default_na=False so a post whose entire text is "nan" or "null"
    # is not turned into a missing value
    df = pd.read_csv(args.input, keep_default_na=False, na_values=[])
    if args.text_column not in df.columns:
        raise SystemExit(f"input has no column {args.text_column!r}; "
                         f"found {list(df.columns)}")

    rows, skipped = [], 0
    # Zero-padded to the width the corpus actually needs, so post 1,000 of a
    # 5,500-post corpus sorts after post 999 instead of before it.
    width = max(4, len(str(len(df))))

    for i, text in enumerate(df[args.text_column].astype(str), start=1):
        tokens = tokenize_text(text)
        if not tokens:
            skipped += 1
            continue
        doc_id = f"post_{i:0{width}d}"
        groups = [tokens] if args.no_sentence_split else split_sentences(tokens)
        for sent_id, sent in enumerate([g for g in groups if g], start=1):
            for tok_id, tok in enumerate(sent):
                rows.append({
                    "doc_id": doc_id, "sent_id": sent_id, "tok_id": tok_id,
                    "token": tok, "lid": "", "borrowed_suffix": "", "ner": "",
                })

    if not rows:
        raise SystemExit("no tokens produced — check --text-column")

    out = pd.DataFrame(rows, columns=[
        "doc_id", "sent_id", "tok_id", "token", "lid", "borrowed_suffix", "ner"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sep = "\t" if args.output.suffix.lower() in (".tsv", ".tab") else ","
    out.to_csv(args.output, sep=sep, index=False, encoding="utf-8")

    print(f"{len(df):,} posts -> {out['doc_id'].nunique():,} documents, "
          f"{out.groupby(['doc_id','sent_id']).ngroups:,} sentences, "
          f"{len(out):,} tokens")
    if skipped:
        print(f"skipped {skipped} posts that were empty or whitespace only")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
