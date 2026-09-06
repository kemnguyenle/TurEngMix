#!/usr/bin/env python3
"""Filter collected entries down to the code-mixed ones.

    python corpus_construction/filter_code_mixed.py \
        --input data/raw/entries.csv --output data/raw/entries_code_mixed.csv

Two stages:

1. `langdetect`, as a cheap recall-oriented pre-filter.
2. Two GPT-4o prompts, run over the survivors, whose agreement decides
   inclusion.

Both stages write every intermediate decision to the output file rather than
only the survivors, so the selection is auditable after the fact.

"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

PROMPT_BASIC = """\
Detect if the given Turkish social media post contains any English words.

Output Format
Output "True" if English words are present, otherwise "False".
Number your answers to match the post numbers, and put each response on its own line.

Example output:
1. True
2. False

Posts:
{posts}
"""

PROMPT_STRICT = """\
You will be given Turkish social media posts. Go through each post word by
word and detect whether it contains any English words mixed into the Turkish,
including individual English words or whole phrases in English. Do not count
URLs or brand names as English words.

Output Format
Output "True" if English words are present, otherwise "False".
Number your answers to match the post numbers, and put each response on its own line.

Example output:
1. True
2. False

Posts:
{posts}
"""


def langdetect_is_code_mixed(text: str) -> bool:
    """True if per-word detection finds both Turkish and English."""
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0          # without this, results vary between runs
    if not isinstance(text, str) or not text.strip():
        return False
    langs = set()
    for word in WORD_RE.findall(text):
        try:
            langs.add(detect(word))
        except LangDetectException:
            # A word too short or too ambiguous to profile
            continue
    return "en" in langs and "tr" in langs


def ask_batch(client, model: str, template: str, posts: list[str],
              retries: int = 4) -> list[bool | None]:
    """Run one prompt over a batch. Returns None for any post left unanswered.
    """
    numbered = "\n".join(f"{i}. {p}" for i, p in enumerate(posts, 1))
    prompt = template.format(posts=numbered)

    text = ""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0, seed=42,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.choices[0].message.content or ""
            break
        except Exception as e:
            if attempt == retries - 1:
                print(f"  batch failed after {retries} attempts: {e}", file=sys.stderr)
                return [None] * len(posts)
            time.sleep(min(2 ** attempt, 30))

    answers: list[bool | None] = []
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for i in range(1, len(posts) + 1):
        line = next((l for l in lines if l.startswith(f"{i}.")), None)
        if line is None:
            answers.append(None)
            continue
        answers.append(line.split(".", 1)[1].strip().lower().startswith("true"))
    return answers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--text-column", default="entry", dest="text_column")
    ap.add_argument("--model", default="gpt-4o")
    ap.add_argument("--batch-size", type=int, default=50, dest="batch_size")
    ap.add_argument("--skip-llm", action="store_true", dest="skip_llm",
                    help="run only the langdetect stage")
    args = ap.parse_args()

    df = pd.read_csv(args.input, keep_default_na=False, na_values=[])
    print(f"loaded {len(df):,} entries")

    print("stage 1: langdetect")
    df["langdetect_code_mixed"] = [
        langdetect_is_code_mixed(t) for t in df[args.text_column].astype(str)
    ]
    n1 = int(df["langdetect_code_mixed"].sum())
    print(f"  {n1:,} of {len(df):,} entries flagged ({n1 / max(1, len(df)):.1%})")

    if args.skip_llm:
        df.to_csv(args.output, index=False)
        print(f"wrote {args.output}")
        return 0

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise SystemExit("OPENAI_API_KEY is not set.")
    from openai import OpenAI
    client = OpenAI(api_key=key)

    candidates = df[df["langdetect_code_mixed"]].copy()
    posts = candidates[args.text_column].astype(str).tolist()

    # Checkpoint per batch
    ckpt = args.output.with_suffix(".ckpt.json")
    basic: list[bool | None] = []
    strict: list[bool | None] = []
    if ckpt.exists():
        saved = json.loads(ckpt.read_text(encoding="utf-8"))
        basic, strict = saved["basic"], saved["strict"]
        print(f"resuming after {len(basic)} posts")

    print(f"stage 2: {args.model}, two prompts, batches of {args.batch_size}")
    for start in range(len(basic), len(posts), args.batch_size):
        batch = posts[start:start + args.batch_size]
        basic += ask_batch(client, args.model, PROMPT_BASIC, batch)
        strict += ask_batch(client, args.model, PROMPT_STRICT, batch)
        ckpt.write_text(json.dumps({"basic": basic, "strict": strict}),
                        encoding="utf-8")
        print(f"  {min(start + len(batch), len(posts))}/{len(posts)}")

    candidates["gpt_basic"] = basic
    candidates["gpt_strict"] = strict
    candidates["code_mixed"] = [
        bool(b) and bool(s) for b, s in zip(basic, strict)
    ]
    unanswered = sum(b is None or s is None for b, s in zip(basic, strict))

    merged = df.merge(
        candidates[["gpt_basic", "gpt_strict", "code_mixed"]],
        left_index=True, right_index=True, how="left",
    )
    merged["code_mixed"] = merged["code_mixed"].fillna(False)
    merged.to_csv(args.output, index=False)
    ckpt.unlink(missing_ok=True)

    n2 = int(merged["code_mixed"].sum())
    agree = int((candidates["gpt_basic"] == candidates["gpt_strict"]).sum())
    print(f"\nboth prompts said yes : {n2:,}")
    print(f"prompts agreed        : {agree:,} of {len(candidates):,} "
          f"({agree / max(1, len(candidates)):.1%})")
    if unanswered:
        print(f"posts left unanswered : {unanswered} (recorded as null, not dropped)")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
