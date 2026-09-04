#!/usr/bin/env python3
"""Collect Ekşi Sözlük entries for the topic list used to build TurEngMix.

    python corpus_construction/scrape_eksisozluk.py --out data/raw/entries.csv

This script cannot reproduce the original corpus. It reads a live website
whose contents change, so a run today returns different entries from our run.
It is released to document *how* the corpus was collected, not as a step in
reproducing the paper's numbers — for that, use the released annotation file.

Differences from the original: entries are deduplicated on (topic, text) and
the output file is rewritten rather than appended, so an interrupted run that
is restarted does not silently double the corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import random
import sys
from pathlib import Path

import pandas as pd

TOPICS = [
    # Tech / programming / AI
    "spotify", "google", "python", "ai", "chatgpt", "openai", "midjourney", "dalle",
    "javascript", "react", "nodejs", "git", "blockchain", "web3", "crypto", "nft",
    "startup", "venture capital",
    # Social media / internet culture
    "facebook", "instagram", "twitter", "youtube", "tiktok", "linkedin", "discord",
    "reddit", "influencer", "meme", "shitpost", "gaming", "twitch", "streamer",
    # Entertainment / gaming / anime
    "netflix", "prime video", "disney+", "minecraft", "lol", "anime", "kpop", "manga",
    "webtoon", "rap", "hiphop", "edm", "dj",
    # Sports
    "futbol", "basketbol", "tennis", "formula 1", "voleybol", "gym", "workout", "fitness",
    # Lifestyle / travel / food / shopping
    "yemek", "tatil", "seyahat", "alışveriş", "fashion", "sneakers", "nike", "adidas",
    "apple", "samsung", "mobile", "app", "airbnb", "hotel", "vegan", "fitness lifestyle",
    # Daily life / work / education
    "üniversite", "iş", "freelance", "marketing", "growth", "seo", "branding",
    "startup life", "gündem", "aşk", "sağlık", "covid", "ekonomi", "politika",
    "haberler", "finance",
    # Misc / pop culture
    "lol memes", "viral", "challenge", "trend", "gaming memes", "internet slang",
    "funny videos", "edutainment", "tech news", "apps review", "music", "concert",
    "festival",
]


def fingerprint(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()


async def collect(topics: list[str], max_pages: int, delay: float) -> list[dict]:
    from eksipy import Eksi

    eksi = Eksi()
    rows: list[dict] = []
    seen: set[str] = set()

    for topic_name in topics:
        print(f"topic: {topic_name}", flush=True)
        try:
            topic = await eksi.getTopic(topic_name)
        except Exception as e:
            print(f"  could not open topic: {e}", file=sys.stderr)
            await asyncio.sleep(random.uniform(3.0, 6.0))
            continue

        kept = 0
        for page in range(1, max_pages + 1):
            try:
                entries = await topic.getEntrys(page=page)
            except Exception as e:
                print(f"  page {page}: {e}", file=sys.stderr)
                break
            if not entries:
                break

            for entry in entries:
                content = entry.entry
                text = content.text() if hasattr(content, "text") else str(content)
                text = text.strip()
                if not text:
                    continue
                fp = fingerprint(text)
                # The same entry is reachable from more than one topic, and a
                # restarted run revisits pages. Deduplicating on content keeps
                # the corpus size honest.
                if fp in seen:
                    continue
                seen.add(fp)
                rows.append({"topic": topic_name, "page": page,
                             "entry_id": getattr(entry, "id", None), "entry": text})
                kept += 1

            await asyncio.sleep(delay)
        print(f"  kept {kept} new entries", flush=True)
        await asyncio.sleep(random.uniform(1.5, 3.0))

    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pages", type=int, default=200, dest="max_pages")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="seconds between page requests")
    ap.add_argument("--topics", nargs="*", default=TOPICS)
    args = ap.parse_args()

    rows = asyncio.run(collect(args.topics, args.max_pages, args.delay))
    if not rows:
        print("no entries collected", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)          # rewritten, never appended
    print(f"\nwrote {args.out}: {len(df):,} unique entries "
          f"across {df['topic'].nunique()} topics")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
