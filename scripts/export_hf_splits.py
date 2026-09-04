#!/usr/bin/env python3
"""Write the split CSVs the Hugging Face dataset card expects.

    python scripts/export_hf_splits.py --out-dir hf_upload/

Produces `splits/{train,validation,test}.csv` plus the dataset card, ready to
upload. Publishing the split as real dataset splits is what stops two groups
benchmarking on TurEngMix from partitioning it differently and reporting
numbers that cannot be compared.

Only the seven annotation columns are exported; the `source_*` provenance
columns stay in the repository, where they are useful for tracing back to the
raw export, and would only be noise on the Hub.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turengmix.data import (  # noqa: E402
    BENCHMARK_COLUMNS, DEFAULT_BENCHMARK, load_benchmark, load_split,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CARD = REPO_ROOT / "data" / "huggingface_dataset_card.md"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "hf_upload")
    ap.add_argument("--corpus", type=Path,
                    help="optional path to TurEngMix_Corpus.csv to copy alongside")
    args = ap.parse_args()

    df = load_benchmark(args.benchmark)
    split = load_split()

    splits_dir = args.out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, docs in split.items():
        part = df[df["doc_id"].isin(set(docs))][BENCHMARK_COLUMNS]
        path = splits_dir / f"{name}.csv"
        part.to_csv(path, index=False)
        print(f"{name:<11} {part['doc_id'].nunique():>4} documents  "
              f"{len(part):>7,} tokens  -> {path.relative_to(args.out_dir)}")

    if CARD.exists():
        shutil.copy(CARD, args.out_dir / "README.md")
        print(f"\ncard        -> {args.out_dir / 'README.md'}")
    if args.corpus and args.corpus.exists():
        shutil.copy(args.corpus, args.out_dir / "TurEngMix_Corpus.csv")
        print(f"corpus      -> {args.out_dir / 'TurEngMix_Corpus.csv'}")

    print(f"""
Before uploading {args.out_dir}:

  1. Remove the HTML comment at the top of README.md.
  2. Confirm the `license:` field — the card proposes cc-by-nc-sa-4.0 in place
     of the cc-by-nc-nd-4.0 currently published. See DATA_LICENSE.
  3. Add TurEngMix_Corpus.csv if it is not already there (--corpus).

Then:  huggingface-cli upload ilydoa/TurEngMix {args.out_dir} . --repo-type=dataset

Verify the load afterwards:
  python -c "from datasets import load_dataset; \\
    d = load_dataset('ilydoa/TurEngMix', 'benchmark'); print(d)"
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
