#!/usr/bin/env python3
"""One-off: remap committed split files from the old doc_id scheme
(post_0001_1, post_007_47) to the simplified scheme (post_0001).

The partition is unchanged — same 250 posts, same train/validation/test
buckets. Only the identifier form changes, so the split is remapped rather
than regenerated: make_splits.py would reshuffle and is not guaranteed to
reproduce the published partition under cleaned ids.

    python scripts/remap_splits.py            # writes in place
    python scripts/remap_splits.py --check    # verify only, write nothing
"""
from __future__ import annotations

import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR = REPO_ROOT / "data" / "splits"

# token counts per split, for the header line — unchanged from the paper's Table 7
TOKENS = {"train": 12466, "validation": 1173, "test": 1373}

HEADER = (
    "# TurEngMix document-level split — {name}\n"
    "# {n} documents, {t} tokens. Partition unchanged from the original release;\n"
    "# doc_ids remapped from the two-scheme export (post_0001_1, post_007_47) to\n"
    "# the simplified post_NNNN scheme by scripts/remap_splits.py.\n"
    "# Committed deliberately: do not edit by hand, do not regenerate with make_splits.py.\n"
)


def old_to_new(old_id: str) -> str:
    """post_0001_1 / post_007_47 -> post_NNNN, keyed on the unique trailing number."""
    num = int(old_id.rsplit("_", 1)[-1])
    return f"post_{num:03d}"


def read_ids(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.startswith("#")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-dir", type=Path, default=SPLIT_DIR)
    ap.add_argument("--check", action="store_true", help="verify only, write nothing")
    args = ap.parse_args()

    all_new: dict[str, list[str]] = {}
    for name in ("train", "validation", "test"):
        path = args.split_dir / f"{name}.txt"
        old = read_ids(path)
        new = sorted({old_to_new(x) for x in old}, key=lambda d: int(d.rsplit("_", 1)[-1]))
        if len(new) != len(old):
            raise SystemExit(f"{name}: {len(old)} old ids collapsed to {len(new)} "
                             f"new ids — trailing numbers are not unique")
        all_new[name] = new

    # partition sanity: 250 unique docs, no overlap, expected sizes
    sizes = {k: len(v) for k, v in all_new.items()}
    if sizes != {"train": 200, "validation": 25, "test": 25}:
        raise SystemExit(f"unexpected split sizes: {sizes}")
    everything = [d for v in all_new.values() for d in v]
    if len(set(everything)) != 250:
        raise SystemExit(f"expected 250 unique docs, got {len(set(everything))}")

    if args.check:
        print("check passed:", sizes, "— 250 unique docs, no overlap")
        return 0

    for name, ids in all_new.items():
        path = args.split_dir / f"{name}.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(HEADER.format(name=name, n=len(ids), t=TOKENS[name]))
            f.write("\n".join(ids) + "\n")
        print(f"wrote {path}  ({len(ids)} docs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())