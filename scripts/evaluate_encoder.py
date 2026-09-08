#!/usr/bin/env python3
"""Run a fine-tuned encoder over a split and write token-level predictions.

    python scripts/evaluate_encoder.py --task lid --model-dir models/berturk_lid

Writes results/predictions/<name>.csv in the same shape as the prompted-LLM
runs, so `scripts/report.py` scores encoders and LLMs through the same
code path. 

--max-length defaults to 512 to match train_encoder.py. 
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turengmix.data import (  # noqa: E402
    DEFAULT_BENCHMARK, load_benchmark, load_split, sentences, write_predictions,
)
from turengmix.labels import UNK, id2label  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "results" / "predictions"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=("lid", "ner"))
    ap.add_argument("--model-dir", required=True, type=Path, dest="model_dir")
    ap.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    ap.add_argument("--split", default="test",
                    choices=("train", "validation", "test", "all"))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--run-name", help="defaults to <model-dir name>_<split>")
    ap.add_argument("--batch-size", type=int, default=16, dest="batch_size")
    ap.add_argument("--max-length", type=int, default=512, dest="max_length")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    i2l = id2label(args.task)

    df = load_benchmark(args.benchmark)
    if args.split != "all":
        keep = set(load_split()[args.split])
        df = df[df["doc_id"].isin(keep)].reset_index(drop=True)
    units = sentences(df, args.task)

    tok_ids: dict[tuple[str, int], list[int]] = {}
    for (doc_id, sent_id), g in df.groupby(["doc_id", "sent_id"], sort=False):
        tok_ids[(str(doc_id), int(sent_id))] = g["tok_id"].tolist()

    predictions: dict[tuple[str, int, int], str] = {}
    truncated_words = 0

    with torch.no_grad():
        for start in range(0, len(units), args.batch_size):
            batch = units[start:start + args.batch_size]
            enc = tokenizer(
                [u.tokens for u in batch], is_split_into_words=True,
                truncation=True, max_length=args.max_length,
                padding=True, return_tensors="pt",
            )
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
            argmax = torch.argmax(logits, dim=-1).cpu().numpy()

            for i, unit in enumerate(batch):
                word_ids = enc.word_ids(batch_index=i)
                # First subword of each word carries the prediction, matching
                # how labels were aligned during training.
                first: dict[int, int] = {}
                for pos, w in enumerate(word_ids):
                    if w is not None and w not in first:
                        first[w] = pos
                ids = tok_ids[(unit.doc_id, unit.sent_id)]
                for w, tok_id in enumerate(ids):
                    if w in first:
                        predictions[(unit.doc_id, unit.sent_id, tok_id)] = \
                            i2l[int(argmax[i, first[w]])]
                    else:
                        # Word fell past the truncation point. Recorded as UNK
                        predictions[(unit.doc_id, unit.sent_id, tok_id)] = UNK
                        truncated_words += 1

    run = args.run_name or f"{args.model_dir.name}_{args.split}"
    path = args.out_dir / f"{run}.csv"
    out = write_predictions(df, predictions, f"pred_{args.task}", path)

    print(f"wrote {path}  ({len(out):,} tokens, {len(units)} sentences)")
    if truncated_words:
        print(f"WARNING: {truncated_words} words fell past --max-length "
              f"{args.max_length} and are recorded as {UNK}. Raise it to match "
              f"the value used in training.")
    print("\nScore it with:  python scripts/report.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
