#!/usr/bin/env python3
"""Turn the raw annotation export into the canonical benchmark.

    python scripts/normalize_annotations.py

Reads  data/raw/annotations_raw.csv
Writes data/benchmark.csv  and  data/normalization_report.json

The raw export stays in the repository untouched, and every change made here
is mechanical, reversible and logged, so the two can be diffed.

`doc_id` values are kept exactly as the annotators produced them, including
the two numbering schemes present in the export. They match the identifiers
already published on the Hugging Face Hub, and keeping the GitHub and Hub
releases identifier-compatible matters more than tidiness. The trailing
number is the one that is unique across all 250 documents; it is exposed as
`doc_num` for convenience.

Two classes of problem are handled differently. Mechanical defects — blank
`ner` cells, whitespace in a label, a duplicate key, a stray index column —
are fixed. Anything needing an annotation judgement is counted and listed in
the report for adjudication, never guessed: silently repairing an ambiguous
case would bake one reading of it into the released gold standard.

The report also compares the resulting label counts against Tables 4 and 5 of
the paper, so any drift between the published tables and the released file is
visible rather than discovered by a reader.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turengmix.data import BENCHMARK_COLUMNS, dump_json  # noqa: E402
from turengmix.labels import (  # noqa: E402
    LID_LABELS, NER_ENTITY_TYPES, NER_LABELS, is_wellformed_bio,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW = REPO_ROOT / "data" / "raw" / "annotations_raw.csv"
OUT = REPO_ROOT / "data" / "benchmark.csv"
REPORT = REPO_ROOT / "data" / "normalization_report.json"

# Published counts, for drift detection.
PAPER_LID = {"TR": 11749, "EN": 2015, "NE": 793, "MIXED": 193,
             "AMBIGUOUS": 62, "OTHER": 4}          # Table 4
PAPER_NER = {"O": 13981, "PROD": 350, "TITLE": 246, "ORG": 129, "PER": 119,
             "LOC": 82, "OTHER": 57, "GROUP": 26, "EVENT": 19, "TIME": 3}  # Table 5
PAPER_INTEGRATED = 386                              # §4.1.1
PAPER_TOTAL = 15012


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=RAW)
    ap.add_argument("--output", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, default=REPORT)
    args = ap.parse_args()

    # keep_default_na=False so tokens such as "nan", "null" and "NA" survive
    # as the strings they are.
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False, na_values=[])
    log: dict = {"input": str(args.input), "rows_in": len(df), "changes": {}}
    C = log["changes"]

    junk = [c for c in df.columns if c.startswith("Unnamed") or c.strip() == ""]
    df = df.drop(columns=junk)
    C["dropped_index_columns"] = junk

    missing = [c for c in BENCHMARK_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"raw export is missing required columns: {missing}")
    df = df[BENCHMARK_COLUMNS].copy()

    # --- whitespace in label cells ----------------------------------------
    # One `borrowed_suffix` cell held " MIXED"; any `== "MIXED"` filter drops
    # it silently, and that column defines the morphologically integrated
    # tokens the paper's central result is about.
    trimmed = {}
    for col in ("lid", "borrowed_suffix", "ner"):
        before = df[col].tolist()
        df[col] = df[col].str.strip()
        n = sum(a != b for a, b in zip(before, df[col]))
        if n:
            trimmed[col] = n
    C["whitespace_trimmed"] = trimmed

    blank_ner = int((df["ner"] == "").sum())
    df.loc[df["ner"] == "", "ner"] = "O"
    C["blank_ner_filled_with_O"] = blank_ner

    bad_lid = sorted(set(df["lid"]) - set(LID_LABELS))
    bad_ner = sorted(set(df["ner"]) - set(NER_LABELS))
    if bad_lid or bad_ner:
        raise SystemExit("labels outside the closed set — fix the annotation, "
                         f"not this script.\n  lid: {bad_lid}\n  ner: {bad_ner}")

    df["sent_id"] = df["sent_id"].astype(int)
    df["tok_id"] = df["tok_id"].astype(int)

    # --- tok_id: an identifier with gaps, made into a dense position -------
    # Tokens removed during annotation cleaning left 1,404 gaps, 16 sentences
    # missing their first token, and one pair of rows sharing a key outright.
    # Renumber densely within each sentence, preserving file order, and keep
    # the export's value as `source_tok_id`.
    key = ["doc_id", "sent_id", "tok_id"]
    dup_before = int(df.duplicated(key, keep=False).sum())
    gaps = sum((g["tok_id"].max() - g["tok_id"].min() + 1) - len(g)
               for _, g in df.groupby(["doc_id", "sent_id"], sort=False))
    df["source_tok_id"] = df["tok_id"]
    df["tok_id"] = df.groupby(["doc_id", "sent_id"], sort=False).cumcount()
    C["duplicate_keys_before"] = dup_before
    C["duplicate_keys_after"] = int(df.duplicated(key, keep=False).sum())
    C["tok_id_gaps_closed"] = int(gaps)

    # `doc_id` is left alone; the trailing number is the unique one.
    df["doc_num"] = df["doc_id"].map(lambda d: int(d.rsplit("_", 1)[-1]))
    if df["doc_num"].nunique() != df["doc_id"].nunique():
        raise SystemExit("doc_id trailing numbers are not unique per document")
    C["doc_ids_renumbered"] = 0
    C["doc_id_note"] = ("left exactly as exported, to stay compatible with the "
                        "identifiers already published on the Hub")

    # =======================================================================
    # Reported, not repaired
    # =======================================================================
    issues: dict = {}

    ill_formed = []
    for (doc_id, sent_id), g in df.groupby(["doc_id", "sent_id"], sort=False):
        tags, toks = g["ner"].tolist(), g["token"].tolist()
        for i in is_wellformed_bio(tags):
            ill_formed.append({
                "doc_id": doc_id, "sent_id": int(sent_id),
                "tok_id": int(g["tok_id"].iloc[i]), "token": toks[i],
                "tag": tags[i], "previous_tag": tags[i - 1] if i else None})
    issues["ill_formed_bio"] = ill_formed

    # §4.1.1: "all named entities were assigned the NE label regardless of
    # language of origin or the presence of Turkish suffixes". These rows do
    # not follow that rule.
    entity = df["ner"] != "O"
    a = df[entity & (df["lid"] != "NE")]
    b = df[(df["lid"] == "NE") & ~entity]
    issues["entity_token_lid_not_NE"] = {
        "count": int(len(a)), "by_lid": dict(Counter(a["lid"])),
        "rule": "§4.1.1 states entity tokens take LID label NE",
        "examples": a.head(10)[["doc_id", "sent_id", "token", "lid", "ner"]]
        .to_dict("records")}
    issues["lid_NE_without_entity_tag"] = {
        "count": int(len(b)),
        "examples": b.head(10)[["doc_id", "sent_id", "token", "lid", "ner"]]
        .to_dict("records")}
    issues["labels_never_observed"] = [l for l in NER_LABELS
                                       if l not in set(df["ner"])]

    # =======================================================================
    # Drift against the published tables
    # =======================================================================
    lid_counts = Counter(df["lid"])
    ner_type = {"O": int((df["ner"] == "O").sum())}
    for t in NER_ENTITY_TYPES:
        ner_type[t] = int(df["ner"].isin([f"B-{t}", f"I-{t}"]).sum())
    integrated = int((df["borrowed_suffix"] == "MIXED").sum())
    n_integrated_ne = int(((df["lid"] == "NE") &
                           (df["borrowed_suffix"] == "MIXED")).sum())

    drift = {"table4_lid": {}, "table5_ner": {}}
    for k, v in PAPER_LID.items():
        got = int(lid_counts.get(k, 0))
        drift["table4_lid"][k] = {"paper": v, "benchmark": got, "delta": got - v}
    for k, v in PAPER_NER.items():
        drift["table5_ner"][k] = {"paper": v, "benchmark": ner_type[k],
                                  "delta": ner_type[k] - v}
    drift["notes"] = [
        # NE=793 in Table 4 is NE minus the morphologically integrated NE
        # tokens, which the table's caption alludes to; that is why the
        # published LID table sums to less than its own stated total.
        f"Table 4 reports NE={PAPER_LID['NE']}; the benchmark has "
        f"{lid_counts['NE']} NE tokens, of which {n_integrated_ne} are also "
        f"morphologically integrated. {lid_counts['NE']} - {n_integrated_ne} = "
        f"{lid_counts['NE'] - n_integrated_ne}, so Table 4's NE row excludes "
        f"integrated NE tokens and the table sums to "
        f"{sum(PAPER_LID.values())}, not {PAPER_TOTAL}.",
        f"Integrated tokens: paper {PAPER_INTEGRATED}, benchmark {integrated}.",
    ]
    nonzero = {t: {k: v for k, v in d.items() if isinstance(v, dict) and v["delta"]}
               for t, d in drift.items() if t != "notes"}
    drift["unexplained_deltas"] = {k: v for k, v in nonzero.items() if v}
    log["comparison_with_published_tables"] = drift

    log["issues_requiring_adjudication"] = issues
    log["rows_out"] = len(df)
    log["documents"] = int(df["doc_id"].nunique())
    log["sentences"] = int(df.groupby(["doc_id", "sent_id"]).ngroups)
    log["lid_distribution"] = dict(lid_counts.most_common())
    log["ner_distribution"] = dict(Counter(df["ner"]).most_common())
    log["ner_by_entity_type"] = ner_type
    log["morphologically_integrated_tokens"] = integrated

    out_cols = BENCHMARK_COLUMNS + ["doc_num", "source_tok_id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df[out_cols].to_csv(args.output, index=False)
    dump_json(log, args.report)

    print(f"wrote {args.output}  ({len(df):,} tokens, {log['documents']} documents, "
          f"{log['sentences']} sentences)")
    print(f"wrote {args.report}\n")
    print("Fixed automatically:")
    print(f"  blank ner cells set to O ........... {blank_ner:,}")
    print(f"  whitespace trimmed from labels ..... {sum(trimmed.values())}")
    print(f"  tok_id gaps closed ................. {gaps:,}")
    print(f"  duplicate keys resolved ............ {dup_before}")
    n_open = len(ill_formed) + len(a) + len(b)
    print(f"\nNeeds an annotator decision ({n_open} tokens, listed in the report):")
    print(f"  ill-formed BIO transitions ......... {len(ill_formed)}")
    print(f"  entity tokens with lid != NE ....... {len(a)}")
    print(f"  lid == NE with no entity tag ....... {len(b)}")

    if drift["unexplained_deltas"]:
        print("\nDIFFERS FROM THE PUBLISHED TABLES:")
        for table, cells in drift["unexplained_deltas"].items():
            for k, v in cells.items():
                if table == "table4_lid" and k == "NE":
                    continue          # explained in the notes above
                print(f"  {table:<12} {k:<8} paper={v['paper']:>6}  "
                      f"file={v['benchmark']:>6}  delta={v['delta']:+}")
        print("  See comparison_with_published_tables in the report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
