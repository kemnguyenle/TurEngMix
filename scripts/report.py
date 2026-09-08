#!/usr/bin/env python3
"""Regenerate the paper's result tables from prediction files.

    python scripts/report.py

Reads every prediction CSV in results/predictions/ and writes, to results/:

    table08_lid_per_class.md    per-class F1, LID
    table09_lid_macro.md        macro F1, LID
    table10_ner_bio_macro.md    BIO-level macro F1, NER
    table11_ner_binary.md       NE vs Non-NE F1
    table12_integration.md      error rates on integrated tokens, chi-square
    tableA3_ner_per_type.md     per-entity-type F1, decoder LLMs
    tableA4_ner_per_type.md     per-entity-type F1, encoders
    metrics.json                every number above, machine-readable
    <run>_report.txt            full classification report per run
    <run>_confusion.csv         confusion matrix per run, including UNK

Evaluation scope follows the paper and is chosen per run, not globally:
decoder LLMs were run over the complete benchmark (250 posts), encoders were
evaluated on the test split. Set it explicitly with `scope` in a run's
.meta.json, or let the filename decide.

Encoder runs are averaged over random seeds and reported as mean +/- SD, as in
§5.2. Name them `<model>_<task>_seed<N>` and they group automatically.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from turengmix import scoring  # noqa: E402
from turengmix.data import dump_json, load_benchmark, load_split  # noqa: E402
from turengmix.labels import LID_LABELS, NER_ENTITY_TYPES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = REPO_ROOT / "results" / "predictions"
OUT_DIR = REPO_ROOT / "results"

SEED_RE = re.compile(r"^(?P<base>.+?)[_-]seed(?P<seed>\d+)$", re.I)
# Encoder checkpoints are evaluated on the held-out test split; prompted
# decoders were run over the whole benchmark.
ENCODER_HINTS = ("berturk", "xlm", "roberta", "bertweet", "encoder")


def infer_scope(run: str, meta: dict) -> str:
    if meta.get("scope") in ("all", "test"):
        return meta["scope"]
    low = run.lower()
    return "test" if any(h in low for h in ENCODER_HINTS) else "all"


def score_run(path: Path, split_dir: Path) -> dict | None:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[])
    task = ("lid" if "pred_lid" in df.columns
            else "ner" if "pred_ner" in df.columns else None)
    if task is None:
        print(f"  skipping {path.name}: no pred_lid or pred_ner column")
        return None

    meta_path = path.with_suffix("").with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    run = path.stem
    scope = infer_scope(run, meta)
    if scope == "test":
        df = df[df["doc_id"].isin(set(load_split(split_dir)["test"]))]
    if df.empty:
        print(f"  skipping {path.name}: no rows in scope '{scope}'")
        return None

    gold = df[task].tolist()
    pred = df[f"pred_{task}"].tolist()
    m = SEED_RE.match(run)

    result: dict = {
        "run": run,
        "base": m.group("base") if m else run,
        "seed": int(m.group("seed")) if m else meta.get("seed"),
        "task": task,
        "scope": scope,
        "n_tokens": len(df),
        "n_documents": df["doc_id"].nunique(),
        "token_level": scoring.token_metrics(gold, pred, task),
        "error_rate_by_gold_lid": scoring.error_rate_by_lid(
            gold, pred, df["lid"].tolist(), task),
        "integration": scoring.integration_error_analysis(
            gold, pred, df["borrowed_suffix"].tolist(), task),
    }
    if task == "ner":
        result["per_entity_type"] = scoring.ner_type_metrics(gold, pred)
        result["binary"] = scoring.ner_binary_metrics(gold, pred)
    if meta:
        result["run_metadata"] = meta
    return result


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------

def f(x, nd=4):
    return "-" if x is None or x != x else f"{x:.{nd}f}"


def group_by_base(results: list[dict]) -> list[tuple[str, list[dict]]]:
    """Collapse `<base>_seedN` runs into one row each, preserving order."""
    order, groups = [], {}
    for r in results:
        if r["base"] not in groups:
            groups[r["base"]] = []
            order.append(r["base"])
        groups[r["base"]].append(r)
    return [(b, groups[b]) for b in order]


def cell(runs: list[dict], get) -> str:
    """One cell: a plain number, or mean +/- SD when a run has several seeds."""
    vals = [get(r) for r in runs]
    if len(vals) == 1:
        return f(vals[0])
    agg = scoring.aggregate_seeds(vals)
    return f"{agg['mean']:.4f} ± {agg['std']:.4f}"


def header(title: str, note: str, baseline: dict | None = None) -> list[str]:
    out = [f"# {title}", "", note, ""]
    if baseline:
        out += [f"Majority-class baseline ({baseline['majority_class']}, test split): "
                f"**{baseline['macro_f1']:.4f}**", ""]
    return out


def table09(results, baseline) -> str:
    runs = [r for r in results if r["task"] == "lid"]
    lines = header(
        "Table 9 — Macro F1, language identification",
        "Encoder rows are mean ± SD across seeds.", baseline)
    lines += ["| Model | Scope | Macro F1 | Accuracy | Malformed | Tokens |",
              "|:--|:--|--:|--:|--:|--:|"]
    for base, rs in group_by_base(runs):
        lines.append(
            f"| `{base}` | {rs[0]['scope']} | "
            f"{cell(rs, lambda r: r['token_level']['macro_f1'])} | "
            f"{cell(rs, lambda r: r['token_level']['accuracy'])} | "
            f"{rs[0]['token_level']['malformed_rate']:.2%} | "
            f"{rs[0]['n_tokens']:,} |")
    return "\n".join(lines) + "\n"


def table08(results) -> str:
    runs = [r for r in results if r["task"] == "lid"]
    lines = header(
        "Table 8 — Per-class F1, language identification")
    lines += ["| Model | " + " | ".join(LID_LABELS) + " |",
              "|:--|" + "--:|" * len(LID_LABELS)]
    for base, rs in group_by_base(runs):
        cells = [cell(rs, lambda r, c=c: r["token_level"]["per_class"][c]["f1"])
                 for c in LID_LABELS]
        lines.append(f"| `{base}` | " + " | ".join(cells) + " |")
    if runs:
        sup = runs[0]["token_level"]["per_class"]
        lines.append("| _support_ | " + " | ".join(
            f"{sup[c]['support']:,}" for c in LID_LABELS) + " |")
    return "\n".join(lines) + "\n"


def table10(results, baseline) -> str:
    runs = [r for r in results if r["task"] == "ner"]
    lines = header(
        "Table 10 — BIO-level macro F1, named entity recognition",
        "Macro-averaged over the nineteen BIO labels.",
        baseline)
    lines += ["| Model | Scope | Macro F1 | Accuracy | Malformed | Tokens |",
              "|:--|:--|--:|--:|--:|--:|"]
    for base, rs in group_by_base(runs):
        lines.append(
            f"| `{base}` | {rs[0]['scope']} | "
            f"{cell(rs, lambda r: r['token_level']['macro_f1'])} | "
            f"{cell(rs, lambda r: r['token_level']['accuracy'])} | "
            f"{rs[0]['token_level']['malformed_rate']:.2%} | "
            f"{rs[0]['n_tokens']:,} |")
    return "\n".join(lines) + "\n"


def table11(results) -> str:
    runs = [r for r in results if r["task"] == "ner"]
    lines = header(
        "Table 11 — Binary entity detection",
        "All entity categories collapsed into NE vs Non-NE.")
    lines += ["| Model | Non-NE F1 | NE F1 |", "|:--|--:|--:|"]
    for base, rs in group_by_base(runs):
        lines.append(f"| `{base}` | "
                     f"{cell(rs, lambda r: r['binary']['non_ne_f1'])} | "
                     f"{cell(rs, lambda r: r['binary']['ne_f1'])} |")
    return "\n".join(lines) + "\n"


def tableA3(results, encoders: bool) -> str:
    runs = [r for r in results if r["task"] == "ner"
            and (r["scope"] == "test") == encoders]
    name = "A4 — encoder models" if encoders else "A3 — decoder LLMs"
    lines = header(
        f"Table {name}: per-entity-type F1",
        "B- and I- collapsed into one class per entity type.")
    cols = ["O"] + NER_ENTITY_TYPES
    lines += ["| Model | " + " | ".join(cols) + " |",
              "|:--|" + "--:|" * len(cols)]
    for base, rs in group_by_base(runs):
        cells = [cell(rs, lambda r, c=c: r["per_entity_type"]["per_type"][c]["f1"])
                 for c in cols]
        lines.append(f"| `{base}` | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def table12(results) -> str:
    lines = header(
        "Table 12 — Errors on morphologically integrated tokens",
        "Tokens whose English-origin stem carries a Turkish suffix "
        "(`borrowed_suffix == MIXED`, 386 tokens, including morphologically "
        "adapted named entities) against all other evaluated tokens. "
        "Chi-square is a 2x2 test on the raw counts; the odds ratio "
        "accompanies it, as in §6.3.3.")
    lines += ["| Model | Task | Other tokens | Mixed tokens | Ratio | Odds ratio | chi2 | p |",
              "|:--|:--|--:|--:|--:|--:|--:|:--|"]
    for base, rs in group_by_base(results):
        r = rs[0]
        i = r["integration"]
        p = i.get("p_value")
        p_str = "-" if p is None else ("< 0.001" if p < 0.001 else f"{p:.3f}")
        lines.append(
            f"| `{base}` | {r['task'].upper()} | "
            f"{i['other_error_rate']:.1%} | {i['mixed_error_rate']:.1%} | "
            f"{f(i['ratio'], 2)}x | {f(i['odds_ratio'], 2)} | "
            f"{f(i.get('chi2'), 2)} | {p_str} |")

    lines += ["", "## Error rate by gold LID class", "",
              "Wilson 95% intervals. Several strata are very small — `OTHER` "
              "has four tokens in the whole benchmark.", ""]
    for base, rs in group_by_base(results):
        r = rs[0]
        lines += [f"### `{base}` ({r['task'].upper()})", "",
                  "| Gold LID | Tokens | Errors | Error rate | 95% CI |",
                  "|:--|--:|--:|--:|:--|"]
        for cls, s in sorted(r["error_rate_by_gold_lid"].items(),
                             key=lambda kv: -kv[1]["n_tokens"]):
            lo, hi = s["ci95"]
            lines.append(f"| {cls} | {s['n_tokens']:,} | {s['n_errors']:,} | "
                         f"{s['error_rate']:.3f} | [{lo:.3f}, {hi:.3f}] |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, nargs="*")
    ap.add_argument("--pred-dir", type=Path, default=PRED_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--benchmark", type=Path, default=None)
    args = ap.parse_args()

    paths = args.predictions or sorted(args.pred_dir.glob("*.csv"))
    if not paths:
        print(f"No prediction files in {args.pred_dir}.")
        print("Produce them with scripts/run_llm.py or scripts/evaluate_encoder.py,")
        print("or drop the released prediction CSVs there.")
        return 1

    split_dir = REPO_ROOT / "data" / "splits"
    df = load_benchmark(args.benchmark) if args.benchmark else load_benchmark()
    test = df[df["doc_id"].isin(set(load_split(split_dir)["test"]))]
    baselines = {t: scoring.majority_baseline(test[t].tolist(), t)
                 for t in ("lid", "ner")}
    print(f"majority-class baseline (test split): "
          f"LID {baselines['lid']['macro_f1']:.4f}  "
          f"NER {baselines['ner']['macro_f1']:.4f}   "
          f"(paper: 0.1414 / 0.0502)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    print(f"\nscoring {len(paths)} run(s)")
    for p in paths:
        r = score_run(p, split_dir)
        if r is None:
            continue
        results.append(r)

        d = pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[])
        if r["scope"] == "test":
            d = d[d["doc_id"].isin(set(load_split(split_dir)["test"]))]
        task = r["task"]
        gold, pred = d[task].tolist(), d[f"pred_{task}"].tolist()
        axis, cm = scoring.confusion(gold, pred, task)
        pd.DataFrame(cm, index=axis, columns=axis).to_csv(
            args.out_dir / f"{r['run']}_confusion.csv")
        (args.out_dir / f"{r['run']}_report.txt").write_text(
            scoring.report_text(gold, pred, task), encoding="utf-8")

        print(f"  {r['run']:<38} {task.upper()}  scope={r['scope']:<4}  "
              f"macro-F1 {r['token_level']['macro_f1']:.4f}  "
              f"malformed {r['token_level']['malformed_rate']:.2%}")

    writes = {
        "table08_lid_per_class.md": table08(results),
        "table09_lid_macro.md": table09(results, baselines["lid"]),
        "table10_ner_bio_macro.md": table10(results, baselines["ner"]),
        "table11_ner_binary.md": table11(results),
        "table12_integration.md": table12(results),
        "tableA3_ner_per_type.md": tableA3(results, encoders=False),
        "tableA4_ner_per_type.md": tableA3(results, encoders=True),
    }
    for name, text in writes.items():
        (args.out_dir / name).write_text(text, encoding="utf-8")
    dump_json({"baselines": baselines, "runs": results},
              args.out_dir / "metrics.json")

    print(f"\nwrote {len(writes) + 1} files to {args.out_dir}")
    for name in list(writes) + ["metrics.json"]:
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
