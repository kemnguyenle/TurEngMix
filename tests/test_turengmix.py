"""Regression tests.

Each test here pins down a bug that was live in the original scripts and that
would change a number in the paper if it came back. Run with:

    pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "corpus_construction"))

from turengmix import scoring  # noqa: E402
from turengmix.labels import (  # noqa: E402
    LID_LABELS, NER_ENTITY_TYPES, NER_LABELS, UNK, is_wellformed_bio,
)
from turengmix.prompting import format_tokens, parse_response  # noqa: E402


# --------------------------------------------------------------------------
# Metric definitions, anchored to the paper's published numbers
# --------------------------------------------------------------------------

BENCHMARK = Path(__file__).resolve().parent.parent / "data" / "benchmark.csv"
SPLITS = BENCHMARK.parent / "splits"
_have_data = BENCHMARK.exists() and (SPLITS / "test.txt").exists()


@pytest.mark.skipif(not _have_data, reason="run normalize_annotations.py and make_splits.py")
def test_majority_baseline_reproduces_the_published_values():
    """Tables 9 and 10 report majority-class baselines of 0.1414 (LID) and
    0.0502 (NER). Reproducing both pins down the evaluation scope (test split)
    and the macro denominator (6 LID classes, 19 BIO labels) at once — if
    either were different, these would not match."""
    from turengmix.data import load_benchmark, load_split
    df = load_benchmark(BENCHMARK)
    test = df[df["doc_id"].isin(set(load_split(SPLITS)["test"]))]
    assert scoring.majority_baseline(test["lid"].tolist(), "lid")["macro_f1"] \
        == pytest.approx(0.1414, abs=5e-5)
    assert scoring.majority_baseline(test["ner"].tolist(), "ner")["macro_f1"] \
        == pytest.approx(0.0502, abs=5e-5)


@pytest.mark.skipif(not _have_data, reason="run make_splits.py")
def test_split_reproduces_table_7():
    from turengmix.data import load_benchmark, load_split
    df = load_benchmark(BENCHMARK)
    split = load_split(SPLITS)
    for name, (posts, sents, toks) in {
        "train": (200, 254, 12466),
        "validation": (25, 26, 1173),
        "test": (25, 41, 1373),
    }.items():
        s = df[df["doc_id"].isin(set(split[name]))]
        assert len(split[name]) == posts
        assert s.groupby(["doc_id", "sent_id"]).ngroups == sents
        assert len(s) == toks


def test_macro_denominator_is_the_closed_label_set():
    """A model that never predicts a class must still be averaged over it,
    or two models' macro-F1 land on different scales."""
    gold = ["TR"] * 8 + ["OTHER"] * 2
    timid = scoring.token_metrics(gold, ["TR"] * 10, "lid")
    bold = scoring.token_metrics(gold, ["TR"] * 8 + ["EN"] * 2, "lid")
    assert len(timid["per_class"]) == len(LID_LABELS) == len(bold["per_class"])
    assert timid["per_class"]["OTHER"]["support"] == 2


def test_ner_macro_is_over_nineteen_bio_labels():
    m = scoring.token_metrics(["O"] * 5, ["O"] * 5, "ner")
    assert len(m["per_class"]) == 19


def test_malformed_output_counts_as_incorrect():
    """§5.1: "malformed outputs were treated as incorrect"."""
    m = scoring.token_metrics(["TR", "EN"], ["TR", UNK], "lid")
    assert m["accuracy"] == 0.5 and m["n_malformed"] == 1
    assert scoring.token_metrics(["TR"] * 4, [UNK] * 4, "lid")["accuracy"] == 0.0


def test_per_entity_type_collapses_bio_prefixes():
    """Tables A3/A4 report ten type-level classes, not nineteen BIO labels."""
    t = scoring.ner_type_metrics(["B-PER", "I-PER", "O"], ["B-PER", "I-PER", "O"])
    assert set(t["per_type"]) == {"O", *NER_ENTITY_TYPES}
    assert t["per_type"]["PER"]["support"] == 2


def test_binary_ner_collapses_all_entity_types():
    b = scoring.ner_binary_metrics(["B-PER", "I-ORG", "O"], ["B-LOC", "I-LOC", "O"])
    assert b["ne_f1"] == 1.0 and b["ne_support"] == 2


def test_integration_analysis_matches_the_table_12_shape():
    """Table 12 compares integrated tokens against *all other* evaluated
    tokens, and §6.3.3 reports a chi-square and an odds ratio alongside."""
    gold = ["O"] * 20
    pred = ["O"] * 20
    borrowed = [""] * 20
    for i in range(4):                       # 4 integrated tokens, all wrong
        borrowed[i], pred[i] = "MIXED", "B-PER"
    pred[10] = pred[11] = "B-PER"            # 2 of 16 others wrong

    r = scoring.integration_error_analysis(gold, pred, borrowed, "ner")
    assert r["mixed_error_rate"] == 1.0
    assert r["other_error_rate"] == pytest.approx(0.125)
    assert r["ratio"] == pytest.approx(8.0)
    assert r["n_integrated"] == 4 and r["n_other"] == 16
    assert r["chi2"] is not None and r["p_value"] < 0.05


def test_whitespace_in_borrowed_suffix_does_not_hide_a_token():
    """One raw cell held " MIXED"; an `== "MIXED"` filter dropped it silently,
    in the column that defines Table 12's numerator."""
    r = scoring.integration_error_analysis(
        ["O", "O"], ["B-PER", "B-PER"], [" MIXED", "MIXED"], "ner")
    assert r["n_integrated"] == 2


def test_seed_aggregation_reports_mean_and_sd():
    """§5.2 reports encoder results as mean +/- SD over four seeds."""
    a = scoring.aggregate_seeds([0.40, 0.44, 0.48, 0.52])
    assert a["n"] == 4 and a["mean"] == pytest.approx(0.46)
    assert a["std"] > 0


def test_wilson_interval_widens_on_small_strata():
    narrow = scoring.wilson(200, 400)
    wide = scoring.wilson(2, 4)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


# --------------------------------------------------------------------------
# One input format, one partial-output policy, for every model
# --------------------------------------------------------------------------

def test_input_format_is_index_tab_token():
    assert format_tokens(["bu", "link'teki"]) == "1\tbu\n2\tlink'teki"


def test_partial_output_is_kept_not_discarded():
    """The original GPT LID scripts replaced a whole post with UNK when the
    model returned one stray line; the other three scripts kept partials. That
    asymmetry penalised only the models scored the first way."""
    r = parse_response("1|TR\n2|EN\n9|TR", "lid", n_expected=3)
    assert not r.aligned and r.extra_indices == [9] and r.missing_indices == [3]
    assert r.ordered() == ["TR", "EN", UNK]      # two good labels survive


def test_parser_accepts_the_documented_separator_and_near_misses():
    for line in ("1|TR", "1\tTR", "1: TR", "  1 | TR  "):
        assert parse_response(line, "lid", 1).ordered() == ["TR"]


def test_parser_rejects_labels_outside_the_closed_set():
    r = parse_response("1|TURKISH\n2|TR", "lid", n_expected=2)
    assert r.ordered() == [UNK, "TR"]
    assert r.unparsed_lines == ["1|TURKISH"]


def test_longest_label_wins_so_b_other_is_not_read_as_o():
    assert parse_response("1|B-OTHER", "ner", 1).ordered() == ["B-OTHER"]
    assert parse_response("1|O", "ner", 1).ordered() == ["O"]


def test_repeated_index_keeps_the_first_and_is_recorded():
    r = parse_response("1|TR\n1|EN", "lid", n_expected=1)
    assert r.ordered() == ["TR"] and r.duplicate_indices == [1]


def test_reasoning_block_is_not_parsed_as_labels():
    from turengmix.prompting import strip_reasoning
    raw = "<think>Token 1 looks English, so 1|EN</think>\n1|TR"
    assert parse_response(strip_reasoning(raw), "lid", 1).ordered() == ["TR"]


# --------------------------------------------------------------------------
# Label hygiene
# --------------------------------------------------------------------------

def test_bio_wellformedness_flags_orphan_i_tags():
    assert is_wellformed_bio(["O", "B-ORG", "I-ORG", "O"]) == []
    assert is_wellformed_bio(["I-ORG"]) == [0]            # sentence-initial I-
    assert is_wellformed_bio(["O", "I-EVENT"]) == [1]     # no matching B-
    assert is_wellformed_bio(["B-ORG", "I-PER"]) == [1]   # type switches


def test_ner_label_set_is_complete_and_ordered():
    assert NER_LABELS[0] == "O"
    assert len(NER_LABELS) == 1 + 2 * 9
    assert "B-TIME" in NER_LABELS and "I-TIME" in NER_LABELS


# --------------------------------------------------------------------------
# Tokenizer
# --------------------------------------------------------------------------

def test_trailing_punctuation_is_split_consistently():
    """`Google'e.` kept its period while `Spotify'da.` was split — the old
    heuristic fired only on tokens with an apostrophe ending in "e"."""
    from tokenize_posts import tokenize_text
    for text, stem in (("Google'e.", "Google'e"), ("Spotify'da.", "Spotify'da"),
                       ("iPhone'e!", "iPhone'e"), ("meeting'e?", "meeting'e")):
        assert tokenize_text(text)[0] == stem
        assert len(tokenize_text(text)) == 2


def test_apostrophe_suffix_stays_attached_to_its_stem():
    from tokenize_posts import tokenize_text
    assert tokenize_text("bu entry'nin altinda") == ["bu", "entry'nin", "altinda"]


def test_empty_post_yields_no_tokens_instead_of_crashing():
    from tokenize_posts import tokenize_text
    assert tokenize_text("") == [] and tokenize_text("   ") == []


def test_urls_do_not_swallow_trailing_punctuation():
    from tokenize_posts import tokenize_text
    assert tokenize_text("link http://a.co/x, sonra") == [
        "link", "http://a.co/x", ",", "sonra"]


def test_flag_emoji_are_recognised():
    from tokenize_posts import tokenize_text
    assert "🇹🇷" in tokenize_text("bayrak 🇹🇷 iste")


# --------------------------------------------------------------------------
# Released data
# --------------------------------------------------------------------------

@pytest.mark.skipif(not _have_data,
                    reason="run scripts/normalize_annotations.py first")
class TestBenchmark:
    def test_loads_and_satisfies_its_invariants(self):
        from turengmix.data import load_benchmark
        df = load_benchmark(BENCHMARK)          # raises on a duplicate key
        assert not df.duplicated(["doc_id", "sent_id", "tok_id"]).any()
        assert set(df["lid"]) <= set(LID_LABELS)
        assert set(df["ner"]) <= set(NER_LABELS)
        assert (df["ner"] != "").all()          # blanks became O

    def test_tok_id_is_dense_within_every_sentence(self):
        from turengmix.data import load_benchmark
        df = load_benchmark(BENCHMARK)
        for _, g in df.groupby(["doc_id", "sent_id"], sort=False):
            assert list(g["tok_id"]) == list(range(len(g)))

    def test_split_is_by_document_and_covers_the_benchmark(self):
        from turengmix.data import load_benchmark, load_split
        df = load_benchmark(BENCHMARK)
        split = load_split(SPLITS)
        every = [d for docs in split.values() for d in docs]
        assert len(every) == len(set(every)) == df["doc_id"].nunique()
