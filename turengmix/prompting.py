"""Shared input formatting and output parsing for prompted models.

The original scripts built the model input three different ways against
prompts that specified a fourth (`"{i}: {token}"`, `"{i}\\n{token}"` and
`"{i}\\t{token}"` for prompts that all said "index, tab, token"), and handled
misaligned output two different ways. Both are single functions here so every
model sees the same input and every model's failures are treated identically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .labels import UNK, labels_for


def format_tokens(tokens: list[str]) -> str:
    """The one input format, matching what the prompt files describe:
    one line per token, `INDEX<TAB>TOKEN`, indices 1-based."""
    return "\n".join(f"{i}\t{tok}" for i, tok in enumerate(tokens, start=1))


def _pattern(task: str) -> re.Pattern:
    # Longest-first alternation so `B-OTHER` cannot be shadowed by a shorter
    # prefix, and so bare `O` is only matched when nothing longer fits.
    labels = sorted(labels_for(task), key=len, reverse=True)
    alt = "|".join(re.escape(lab) for lab in labels)
    # Accept `|`, tab or colon as the separator. The prompts ask for `|`;
    # tolerating the near misses recovers labels a stricter parser would throw
    # away, which keeps the parse-failure rate a measure of the model rather
    # than of our regex.
    return re.compile(rf"^\s*(\d+)\s*(?:\||\t|:)\s*({alt})\s*$")


@dataclass
class ParseResult:
    """Outcome of parsing one model response."""

    labels: dict[int, str]
    n_expected: int
    duplicate_indices: list[int] = field(default_factory=list)
    unparsed_lines: list[str] = field(default_factory=list)

    @property
    def missing_indices(self) -> list[int]:
        return sorted(set(range(1, self.n_expected + 1)) - set(self.labels))

    @property
    def extra_indices(self) -> list[int]:
        return sorted(set(self.labels) - set(range(1, self.n_expected + 1)))

    @property
    def aligned(self) -> bool:
        return not self.missing_indices and not self.extra_indices

    def ordered(self) -> list[str]:
        """Labels in token order, `UNK` wherever the model gave us nothing.

        Partial output is always kept. Discarding a whole post because one
        line was malformed — as the original GPT LID scripts did, while the
        other three scripts kept partials — makes the models' error rates
        incomparable, since it penalises only the models scored that way.
        """
        return [self.labels.get(i, UNK) for i in range(1, self.n_expected + 1)]


def parse_response(text: str, task: str, n_expected: int) -> ParseResult:
    """Parse `INDEX|LABEL` lines out of a model response."""
    pattern = _pattern(task)
    labels: dict[int, str] = {}
    duplicates: list[int] = []
    unparsed: list[str] = []

    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            unparsed.append(line)
            continue
        idx, tag = int(m.group(1)), m.group(2)
        # A repeated index means the model contradicted itself; keeping the
        # first and recording the collision is more faithful than letting a
        # later line silently overwrite an earlier one.
        if idx in labels:
            duplicates.append(idx)
            continue
        labels[idx] = tag

    return ParseResult(
        labels=labels,
        n_expected=n_expected,
        duplicate_indices=sorted(set(duplicates)),
        unparsed_lines=unparsed,
    )


def strip_reasoning(text: str) -> str:
    """Remove a `<think>...</think>` block from a reasoning model's output.

    Qwen3 emits reasoning by default. The runner disables it where the
    provider supports doing so; this is the belt-and-braces path for when it
    comes back anyway, so reasoning prose is never parsed as labels.
    """
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
