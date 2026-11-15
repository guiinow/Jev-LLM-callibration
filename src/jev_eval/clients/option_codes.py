"""Single-token option codes (A, B, ..., Z, AA, AB, ...) for option-probability scoring.

The model is asked to answer with the code of an option; its next-token distribution over the
codes gives one probability per option. That only works if every code is ONE token in the
model's vocabulary, so codes the tokenizer splits are skipped (as in the Jev benchmark).
"""
import itertools
import string
from typing import Callable, Iterator


def code_candidates() -> Iterator[str]:
    yield from string.ascii_uppercase
    for first, second in itertools.product(string.ascii_uppercase, repeat=2):
        yield first + second


def assign_codes(labels: list[str], is_single_token: Callable[[str], bool]) -> dict[str, str]:
    """label -> code, in label order, using the first single-token codes."""
    codes: list[str] = []
    for code in code_candidates():
        if is_single_token(code):
            codes.append(code)
            if len(codes) == len(labels):
                return dict(zip(labels, codes))
    raise ValueError(f"Only {len(codes)} single-token codes are available for {len(labels)} options.")
