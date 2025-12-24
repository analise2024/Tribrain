from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TextPreferencePair:
    instruction: str
    chosen: str
    rejected: str


def load_text_pairs(pairs_jsonl: Path) -> list[TextPreferencePair]:
    pairs: list[TextPreferencePair] = []
    for line in pairs_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        pairs.append(
            TextPreferencePair(
                instruction=str(obj.get("goal_instruction", "")),
                chosen=str(obj.get("chosen_prompt", "")),
                rejected=str(obj.get("rejected_prompt", "")),
            )
        )
    return pairs


def vectorize_pairs(
    pairs: Iterable[TextPreferencePair],
    feature_fn: Callable[[str, str], np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Turn text pairs into feature vectors using a provided feature_fn(text)->np.ndarray.

    This allows training the small Bradley-Terry model using any embedding backend you like
    (e.g., bag-of-words, sentence transformers, etc.)."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for p in pairs:
        x_c = feature_fn(p.instruction, p.chosen)
        x_r = feature_fn(p.instruction, p.rejected)
        out.append((x_c, x_r))
    return out
