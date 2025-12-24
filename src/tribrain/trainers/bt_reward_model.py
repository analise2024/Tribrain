from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..features.text_pair import pair_feature
from ..reward.preference import BradleyTerryRewardModel, PreferenceExample


@dataclass(frozen=True)
class BTTrainConfig:
    """Training config for the lightweight Bradley–Terry text reward model.

    This model is intentionally CPU-first and fast. It can be used to score
    candidate *prompts* (text) conditioned on the goal instruction.
    """

    bow_dim: int = 64  # feature dim per side; total dim = 2*bow_dim
    epochs: int = 3
    lr: float = 0.12
    l2: float = 1e-3
    seed: int = 0


def train_bt_text_reward_model(pairs_jsonl: Path, out_path: Path, cfg: BTTrainConfig | None = None) -> Path:
    cfg = cfg or BTTrainConfig()
    pairs_jsonl = Path(pairs_jsonl)
    out_path = Path(out_path)

    examples: list[PreferenceExample] = []
    for line in pairs_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj: Any
        try:
            obj = json.loads(line)
        except Exception:
            continue
        instr = str(obj.get("goal_instruction", ""))
        chosen = str(obj.get("chosen_prompt", ""))
        rejected = str(obj.get("rejected_prompt", ""))
        if not chosen or not rejected:
            continue
        x_c = pair_feature(instr, chosen, int(cfg.bow_dim))
        x_r = pair_feature(instr, rejected, int(cfg.bow_dim))
        examples.append(PreferenceExample(chosen=x_c, rejected=x_r, weight=1.0))

    if not examples:
        raise ValueError("No usable preference pairs found.")

    # Shuffle deterministically for training stability.
    rng = np.random.default_rng(int(cfg.seed))
    rng.shuffle(examples)

    dim = int(2 * int(cfg.bow_dim))
    model = BradleyTerryRewardModel(dim=dim, lr=float(cfg.lr), l2=float(cfg.l2))
    model.fit(examples, epochs=int(cfg.epochs))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(out_path)
    return out_path
