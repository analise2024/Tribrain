from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PreferenceExample:
    """A pairwise preference: chosen should be preferred to rejected.

    Features are numeric vectors describing each candidate (e.g., critic scores, costs, ontology features).
    """

    chosen: np.ndarray
    rejected: np.ndarray
    weight: float = 1.0


def sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable
    x = np.clip(x, -60, 60)
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float32)


@dataclass
class BradleyTerryRewardModel:
    """Small preference-based reward model.

    Implements a Bradley-Terry / logistic model on feature differences:
        P(chosen > rejected) = sigmoid(w · (x_chosen - x_rejected))

    This is CPU-friendly, online-trainable, and works well as a stabilizer for noisy critics.

    Notes:
    - This is not a neural net; it is intentionally cheap and interpretable.
    - For LoRA fine-tuning of a larger reward model, see scripts/train_lora_reward_model.py
    """

    dim: int
    lr: float = 0.08
    l2: float = 1e-3
    w: np.ndarray | None = None
    n_updates: int = 0

    def __post_init__(self) -> None:
        if self.w is None:
            self.w = np.zeros((self.dim,), dtype=np.float32)

    def prob_prefer(self, x_chosen: np.ndarray, x_rejected: np.ndarray) -> float:
        d = (np.asarray(x_chosen, dtype=np.float32).reshape(-1) -
             np.asarray(x_rejected, dtype=np.float32).reshape(-1))
        if d.shape[0] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {d.shape[0]}")
        w = self.w
        assert w is not None
        return float(sigmoid(np.array(np.dot(w, d), dtype=np.float32)))

    def reward(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got {x.shape[0]}")
        w = self.w
        assert w is not None
        return float(np.dot(w, x))

    def update_pair(self, ex: PreferenceExample) -> float:
        x_c = np.asarray(ex.chosen, dtype=np.float32).reshape(-1)
        x_r = np.asarray(ex.rejected, dtype=np.float32).reshape(-1)
        if x_c.shape[0] != self.dim or x_r.shape[0] != self.dim:
            raise ValueError("Feature dim mismatch")
        d = x_c - x_r
        # maximize log sigmoid(w·d) with L2
        w = self.w
        assert w is not None
        z = float(np.dot(w, d))
        p = float(sigmoid(np.array(z)))
        grad = (1.0 - p) * d  # derivative of log sigmoid
        self.w = (w + (self.lr * ex.weight) * grad - self.lr * self.l2 * w).astype(np.float32)
        self.n_updates += 1
        return p

    def fit(self, examples: Iterable[PreferenceExample], epochs: int = 1) -> None:
        for _ in range(max(1, epochs)):
            for ex in examples:
                self.update_pair(ex)

    def save(self, path: Path) -> None:
        obj = {
            "dim": self.dim,
            "lr": self.lr,
            "l2": self.l2,
            "w": self.w.tolist() if self.w is not None else None,
            "n_updates": self.n_updates,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> BradleyTerryRewardModel:
        obj = json.loads(path.read_text(encoding="utf-8"))
        w = np.asarray(obj.get("w", []), dtype=np.float32)
        rm = cls(dim=int(obj["dim"]), lr=float(obj.get("lr", 0.08)), l2=float(obj.get("l2", 1e-3)), w=w)
        rm.n_updates = int(obj.get("n_updates", 0))
        return rm
