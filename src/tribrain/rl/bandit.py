from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Arm:
    name: str
    # Beta prior
    a: float = 1.0
    b: float = 1.0

    def sample(self, rng: random.Random) -> float:
        # Thompson sampling for Bernoulli reward
        return rng.betavariate(self.a, self.b)

    def update(self, reward: float, weight: float = 1.0) -> None:
        # reward in [0,1]
        r = max(0.0, min(1.0, float(reward))) * float(weight)
        self.a += r
        self.b += (1.0 - r)


class PromptEditBandit:
    """Ultra-light RL for choosing which edit templates to apply.

    This is designed to be:
    - stable,
    - low compute,
    - fast to update from sparse feedback,
    - compatible with human-in-the-loop (reward can be a human label).
    """

    def __init__(self, arms: list[str] | None = None, seed: int = 0):
        names = arms or [
            "physics",
            "task",
            "smoothness",
            "quality",
            "minimal",
        ]
        self.arms = {n: Arm(name=n) for n in names}
        self.rng = random.Random(seed)
        self.last_choice: str | None = None

    def set_seed(self, seed: int) -> None:
        self.rng = random.Random(int(seed))

    def choose(self) -> str:
        best = None
        best_v = -1.0
        for a in self.arms.values():
            v = a.sample(self.rng)
            if v > best_v:
                best_v = v
                best = a.name
        self.last_choice = best
        return str(best)

    def update(self, reward: float, arm_name: str | None = None, weight: float = 1.0) -> None:
        name = arm_name or self.last_choice
        if not name or name not in self.arms:
            return
        self.arms[name].update(reward=reward, weight=weight)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: {"a": v.a, "b": v.b} for k, v in self.arms.items()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(data, dict):
            return
        for name, ab in data.items():
            if name in self.arms and isinstance(ab, dict):
                self.arms[name].a = float(ab.get("a", self.arms[name].a))
                self.arms[name].b = float(ab.get("b", self.arms[name].b))