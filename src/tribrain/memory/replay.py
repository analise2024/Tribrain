from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .episodic import Episode, EpisodeStore


@dataclass
class ReplayConfig:
    k: int = 256
    strategy: str = "uniform"  # uniform|nearest
    seed: int = 0


class ReplayEngine:
    """Replay updates for continual improvement.

    This is a practical self-improvement mechanism: we can update the controller and
    meta-value model from stored episodes without regenerating videos.
    """

    def __init__(self, store: EpisodeStore, cfg: ReplayConfig | None = None):
        self.store = store
        self.cfg = cfg or ReplayConfig()

    def sample(self, query_ctx: np.ndarray | None = None) -> list[Episode]:
        if self.cfg.strategy == "nearest" and query_ctx is not None:
            return self.store.nearest(query_ctx, k=self.cfg.k)
        return self.store.sample(self.cfg.k, seed=self.cfg.seed)

    def replay_update(
        self,
        controller: _ControllerUpdater | None,
        meta_value_model: _MetaValueUpdater | None,
        query_ctx: np.ndarray | None = None,
    ) -> int:
        eps = self.sample(query_ctx=query_ctx)
        if not eps:
            return 0

        n = 0
        for ep in eps:
            if controller is not None:
                # Expect controller.update(arm, ctx_vec, reward)
                with suppress(Exception):
                    controller.update(ep.arm, ep.ctx_vec, float(ep.smoothed_reward))
            if meta_value_model is not None:
                # Expect meta_value_model.update(meta_x, delta_score)
                with suppress(Exception):
                    meta_value_model.update(ep.meta_x, float(ep.delta_score))
            n += 1
        return n


class _ControllerUpdater(Protocol):
    def update(self, arm: str, ctx_vec: np.ndarray, reward: float) -> None: ...


class _MetaValueUpdater(Protocol):
    def update(self, meta_x: np.ndarray, delta_score: float) -> None: ...
