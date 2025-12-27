from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .episodic import Episode, EpisodeStore


@dataclass(frozen=True)
class ReplayConfig:
    """Replay configuration.

    strategy:
      - uniform: uniform random sample
      - nearest: cosine NN to query_ctx (falls back to uniform if query_ctx is None)
      - biphase: two-phase selector (offline consolidation in replay-update)

    The biphase fields are only used when strategy == 'biphase'.
    """

    k: int = 256
    strategy: str = "uniform"  # uniform|nearest|biphase
    seed: int = 0

    # Bi-phase tuning (safe defaults)
    biphase_online_batch_size: int = 8
    biphase_offline_batch_size: int = 64
    biphase_interleave_ratio: float = 0.5
    biphase_online_lr: float = 0.002
    biphase_offline_lr: float = 0.01
    biphase_replay_penalty_alpha: float = 0.05


class ReplayEngine:
    def __init__(self, store: EpisodeStore, cfg: ReplayConfig):
        self.store = store
        self.cfg = cfg

    def _sample(self, query_ctx: np.ndarray | None = None) -> tuple[list[Episode], str, float]:
        """Return (episodes, phase, lr_scale)."""
        strat = str(self.cfg.strategy).lower().strip()
        if strat == "nearest":
            if query_ctx is None:
                return (self.store.sample(int(self.cfg.k), seed=int(self.cfg.seed)), "single", 1.0)
            return (self.store.nearest(query_ctx, int(self.cfg.k)), "single", 1.0)

        if strat == "biphase":
            from .biphase import BiPhaseReplayConfig, BiPhaseReplayEngine

            bcfg = BiPhaseReplayConfig(
                online_batch_size=int(self.cfg.biphase_online_batch_size),
                offline_batch_size=int(self.cfg.biphase_offline_batch_size),
                interleave_ratio=float(self.cfg.biphase_interleave_ratio),
                online_lr=float(self.cfg.biphase_online_lr),
                offline_lr=float(self.cfg.biphase_offline_lr),
                replay_penalty_alpha=float(self.cfg.biphase_replay_penalty_alpha),
                seed=int(self.cfg.seed),
            )
            ben = BiPhaseReplayEngine(self.store, bcfg)
            eps = ben.select_offline_batch()
            return (eps, "offline", float(bcfg.offline_lr))

        # default: uniform
        return (self.store.sample(int(self.cfg.k), seed=int(self.cfg.seed)), "single", 1.0)

    def replay_update(
        self,
        *,
        controller: _ControllerUpdater | None = None,
        meta_value_model: _MetaValueUpdater | None = None,
        query_ctx: np.ndarray | None = None,
    ) -> int:
        """Replay a batch of episodes to update lightweight models.

        This is best-effort: failures in one updater do not stop the others.
        Returns number of episodes processed.
        """
        eps, _phase, lr = self._sample(query_ctx=query_ctx)
        if not eps:
            return 0

        n = 0
        for ep in eps:
            if controller is not None:
                # controller.update(arm, ctx_vec, reward)
                # Use lr as a light scaling (no hard dependency on controller internals).
                with suppress(Exception):
                    controller.update(ep.arm, ep.ctx_vec, float(ep.smoothed_reward) * float(lr))
            if meta_value_model is not None:
                # meta_value_model.update(meta_x, delta_score)
                with suppress(Exception):
                    meta_value_model.update(ep.meta_x, float(ep.delta_score) * float(lr))
            n += 1
        return n


class _ControllerUpdater(Protocol):
    def update(self, arm: str, ctx_vec: np.ndarray, reward: float) -> None: ...


class _MetaValueUpdater(Protocol):
    def update(self, meta_x: np.ndarray, delta_score: float) -> None: ...

