from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..types import FailureMode


@dataclass
class MetaConfig:
    target_score: float = 0.85
    patience: int = 2
    min_delta: float = 0.01
    max_iters: int = 4
    drift_threshold: float = 0.15  # how much a key dimension can fall while overall rises


@dataclass
class MetaState:
    iter_idx: int = 0
    best_score: float = -1.0
    best_iter: int = -1
    plateau_steps: int = 0
    last_score: float = -1.0
    last_dims: dict[FailureMode, float] = field(default_factory=dict)
    drift_flag: bool = False
    stop_reason: str = ""


class MetaController:
    def __init__(self, cfg: MetaConfig | None = None):
        self.cfg = cfg or MetaConfig()

    @staticmethod
    def overall_score(fused_scores: Mapping[FailureMode, float]) -> float:
        # Conservative: average across available dimensions, but ignore OTHER unless provided.
        dims = [FailureMode.PHYSICS, FailureMode.TASK, FailureMode.SMOOTHNESS, FailureMode.QUALITY]
        vals = [float(max(0.0, min(1.0, fused_scores.get(d, 0.0)))) for d in dims]
        return float(sum(vals) / len(vals))

    def update(self, state: MetaState, fused_scores: Mapping[FailureMode, float]) -> float:
        score = self.overall_score(fused_scores)
        dims = {m: float(max(0.0, min(1.0, fused_scores.get(m, 0.0)))) for m in FailureMode}

        # Best tracking
        if score > state.best_score:
            state.best_score = score
            state.best_iter = state.iter_idx

        # Plateau detection
        if state.last_score >= 0.0 and (score - state.last_score) < self.cfg.min_delta:
            state.plateau_steps += 1
        else:
            state.plateau_steps = 0

        # Drift detection: overall improves but key dimension collapses
        state.drift_flag = False
        if state.last_score >= 0.0 and score > state.last_score:
            for key in (FailureMode.TASK, FailureMode.PHYSICS):
                prev = float(state.last_dims.get(key, dims[key]))
                if prev - dims[key] > self.cfg.drift_threshold:
                    state.drift_flag = True
                    break

        state.last_score = score
        state.last_dims = dims
        return score

    def should_stop(self, state: MetaState) -> bool:
        if state.last_score >= self.cfg.target_score:
            state.stop_reason = "target_reached"
            return True
        if state.plateau_steps >= self.cfg.patience:
            state.stop_reason = "plateau"
            return True
        if state.iter_idx + 1 >= self.cfg.max_iters:
            state.stop_reason = "max_iters"
            return True
        return False
