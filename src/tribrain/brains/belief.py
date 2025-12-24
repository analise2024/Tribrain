from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..types import CriticResult, FailureMode

if TYPE_CHECKING:
    from ..storage.calibration import CalibrationStore


@dataclass
class DirichletBelief:
    """A simple Dirichlet belief over failure modes.

    Higher alpha for a mode => more belief that this mode is the main source of failure.
    """
    alpha: dict[FailureMode, float] = field(default_factory=lambda: {m: 1.0 for m in FailureMode})

    def update_from_scores(self, fused_scores: Mapping[FailureMode, float], strength: float = 2.0) -> None:
        # Evidence is larger when score is low.
        for m in FailureMode:
            s = float(max(0.0, min(1.0, fused_scores.get(m, 0.0))))
            self.alpha[m] = float(self.alpha.get(m, 1.0) + strength * (1.0 - s))

    def probs(self) -> dict[FailureMode, float]:
        keys = list(FailureMode)
        a = np.asarray([max(1e-6, float(self.alpha.get(k, 1.0))) for k in keys], dtype=np.float64)
        p = a / float(np.sum(a))
        return {k: float(v) for k, v in zip(keys, p, strict=True)}


@dataclass
class BetaReliability:
    """Beta(a,b) reliability for a critic: mean = a/(a+b)."""
    a: float = 1.0
    b: float = 1.0

    def mean(self) -> float:
        return float(self.a / (self.a + self.b))

    def update(self, success: bool, weight: float = 1.0) -> None:
        w = float(max(0.0, weight))
        if success:
            self.a += w
        else:
            self.b += w


@dataclass
class BeliefState:
    modes: DirichletBelief = field(default_factory=DirichletBelief)
    critic_reliability: dict[str, BetaReliability] = field(default_factory=dict)

    def critic_weight(self, critic_name: str) -> float:
        rel = self.critic_reliability.get(critic_name)
        return 1.0 if rel is None else float(rel.mean())

    def fuse(self, results: list[CriticResult]) -> dict[FailureMode, float]:
        # Weighted average fusion over critics, per failure mode.
        # If a mode isn't provided by any critic, it stays 0.
        sums: dict[FailureMode, float] = {m: 0.0 for m in FailureMode}
        wsum: dict[FailureMode, float] = {m: 0.0 for m in FailureMode}
        for r in results:
            w = self.critic_weight(r.critic_name)
            for m, s in r.scores.items():
                sm = float(max(0.0, min(1.0, s)))
                sums[m] += w * sm
                wsum[m] += w
        fused = {}
        for m in FailureMode:
            if wsum[m] <= 1e-9:
                fused[m] = 0.0
            else:
                fused[m] = float(sums[m] / wsum[m])
        return fused

    def update(self, results: list[CriticResult], fused_scores: Mapping[FailureMode, float]) -> None:
        self.modes.update_from_scores(fused_scores)

    def update_reliability(self, critic_name: str, success: bool, weight: float = 1.0) -> None:
        if critic_name not in self.critic_reliability:
            self.critic_reliability[critic_name] = BetaReliability()
        self.critic_reliability[critic_name].update(success=success, weight=weight)

    def sync_reliability_from_store(self, store: CalibrationStore) -> None:
        """Load critic reliabilities from a persistent store.

        This is the practical bridge to human-in-the-loop calibration: once humans label whether a critic
        was correct for a given run, we update a Beta distribution and reuse it in future runs.
        """
        # Avoid import cycles
        from ..storage.calibration import CalibrationStore

        if not isinstance(store, CalibrationStore):
            return
        # Load all known critic names from the store.
        for name in store.list_critics():
            self.critic_reliability[name] = store.reliability(name)

    def to_dict(self) -> dict:
        return {
            "modes": {k.name: float(v) for k, v in self.modes.alpha.items()},
            "critic_reliability": {k: {"a": v.a, "b": v.b} for k, v in self.critic_reliability.items()},
        }

    def load_dict(self, d: dict) -> None:
        if not isinstance(d, dict):
            return
        modes = d.get("modes", {})
        if isinstance(modes, dict):
            for k, v in modes.items():
                try:
                    fm = FailureMode[k]
                    self.modes.alpha[fm] = float(v)
                except Exception:
                    continue
        rel = d.get("critic_reliability", {})
        if isinstance(rel, dict):
            for name, ab in rel.items():
                if not isinstance(ab, dict):
                    continue
                self.critic_reliability[name] = BetaReliability(a=float(ab.get("a", 1.0)), b=float(ab.get("b", 1.0)))
