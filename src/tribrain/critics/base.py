from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..types import CriticResult, FailureMode, GoalSpec, WMArtifact


class Critic(Protocol):
    name: str

    def evaluate(self, artifact: WMArtifact, goal: GoalSpec) -> CriticResult:
        ...


@dataclass(frozen=True)
class CriticBundle:
    critics: Sequence[Critic]

    def evaluate_all(self, artifact: WMArtifact, goal: GoalSpec) -> list[CriticResult]:
        out: list[CriticResult] = []
        for c in self.critics:
            try:
                out.append(c.evaluate(artifact, goal))
            except Exception as e:
                # Non-fatal: a single critic shouldn't crash the run.
                out.append(
                    CriticResult(
                        getattr(c, "name", "unknown"),
                        {
                            FailureMode.PHYSICS: 0.0,
                            FailureMode.TASK: 0.0,
                            FailureMode.SMOOTHNESS: 0.0,
                            FailureMode.QUALITY: 0.0,
                            FailureMode.OTHER: 0.0,
                        },
                        notes=f"critic_error:{type(e).__name__}",
                        extra={"error": str(e)[:400]},
                    )
                )
        return out
