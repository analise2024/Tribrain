from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class FailureMode(str, Enum):
    PHYSICS = "physics"
    TASK = "task"
    SMOOTHNESS = "smoothness"
    QUALITY = "quality"
    OTHER = "other"


@dataclass(frozen=True)
class GoalSpec:
    """High-level goal / instruction for the rollout."""
    instruction: str
    tags: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationSpec:
    """Generation settings passed to the world model adapter."""
    seed: int | None = None
    steps: int | None = None
    fps: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WMArtifact:
    """A generated artifact produced by a world model."""
    path: Path
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CriticResult:
    """Scores are in [0, 1] where higher is better."""
    critic_name: str
    scores: Mapping[FailureMode, float]
    notes: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def score(self, mode: FailureMode) -> float:
        v = self.scores.get(mode, 0.0)
        return float(max(0.0, min(1.0, v)))
