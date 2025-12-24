from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..types import CriticResult, FailureMode


@dataclass(frozen=True)
class IterationTrace:
    iter_idx: int
    prompt: str
    artifact_path: str
    critic_results: list[Mapping[str, Any]]
    fused_scores: Mapping[str, float]
    belief_probs: Mapping[str, float]
    overall_score: float
    meta: Mapping[str, Any]


class JsonlTraceWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _critic_to_json(r: CriticResult) -> dict[str, Any]:
        return {
            "critic_name": r.critic_name,
            "scores": {m.value: float(v) for m, v in r.scores.items()},
            "notes": r.notes,
            "extra": dict(r.extra),
        }

    def append(
        self,
        iter_idx: int,
        prompt: str,
        artifact_path: str | Path,
        critic_results: list[CriticResult],
        fused_scores: Mapping[FailureMode, float],
        belief_probs: Mapping[FailureMode, float],
        overall_score: float,
        meta: Mapping[str, Any],
    ) -> None:
        record = IterationTrace(
            iter_idx=iter_idx,
            prompt=prompt,
            artifact_path=str(artifact_path),
            critic_results=[self._critic_to_json(r) for r in critic_results],
            fused_scores={m.value: float(v) for m, v in fused_scores.items()},
            belief_probs={m.value: float(v) for m, v in belief_probs.items()},
            overall_score=float(overall_score),
            meta=dict(meta),
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            # Make runs resilient to crashes/power loss (best-effort).
            try:
                f.flush()
                os.fsync(f.fileno())
            except Exception:
                pass
