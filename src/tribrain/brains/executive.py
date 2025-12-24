from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from ..types import FailureMode


@dataclass
class ExecutiveConfig:
    max_append_chars: int = 800
    topk_modes: int = 2


@dataclass
class ExecutiveState:
    prompt_history: list[str] = field(default_factory=list)


class ExecutiveRefiner:
    """Deterministic refiner that edits prompts based on belief over failure modes.

    This is intentionally conservative: it never deletes the original instruction and only appends
    short constraints. You can replace it with an LLM-based refiner via the same interface.
    """

    def __init__(self, cfg: ExecutiveConfig | None = None):
        self.cfg = cfg or ExecutiveConfig()

    def refine(
        self,
        prompt: str,
        belief_probs: Mapping[FailureMode, float],
        fused_scores: Mapping[FailureMode, float],
        drift_flag: bool = False,
        preferred_modes: list[FailureMode] | None = None,
    ) -> str:
        base = prompt.strip()
        additions: list[str] = []

        # pick top failure modes (excluding OTHER) by belief probability
        items = [(m, float(belief_probs.get(m, 0.0))) for m in FailureMode if m != FailureMode.OTHER]
        items.sort(key=lambda x: x[1], reverse=True)
        top: list[FailureMode] = [m for m, _p in items[: self.cfg.topk_modes]]

        # RL/policy can request certain modes be emphasized first.
        if preferred_modes:
            pref: list[FailureMode] = [
                m
                for m in preferred_modes
                if m in (FailureMode.PHYSICS, FailureMode.TASK, FailureMode.SMOOTHNESS, FailureMode.QUALITY)
            ]
            top = pref + [m for m in top if m not in pref]
            top = top[: self.cfg.topk_modes]

        # If drift is happening, prioritize instruction/task adherence
        if drift_flag and FailureMode.TASK not in top:
            top = [FailureMode.TASK] + [m for m in top if m != FailureMode.TASK]
            top = top[: self.cfg.topk_modes]

        for m in top:
            if m == FailureMode.PHYSICS:
                additions.append(
                    "Physics constraints: motions must be physically plausible; avoid object interpenetration; "
                    "keep stable contacts; respect gravity; grasp firmly before lifting; no teleporting."
                )
            elif m == FailureMode.SMOOTHNESS:
                additions.append(
                    "Motion constraints: keep movements smooth and continuous; avoid sudden jumps; "
                    "camera remains stable; transitions are temporally consistent."
                )
            elif m == FailureMode.QUALITY:
                additions.append(
                    "Rendering constraints: maintain consistent lighting and textures across frames; avoid flicker; "
                    "avoid blurry artifacts; keep object boundaries stable."
                )
            elif m == FailureMode.TASK:
                additions.append(
                    "Task constraints: strictly complete the instruction end-to-end; verify the final state matches the goal."
                )

        if not additions:
            return base

        suffix = "\n\n" + "\n".join(f"- {a}" for a in additions)
        out = (base + suffix).strip()
        if len(out) > self.cfg.max_append_chars:
            out = out[: self.cfg.max_append_chars].rstrip() + "…"
        return out
