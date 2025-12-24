from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Example:
    instruction: str
    prompt_used: str
    artifact_path: str
    overall_score: float
    meta: dict[str, Any]


def export_preference_dataset(run_dir: Path, out_path: Path, min_score: float = 0.7) -> int:
    """Export a JSONL dataset from a TriBrain run directory.

    The JSONL format is meant to be model-agnostic:
    - you can label examples as good/bad, or use them for training a reward model.
    - you can pair them with a "bad" alternative to make preference pairs.
    """

    trace_path = run_dir / "trace.jsonl"
    if not trace_path.exists():
        raise FileNotFoundError(f"trace.jsonl not found in {run_dir}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with trace_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            score = float(obj.get("overall_score", 0.0))
            if score < float(min_score):
                continue
            ex = Example(
                instruction=str(obj.get("meta", {}).get("goal_instruction", "")),
                prompt_used=str(obj.get("prompt", "")),
                artifact_path=str(obj.get("artifact_path", "")),
                overall_score=score,
                meta=dict(obj.get("meta", {})),
            )
            f_out.write(json.dumps(ex.__dict__, ensure_ascii=False) + "\n")
            n += 1
    return n
