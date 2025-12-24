from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..types import GenerationSpec, GoalSpec, WMArtifact

_VIDEO_EXTS = (".mp4", ".webm", ".mov", ".avi", ".mkv")


def _find_latest_video(out_dir: Path) -> Path:
    candidates = [p for p in out_dir.rglob("*") if p.suffix.lower() in _VIDEO_EXTS and p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No video found under: {out_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


@dataclass
class SubprocessWorldModel:
    """Wraps any external generator command.

    The command may include placeholders:
    - {prompt}  (shell-escaped as a single argument)
    - {out_dir}
    - {seed}
    """
    cmd_template: str

    def predict(self, goal: GoalSpec, out_dir: Path, spec: GenerationSpec) -> WMArtifact:
        out_dir.mkdir(parents=True, exist_ok=True)
        seed = spec.seed if spec.seed is not None else 0
        # Important: keep prompt as one argument (quoted) by inserting it as a token.
        rendered = self.cmd_template.format(prompt=goal.instruction, out_dir=str(out_dir), seed=seed)
        cmd = shlex.split(rendered)

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Generator failed (code={proc.returncode}). Output:\n{proc.stdout}")

        video = _find_latest_video(out_dir)
        return WMArtifact(path=video, metadata={"stdout": proc.stdout, "cmd": cmd})
