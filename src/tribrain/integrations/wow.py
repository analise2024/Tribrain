from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..types import GenerationSpec, GoalSpec, WMArtifact
from .subprocess_world_model import _find_latest_video


@dataclass
class WowWorldModel:
    """Adapter that calls WoW inference scripts from a local WoW checkout."""
    wow_repo: Path
    model: str = "dit-2b"  # "dit-2b", "dit-7b", "wan-14b" (depending on upstream scripts)
    extra_args: str = ""

    def _script(self) -> Path:
        if self.model == "dit-2b":
            return self.wow_repo / "scripts" / "infer_wow_dit_2b.py"
        if self.model == "dit-7b":
            return self.wow_repo / "scripts" / "infer_wow_dit_7b.py"
        if self.model == "wan-14b":
            return self.wow_repo / "demo" / "wan_infer_demo.py"
        raise ValueError(f"Unknown model: {self.model}")

    def predict(self, goal: GoalSpec, out_dir: Path, spec: GenerationSpec) -> WMArtifact:
        out_dir.mkdir(parents=True, exist_ok=True)
        script = self._script()
        if not script.exists():
            raise FileNotFoundError(f"WoW script not found: {script}")

        # Common args: most WoW scripts accept --prompt and --save_dir (or similar).
        # We support both by trying a primary set; if it fails, user can pass extra_args.
        seed = spec.seed if spec.seed is not None else 0

        # Try to be conservative: pass common flags; the user can override with extra_args.
        base_cmd = f"python {shlex.quote(str(script))} --prompt {shlex.quote(goal.instruction)} --seed {seed} --save_dir {shlex.quote(str(out_dir))} {self.extra_args}"
        cmd = shlex.split(base_cmd)

        proc = subprocess.run(
            cmd,
            cwd=str(self.wow_repo),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            # Provide stdout for debugging and hint to use extra_args
            raise RuntimeError(
                "WoW inference failed. You may need to adjust flags via --wow_extra_args.\n"
                f"Command: {cmd}\nOutput:\n{proc.stdout}"
            )

        video = _find_latest_video(out_dir)
        return WMArtifact(path=video, metadata={"stdout": proc.stdout, "cmd": cmd, "wow_repo": str(self.wow_repo)})
