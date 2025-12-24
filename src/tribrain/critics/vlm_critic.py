from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..types import CriticResult, FailureMode, GoalSpec, WMArtifact
from .base import Critic

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

try:
    from PIL import Image  # type: ignore
except Exception:  # pragma: no cover
    Image = None  # type: ignore


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_frames(path: Path, n: int = 3, resize_w: int = 448) -> list[Any]:
    if cv2 is None or Image is None:
        raise RuntimeError("VLM critic requires opencv-python and pillow")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    frames: list[Any] = []
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            # Fallback: attempt to read sequentially.
            idxs = [0, 8, 16]
        else:
            # Evenly spaced sampling (robust to variable FPS / length).
            idxs = [int(round(i * (total - 1) / max(1, n - 1))) for i in range(n)]

        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(0, idx)))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            h, w = frame.shape[:2]
            if w > resize_w:
                scale = resize_w / float(w)
                frame = cv2.resize(frame, (resize_w, int(max(1, h * scale))))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError("No frames extracted")
    return frames


def _parse_json(text: str) -> dict[str, Any]:
    # Try direct JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = _JSON_RE.search(text)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


@dataclass
class HFVlmCritic(Critic):
    """Real VLM critic using HuggingFace Transformers.

    It evaluates:
      - instruction_following (task)
      - physics_sanity (physics)
    returning scores in [0,1].

    Notes:
    - This is "real" (calls an actual VLM), but it is optional: you must install
      transformers + torch and have a compatible model.
    - For Colab, small VLMs (2B) are recommended.
    """

    name: str = "vlm_hf"
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct"
    device: str = "auto"  # auto|cpu|cuda
    max_new_tokens: int = 256
    temperature: float = 0.0
    max_retries: int = 2

    _model: Any = None
    _processor: Any = None

    def _lazy_load(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        try:
            import torch  # type: ignore
            from transformers import AutoModelForVision2Seq, AutoProcessor  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("Install dependencies: pip install torch transformers accelerate") from e

        # Choose device / dtype safely.
        if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
            dev = "cuda"
            dtype = getattr(torch, "float16", torch.float32)
        else:
            dev = "cpu"
            dtype = torch.float32

        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)

        # device_map="auto" requires accelerate; fallback cleanly.
        device_map = None
        if dev == "cuda" and self.device == "auto":
            try:
                import accelerate  # type: ignore  # noqa: F401

                device_map = "auto"
            except Exception:
                device_map = None

        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        if device_map is None:
            self._model.to(dev)

        self._model.eval()

    def evaluate(self, artifact: WMArtifact, goal: GoalSpec) -> CriticResult:
        self._lazy_load()
        frames = _extract_frames(artifact.path, n=3)

        # Build a strict JSON request
        prompt = (
            "You are a strict video critic for a robot world model.\n"
            "Evaluate ONLY from what is visible in the frames.\n"
            "Return JSON with keys: instruction_following, physics_sanity, failure_modes, notes.\n"
            "Scales: instruction_following and physics_sanity are floats in [0,1].\n"
            f"Instruction: {goal.instruction}\n"
            "Do not include any text outside JSON."
        )

        # Budgeted retries for strict JSON.
        data: dict[str, Any] = {}
        out = ""
        for _ in range(max(1, int(self.max_retries) + 1)):
            out = self._generate(frames, prompt)
            data = _parse_json(out)
            if data:
                break

        inst = float(max(0.0, min(1.0, float(data.get("instruction_following", 0.0) or 0.0))))
        phys = float(max(0.0, min(1.0, float(data.get("physics_sanity", 0.0) or 0.0))))

        notes = str(data.get("notes", ""))[:300]
        # Map to FailureModes used by TriBrain
        scores: Mapping[FailureMode, float] = {
            FailureMode.TASK: inst,
            FailureMode.PHYSICS: phys,
        }
        extra = {"raw": data, "raw_text": out[:800]}
        return CriticResult(self.name, scores, notes=notes, extra=extra)

    def _generate(self, frames: list[Any], prompt: str) -> str:
        proc = self._processor
        model = self._model

        # Some processors support chat templates; try the most compatible path first.
        if hasattr(proc, "apply_chat_template"):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        # Attach multiple images if supported by model; otherwise first one will be used.
                        *[{"type": "image", "image": img} for img in frames],
                    ],
                }
            ]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=text, images=frames, return_tensors="pt")
        else:
            inputs = proc(text=prompt, images=frames, return_tensors="pt")

        # Move tensors to model device if possible
        for k, v in list(inputs.items()):
            if hasattr(v, "to") and hasattr(model, "device"):
                inputs[k] = v.to(model.device)

        gen = model.generate(
            **inputs,
            max_new_tokens=int(self.max_new_tokens),
            do_sample=bool(float(self.temperature) > 1e-9),
            temperature=float(self.temperature),
        )
        # Decode last sequence
        if hasattr(proc, "batch_decode"):
            txt = str(proc.batch_decode(gen, skip_special_tokens=True)[0])
        else:
            txt = gen[0].decode("utf-8", errors="ignore")  # type: ignore
        return str(txt)
