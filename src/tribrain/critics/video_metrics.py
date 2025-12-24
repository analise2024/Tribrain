from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..types import CriticResult, FailureMode, GoalSpec, WMArtifact
from .base import Critic


def _iter_frames(path: Path, max_frames: int = 64, resize_w: int = 256) -> Iterable[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frames = 0
    try:
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            # BGR->GRAY
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            if w > resize_w:
                scale = resize_w / float(w)
                gray = cv2.resize(gray, (resize_w, int(h * scale)), interpolation=cv2.INTER_AREA)
            yield gray
            frames += 1
    finally:
        cap.release()


def _safe_score_from_positive(x: float, scale: float = 1.0) -> float:
    """Maps x>=0 to (0,1] with a soft decay."""
    x = max(0.0, float(x))
    return float(1.0 / (1.0 + (x / max(1e-6, scale))))


@dataclass
class FlickerCritic(Critic):
    """Penalizes high frame-to-frame intensity jumps (flicker/jitter)."""
    name: str = "flicker"
    max_frames: int = 64

    def evaluate(self, artifact: WMArtifact, goal: GoalSpec) -> CriticResult:
        frames = list(_iter_frames(artifact.path, max_frames=self.max_frames))
        if len(frames) < 2:
            return CriticResult(self.name, {FailureMode.QUALITY: 0.0}, notes="not enough frames")
        diffs = []
        for a, b in zip(frames[:-1], frames[1:], strict=True):
            diffs.append(float(np.mean(np.abs(b.astype(np.float32) - a.astype(np.float32)))))
        mean_diff = float(np.mean(diffs))
        score = _safe_score_from_positive(mean_diff, scale=12.0)
        return CriticResult(self.name, {FailureMode.SMOOTHNESS: score, FailureMode.QUALITY: score},
                            notes=f"mean_luma_diff={mean_diff:.3f}")


@dataclass
class FlowSmoothnessCritic(Critic):
    """Uses optical flow magnitude stability as a proxy for physically plausible, smooth motion."""
    name: str = "flow_smoothness"
    max_frames: int = 48

    def evaluate(self, artifact: WMArtifact, goal: GoalSpec) -> CriticResult:
        frames = list(_iter_frames(artifact.path, max_frames=self.max_frames))
        if len(frames) < 3:
            return CriticResult(self.name, {FailureMode.PHYSICS: 0.0}, notes="not enough frames")

        mags_list: list[float] = []
        prev = frames[0]
        for cur in frames[1:]:
            flow0 = np.zeros((prev.shape[0], prev.shape[1], 2), dtype=np.float32)
            flow = cv2.calcOpticalFlowFarneback(
                prev, cur, flow0,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mags_list.append(float(np.mean(mag)))
            prev = cur

        mags = np.asarray(mags_list, dtype=np.float32)
        # instability = variance of motion magnitude over time
        instability = float(np.var(mags))
        score = _safe_score_from_positive(instability, scale=0.15)
        return CriticResult(self.name, {FailureMode.PHYSICS: score, FailureMode.SMOOTHNESS: score},
                            notes=f"flow_mag_var={instability:.6f}")


@dataclass
class TemporalJitterCritic(Critic):
    """Estimates global frame-to-frame translation jitter using phase correlation."""
    name: str = "temporal_jitter"
    max_frames: int = 64

    def evaluate(self, artifact: WMArtifact, goal: GoalSpec) -> CriticResult:
        frames = list(_iter_frames(artifact.path, max_frames=self.max_frames))
        if len(frames) < 2:
            return CriticResult(self.name, {FailureMode.SMOOTHNESS: 0.0}, notes="not enough frames")

        shifts = []
        prev = frames[0].astype(np.float32)
        for cur in frames[1:]:
            cur_f = cur.astype(np.float32)
            shift, _response = cv2.phaseCorrelate(prev, cur_f)
            dx, dy = float(shift[0]), float(shift[1])
            shifts.append((dx, dy))
            prev = cur_f

        shifts_np = np.asarray(shifts, dtype=np.float32)
        # jitter = std of translation magnitude
        mag = np.sqrt(np.sum(shifts_np**2, axis=1))
        jitter = float(np.std(mag))
        score = _safe_score_from_positive(jitter, scale=0.8)
        return CriticResult(self.name, {FailureMode.SMOOTHNESS: score, FailureMode.QUALITY: score},
                            notes=f"global_jitter_std={jitter:.4f}")
