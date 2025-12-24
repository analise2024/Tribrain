from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..types import FailureMode


def _as_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def reward_feature_vector(
    fused_scores: Mapping[FailureMode, float],
    prev_score: float,
    prompt_len: int,
    drift_flag: bool,
) -> np.ndarray:
    """Feature vector for reward modeling.

    Designed to be tiny and stable:
    - fused critic scores by mode
    - previous total score
    - prompt length (log-scaled)
    - drift flag
    """
    order = [FailureMode.PHYSICS, FailureMode.TASK, FailureMode.SMOOTHNESS, FailureMode.QUALITY]
    scores = [_as_float(fused_scores.get(m, 0.0)) for m in order]
    vec = np.array(
        [
            *scores,
            _as_float(prev_score),
            np.log1p(max(0, int(prompt_len))) / 10.0,
            1.0 if drift_flag else 0.0,
            1.0,  # bias
        ],
        dtype=np.float32,
    )
    return vec


@dataclass
class BayesianRewardModel:
    """Online Bayesian linear reward model.

    This provides *signal smoothing*: we learn how critic scores map to true improvements.
    It is cheap and works on CPU.
    """

    dim: int
    noise_var: float = 0.05  # assumed observation noise variance
    prior_prec: float = 1.0  # ridge strength

    # Posterior in information form: A (precision), b
    A: np.ndarray | None = None
    b: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.A is None:
            self.A = (self.prior_prec * np.eye(self.dim, dtype=np.float32))
        if self.b is None:
            self.b = np.zeros((self.dim,), dtype=np.float32)

    def update(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"RewardModel: expected dim={self.dim}, got {x.shape[0]}")
        y = float(y)
        inv_sigma2 = 1.0 / max(1e-8, float(self.noise_var))
        A = self.A
        b = self.b
        assert A is not None and b is not None
        self.A = A + inv_sigma2 * np.outer(x, x)
        self.b = b + inv_sigma2 * x * y

    def mean_weights(self) -> np.ndarray:
        A = self.A
        b = self.b
        assert A is not None and b is not None
        return np.asarray(np.linalg.solve(A, b), dtype=np.float32)

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        """Returns (mean, variance) under the posterior."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        w = self.mean_weights()
        mu = float(np.dot(w, x))
        A = self.A
        assert A is not None
        # predictive variance: x^T A^{-1} x * noise_var
        cov_x = np.linalg.solve(A, x)
        var = float(np.dot(x, cov_x) * self.noise_var)
        return mu, max(0.0, var)

    def save(self, path: Path) -> None:
        data = {
            "dim": self.dim,
            "noise_var": self.noise_var,
            "prior_prec": self.prior_prec,
            "A": self.A.tolist() if self.A is not None else None,
            "b": self.b.tolist() if self.b is not None else None,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> BayesianRewardModel:
        data = json.loads(path.read_text(encoding="utf-8"))
        rm = BayesianRewardModel(
            dim=int(data["dim"]),
            noise_var=float(data.get("noise_var", 0.05)),
            prior_prec=float(data.get("prior_prec", 1.0)),
        )
        A = data.get("A")
        b = data.get("b")
        if A is not None:
            rm.A = np.asarray(A, dtype=np.float32)
        if b is not None:
            rm.b = np.asarray(b, dtype=np.float32)
        return rm
