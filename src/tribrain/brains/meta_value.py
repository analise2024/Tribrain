from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MetaValueConfig:
    """Predicts expected next-iteration improvement and decides if continuing is worth it.

    This is the core of "metacognition for cost/energy": it estimates marginal value vs cost.
    """
    dim: int
    noise_var: float = 0.05
    prior_prec: float = 1.0

    # Decision thresholds
    min_expected_delta: float = 0.005          # stop if expected improvement is below this
    cost_penalty: float = 0.15                 # scales cost normalization
    min_value: float = 0.0                     # stop if (expected_delta - cost_penalty*cost) < min_value


@dataclass
class MetaValueModel:
    """Online Bayesian linear model predicting future delta score.

    y ~= score_{t+1} - score_t
    """
    cfg: MetaValueConfig
    A: np.ndarray | None = None
    b: np.ndarray | None = None
    updates: int = 0

    def __post_init__(self) -> None:
        if self.A is None:
            self.A = self.cfg.prior_prec * np.eye(self.cfg.dim, dtype=np.float32)
        if self.b is None:
            self.b = np.zeros((self.cfg.dim,), dtype=np.float32)

    def update(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.cfg.dim:
            raise ValueError(f"MetaValueModel: expected dim={self.cfg.dim}, got {x.shape[0]}")
        inv_sigma2 = 1.0 / max(1e-8, float(self.cfg.noise_var))
        assert self.A is not None and self.b is not None
        self.A = self.A + inv_sigma2 * np.outer(x, x)
        self.b = self.b + inv_sigma2 * x * float(y)
        self.updates += 1

    def predict(self, x: np.ndarray) -> tuple[float, float]:
        """Returns (mean, variance) for predicted delta."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        assert self.A is not None and self.b is not None
        mu_w = np.linalg.solve(self.A, self.b)
        mu = float(mu_w @ x)
        cov = np.linalg.inv(self.A)
        var = float(x.T @ cov @ x) + float(self.cfg.noise_var)
        return mu, var

    def save(self, path: Path) -> None:
        path = Path(path)
        obj = {
            "cfg": {
                "dim": int(self.cfg.dim),
                "noise_var": float(self.cfg.noise_var),
                "prior_prec": float(self.cfg.prior_prec),
                "min_expected_delta": float(self.cfg.min_expected_delta),
                "cost_penalty": float(self.cfg.cost_penalty),
                "min_value": float(self.cfg.min_value),
            },
            "A": self.A.tolist() if self.A is not None else None,
            "b": self.b.tolist() if self.b is not None else None,
            "updates": int(self.updates),
        }
        path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> MetaValueModel:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = MetaValueConfig(**obj["cfg"])
        m = MetaValueModel(cfg=cfg, A=np.array(obj["A"], dtype=np.float32), b=np.array(obj["b"], dtype=np.float32))
        m.updates = int(obj.get("updates", 0))
        return m
