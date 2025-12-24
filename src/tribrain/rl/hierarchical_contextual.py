from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class _BayesLin:
    dim: int
    noise_var: float = 0.1
    prior_prec: float = 1.0
    A: np.ndarray | None = None
    b: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.A is None:
            self.A = (self.prior_prec * np.eye(self.dim, dtype=np.float32))
        if self.b is None:
            self.b = np.zeros((self.dim,), dtype=np.float32)

    def mean(self) -> np.ndarray:
        A = self.A
        b = self.b
        assert A is not None and b is not None
        return np.asarray(np.linalg.solve(A, b), dtype=np.float32)

    def update(self, x: np.ndarray, y: float) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"_BayesLin: expected dim={self.dim}, got {x.shape[0]}")
        y = float(y)
        inv_sigma2 = 1.0 / max(1e-8, float(self.noise_var))
        A = self.A
        b = self.b
        assert A is not None and b is not None
        self.A = A + inv_sigma2 * np.outer(x, x)
        self.b = b + inv_sigma2 * x * y

    def sample_weights(self, rng: np.random.Generator) -> np.ndarray:
        """Thompson sample weights from N(mu, noise_var * A^{-1})."""
        mu = self.mean()
        # sample via precision cholesky
        A = self.A
        assert A is not None
        # Cholesky of A (precision)
        try:
            L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError:
            # Numerical fallback
            L = np.linalg.cholesky(A + 1e-6 * np.eye(self.dim, dtype=np.float32))
        z = rng.normal(size=(self.dim,)).astype(np.float32)
        # Solve L^T v = z => v = (L^T)^{-1} z; covariance = A^{-1}
        v = np.asarray(np.linalg.solve(L.T, z), dtype=np.float32)
        w = mu + np.sqrt(max(1e-8, float(self.noise_var))) * v
        return np.asarray(w, dtype=np.float32)

    def to_dict(self) -> dict:
        return {
            "dim": self.dim,
            "noise_var": self.noise_var,
            "prior_prec": self.prior_prec,
            "A": self.A.tolist() if self.A is not None else None,
            "b": self.b.tolist() if self.b is not None else None,
        }

    @staticmethod
    def from_dict(d: dict) -> _BayesLin:
        m = _BayesLin(dim=int(d["dim"]), noise_var=float(d.get("noise_var", 0.1)), prior_prec=float(d.get("prior_prec", 1.0)))
        if d.get("A") is not None:
            m.A = np.asarray(d["A"], dtype=np.float32)
        if d.get("b") is not None:
            m.b = np.asarray(d["b"], dtype=np.float32)
        return m


@dataclass
class HierarchicalContextualTS:
    """Hierarchical contextual Thompson sampling controller.

    Model:
      reward ≈ x^T w_global + x^T w_arm + ε

    We maintain:
      - a shared Bayesian linear model (global)
      - one Bayesian linear *residual* per arm

    This gives:
      - transfer across arms (global learns structure)
      - specialization (arm residuals learn deviations)

    Cheap, online, and works on CPU.
    """

    dim: int
    arm_names: list[str]
    noise_var: float = 0.1
    prior_prec_global: float = 1.0
    prior_prec_arm: float = 2.0
    seed: int = 0

    global_model: _BayesLin | None = None
    arm_models: dict[str, _BayesLin] | None = None

    def __post_init__(self) -> None:
        if self.global_model is None:
            self.global_model = _BayesLin(self.dim, noise_var=self.noise_var, prior_prec=self.prior_prec_global)
        if self.arm_models is None:
            self.arm_models = {n: _BayesLin(self.dim, noise_var=self.noise_var, prior_prec=self.prior_prec_arm) for n in self.arm_names}
        self.rng = np.random.default_rng(self.seed)

    def set_seed(self, seed: int) -> None:
        """Reset RNG seed for deterministic action selection.

        This does not change the learned posterior parameters (A,b), only the sampling RNG.
        """
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def choose(self, x: np.ndarray) -> str:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"Controller: expected dim={self.dim}, got {x.shape[0]}")
        assert self.global_model is not None
        assert self.arm_models is not None
        wg = self.global_model.sample_weights(self.rng)
        best_arm = self.arm_names[0]
        best_val = -1e9
        for arm in self.arm_names:
            wa = self.arm_models[arm].sample_weights(self.rng)
            val = float(np.dot(wg + wa, x))
            if val > best_val:
                best_val = val
                best_arm = arm
        return best_arm

    def update(self, arm: str, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        assert self.global_model is not None
        assert self.arm_models is not None
        if arm not in self.arm_models:
            return
        # Approximate coordinate update:
        # 1) update global using residual of current arm mean
        mu_a = self.arm_models[arm].mean()
        r_g = float(reward - np.dot(mu_a, x))
        self.global_model.update(x, r_g)
        # 2) update arm residual using residual of global mean
        mu_g = self.global_model.mean()
        r_a = float(reward - np.dot(mu_g, x))
        self.arm_models[arm].update(x, r_a)

    def save(self, path: Path) -> None:
        assert self.global_model is not None
        assert self.arm_models is not None
        data = {
            "dim": self.dim,
            "arm_names": self.arm_names,
            "noise_var": self.noise_var,
            "prior_prec_global": self.prior_prec_global,
            "prior_prec_arm": self.prior_prec_arm,
            "seed": self.seed,
            "global_model": self.global_model.to_dict(),
            "arm_models": {k: v.to_dict() for k, v in self.arm_models.items()},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: Path) -> HierarchicalContextualTS:
        d = json.loads(path.read_text(encoding="utf-8"))
        ctrl = HierarchicalContextualTS(
            dim=int(d["dim"]),
            arm_names=list(d["arm_names"]),
            noise_var=float(d.get("noise_var", 0.1)),
            prior_prec_global=float(d.get("prior_prec_global", 1.0)),
            prior_prec_arm=float(d.get("prior_prec_arm", 2.0)),
            seed=int(d.get("seed", 0)),
        )
        ctrl.global_model = _BayesLin.from_dict(d["global_model"])
        ctrl.arm_models = {k: _BayesLin.from_dict(v) for k, v in d["arm_models"].items()}
        ctrl.rng = np.random.default_rng(ctrl.seed)
        return ctrl
