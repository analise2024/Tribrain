from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from ..ontology.schema import TaskOntology
from ..types import FailureMode


def _stable_hash(text: str) -> int:
    # Deterministic across processes / platforms
    h = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def _hash_bow(text: str, dim: int = 32) -> np.ndarray:
    v = np.zeros((dim,), dtype=np.float32)
    for tok in text.lower().split():
        idx = _stable_hash(tok) % dim
        v[idx] += 1.0
    # tf-log scaling
    v = np.log1p(v)
    # L2 normalize
    n = float(np.linalg.norm(v))
    if n > 1e-8:
        v /= n
    return v


@dataclass(frozen=True)
class ContextFeatures:
    """Compact context used by the controller.

    We intentionally keep this *cheap* and low-dimensional:
    - ontology counts (entities/actions/constraints)
    - prompt length proxy
    - hashed BoW for instruction & compressed ontology
    - last known critic scores (optional)
    - meta flags (optional)
    """

    vector: np.ndarray

    @staticmethod
    def from_state(
        instruction: str,
        ontology: TaskOntology | None,
        compressed_ontology: str = "",
        last_scores: Mapping[FailureMode, float] | None = None,
        drift_flag: bool = False,
        plateau_count: int = 0,
        bow_dim: int = 32,
    ) -> ContextFeatures:
        last_scores = dict(last_scores or {})
        # Basic numeric features
        ent_n = float(len(ontology.entities)) if ontology else 0.0
        act_n = float(len(ontology.actions)) if ontology else 0.0
        con_n = float(len(ontology.constraints)) if ontology else 0.0
        instr_len = float(len(instruction.split()))
        ont_len = float(len(compressed_ontology.split())) if compressed_ontology else 0.0

        # bounded transforms to stabilize updates
        def _log1(x: float) -> float:
            return float(math.log1p(max(0.0, x)))

        nums = np.array(
            [
                _log1(ent_n),
                _log1(act_n),
                _log1(con_n),
                _log1(instr_len),
                _log1(ont_len),
                1.0 if drift_flag else 0.0,
                min(10.0, float(plateau_count)) / 10.0,
            ],
            dtype=np.float32,
        )

        # Failure-mode scores in a fixed order (0 if missing)
        mode_order = [FailureMode.PHYSICS, FailureMode.TASK, FailureMode.SMOOTHNESS, FailureMode.QUALITY]
        scores = np.array([float(last_scores.get(m, 0.0)) for m in mode_order], dtype=np.float32)

        # Hashed text embeddings
        bow_instr = _hash_bow(instruction, dim=bow_dim)
        bow_ont = _hash_bow(compressed_ontology, dim=bow_dim) if compressed_ontology else np.zeros((bow_dim,), dtype=np.float32)

        vec = np.concatenate([nums, scores, bow_instr, bow_ont], axis=0).astype(np.float32)
        return ContextFeatures(vector=vec)

    @property
    def dim(self) -> int:
        return int(self.vector.shape[0])
