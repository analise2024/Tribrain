from __future__ import annotations

import numpy as np

from ..types import FailureMode


def meta_value_features(
    ctx_vec: np.ndarray,
    fused_scores: dict[FailureMode, float],
    overall_score: float,
    iter_idx: int,
    plateau_steps: int,
    drift_flag: bool,
    remaining_wall_frac: float,
    remaining_llm_calls_frac: float,
    remaining_llm_tokens_frac: float,
) -> np.ndarray:
    """Feature vector for meta value model.

    We deliberately keep it cheap and stable. The ctx_vec already contains ontology+bag features.
    """
    ctx_vec = np.asarray(ctx_vec, dtype=np.float32).reshape(-1)
    # Order scores
    order = [FailureMode.PHYSICS, FailureMode.TASK, FailureMode.SMOOTHNESS, FailureMode.QUALITY]
    scores = [float(fused_scores.get(m, 0.0)) for m in order]
    extra = np.array(
        [
            *scores,
            float(overall_score),
            float(iter_idx) / 10.0,
            float(plateau_steps) / 10.0,
            1.0 if drift_flag else 0.0,
            float(max(0.0, min(1.0, remaining_wall_frac))),
            float(max(0.0, min(1.0, remaining_llm_calls_frac))),
            float(max(0.0, min(1.0, remaining_llm_tokens_frac))),
            1.0,  # bias
        ],
        dtype=np.float32,
    )
    return np.concatenate([ctx_vec.astype(np.float32), extra], axis=0)
