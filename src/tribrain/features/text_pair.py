from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-]+", re.UNICODE)


def _stable_hash(text: str) -> int:
    h = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def hashed_bow(text: str, dim: int) -> np.ndarray:
    v = np.zeros((dim,), dtype=np.float32)
    for tok in _TOKEN_RE.findall(text.lower()):
        idx = _stable_hash(tok) % dim
        v[idx] += 1.0
    # L2 normalize
    n = float(np.linalg.norm(v) + 1e-8)
    return v / n


def pair_feature(instruction: str, candidate: str, dim: int) -> np.ndarray:
    """Feature for reward models: concatenated hashed BoW of (instruction, candidate)."""
    a = hashed_bow(instruction, dim)
    b = hashed_bow(candidate, dim)
    return np.concatenate([a, b], axis=0)
