from __future__ import annotations

import os
import random
from contextlib import suppress

import numpy as np


def set_global_seed(seed: int) -> None:
    """Best-effort seeding for deterministic TriBrain runs.

    Note: determinism of the wrapped world model is ultimately controlled by that model.
    We set seeds for TriBrain-side stochastic components and common ML libs if available.
    """
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    with suppress(Exception):
        import torch  # type: ignore[import-not-found]

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with suppress(Exception):
            torch.use_deterministic_algorithms(True)
        with suppress(Exception):
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
