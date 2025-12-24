from __future__ import annotations

import numpy as np

from tribrain.reward.preference import BradleyTerryRewardModel, PreferenceExample


def test_bradley_terry_learns_simple_signal() -> None:
    # Synthetic: chosen has larger first feature
    rng = np.random.default_rng(0)
    dim = 6
    model = BradleyTerryRewardModel(dim=dim, lr=0.2, l2=1e-3)

    examples = []
    for _ in range(200):
        base = rng.normal(size=(dim,)).astype(np.float32)
        chosen = base.copy()
        rejected = base.copy()
        chosen[0] += 1.0
        rejected[0] -= 1.0
        examples.append(PreferenceExample(chosen=chosen, rejected=rejected))
    model.fit(examples, epochs=2)

    # probability should be high on a representative pair
    chosen = np.zeros((dim,), dtype=np.float32)
    chosen[0] = 1.0
    rejected = np.zeros((dim,), dtype=np.float32)
    rejected[0] = -1.0
    p = model.prob_prefer(chosen, rejected)
    assert p > 0.8
