from tribrain.brains.belief import BeliefState
from tribrain.types import CriticResult, FailureMode


def test_belief_updates_increase_failure_mode_weight():
    belief = BeliefState()
    # physics score low -> should increase belief in physics failure
    res = [
        CriticResult("c1", {FailureMode.PHYSICS: 0.2, FailureMode.QUALITY: 0.9}),
        CriticResult("c2", {FailureMode.PHYSICS: 0.3, FailureMode.SMOOTHNESS: 0.8}),
    ]
    fused = belief.fuse(res)
    belief.update(res, fused)
    probs = belief.modes.probs()
    assert probs[FailureMode.PHYSICS] > probs[FailureMode.QUALITY]
