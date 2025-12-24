from tribrain.brains.meta import MetaConfig, MetaController, MetaState
from tribrain.types import FailureMode


def test_meta_stops_on_target():
    mc = MetaController(MetaConfig(target_score=0.5, max_iters=10))
    st = MetaState()
    st.iter_idx = 0
    score = mc.update(st, {FailureMode.PHYSICS: 1.0, FailureMode.TASK: 1.0, FailureMode.SMOOTHNESS: 0.0, FailureMode.QUALITY: 0.0})
    assert score == 0.5
    assert mc.should_stop(st) is True
    assert st.stop_reason == "target_reached"


def test_meta_plateau():
    mc = MetaController(MetaConfig(target_score=0.99, max_iters=10, patience=2, min_delta=0.01))
    st = MetaState()
    st.iter_idx = 0
    mc.update(st, {FailureMode.PHYSICS: 0.5, FailureMode.TASK: 0.5, FailureMode.SMOOTHNESS: 0.5, FailureMode.QUALITY: 0.5})
    assert mc.should_stop(st) is False
    st.iter_idx = 1
    mc.update(st, {FailureMode.PHYSICS: 0.505, FailureMode.TASK: 0.5, FailureMode.SMOOTHNESS: 0.5, FailureMode.QUALITY: 0.5})
    assert mc.should_stop(st) is False
    st.iter_idx = 2
    mc.update(st, {FailureMode.PHYSICS: 0.506, FailureMode.TASK: 0.5, FailureMode.SMOOTHNESS: 0.5, FailureMode.QUALITY: 0.5})
    assert mc.should_stop(st) is True
    assert st.stop_reason == "plateau"
