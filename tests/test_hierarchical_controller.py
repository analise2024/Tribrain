import numpy as np

from tribrain.rl.hierarchical_contextual import HierarchicalContextualTS


def test_hierarchical_contextual_ts_choose_update_save_load(tmp_path):
    dim = 12
    ctrl = HierarchicalContextualTS(dim=dim, arm_names=["a", "b", "c"], seed=123)
    x = np.ones((dim,), dtype=np.float32) * 0.1
    arm = ctrl.choose(x)
    assert arm in ["a", "b", "c"]

    # Update should not crash
    ctrl.update(arm, x, reward=0.5)

    p = tmp_path / "ctrl.json"
    ctrl.save(p)
    ctrl2 = HierarchicalContextualTS.load(p)
    arm2 = ctrl2.choose(x)
    assert arm2 in ["a", "b", "c"]
