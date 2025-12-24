import numpy as np

from tribrain.reward.model import BayesianRewardModel


def test_reward_model_predict_update(tmp_path):
    rm = BayesianRewardModel(dim=8, noise_var=0.1, prior_prec=1.0)
    x = np.ones((8,), dtype=np.float32)
    mu0, var0 = rm.predict(x)
    rm.update(x, 0.3)
    mu1, var1 = rm.predict(x)
    assert var1 >= 0.0
    # After one update, mean should move in the direction of observation
    assert mu1 != mu0

    p = tmp_path / "rm.json"
    rm.save(p)
    rm2 = BayesianRewardModel.load(p)
    mu2, var2 = rm2.predict(x)
    assert abs(mu2 - mu1) < 1e-5
