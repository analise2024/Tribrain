from __future__ import annotations

import numpy as np

from tribrain.brains.meta_value import MetaValueConfig, MetaValueModel


def test_meta_value_model_predicts_sign(tmp_path):
    cfg = MetaValueConfig(dim=8, noise_var=0.01, prior_prec=1.0)
    m = MetaValueModel(cfg=cfg)
    rng = np.random.default_rng(0)

    # synthetic: y ~= x0 - x1
    for _ in range(200):
        x = rng.normal(size=(8,)).astype(np.float32)
        y = float(x[0] - x[1])
        m.update(x, y)

    x_test = np.zeros((8,), dtype=np.float32)
    x_test[0] = 1.0
    mu, var = m.predict(x_test)
    assert mu > 0.2
    assert var > 0.0
