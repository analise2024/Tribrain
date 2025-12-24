from __future__ import annotations

import numpy as np

from tribrain.memory.episodic import Episode, EpisodeStore


def test_episode_store_roundtrip(tmp_path):
    db = tmp_path / "episodes.sqlite"
    store = EpisodeStore(db)

    ctx = np.random.default_rng(0).normal(size=(16,)).astype(np.float32)
    mx = np.random.default_rng(1).normal(size=(24,)).astype(np.float32)

    ep = Episode(
        ts=1.23,
        run_id="run1",
        choice_iter_idx=0,
        ctx_vec=ctx,
        meta_x=mx,
        arm="task",
        fused_scores_next={"TASK": 0.8},
        overall_score_next=0.8,
        delta_score=0.02,
        obs_reward=0.1,
        smoothed_reward=0.09,
        drift_flag_next=False,
        plateau_steps_next=0,
        iter_time_s_next=2.0,
        llm_calls_total_next=1,
        llm_tokens_total_next=100,
    )
    store.add(ep)
    assert store.count() == 1

    got = store.sample(1, seed=0)[0]
    assert got.run_id == "run1"
    assert got.arm == "task"
    assert got.choice_iter_idx == 0
    assert got.ctx_vec.shape[0] == 16
    assert got.meta_x.shape[0] == 24
