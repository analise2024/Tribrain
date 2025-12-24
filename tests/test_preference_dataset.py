from __future__ import annotations

import json
from pathlib import Path

from tribrain.continual.preference_dataset import build_preference_pairs


def test_build_preference_pairs_best_vs_rest(tmp_path: Path):
    trace = tmp_path / "trace.jsonl"
    # Three iterations, different scores
    rows = [
        {"prompt": "p0", "artifact_path": "v0.mp4", "overall_score": 0.6, "critic_scores": {}, "meta": {"goal_instruction": "do x"}, "cost": {}},
        {"prompt": "p1", "artifact_path": "v1.mp4", "overall_score": 0.8, "critic_scores": {}, "meta": {"goal_instruction": "do x"}, "cost": {}},
        {"prompt": "p2", "artifact_path": "v2.mp4", "overall_score": 0.7, "critic_scores": {}, "meta": {"goal_instruction": "do x"}, "cost": {}},
    ]
    trace.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    pairs = build_preference_pairs(trace, min_gap=0.05, top_k=1, strategy="best_vs_rest")
    assert len(pairs) >= 1
    # best should be chosen
    assert pairs[0].chosen_artifact == "v1.mp4"
    assert pairs[0].chosen_prompt == "p1"
