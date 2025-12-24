from __future__ import annotations

import json

from tribrain.dashboard.report import build_dashboard


def test_dashboard_build(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    trace = run_dir / "trace.jsonl"

    rows = [
        {"iter_idx": 0, "prompt": "p0", "artifact_path": "a0", "critic_results": [], "fused_scores": {}, "belief_probs": {}, "overall_score": 0.2, "meta": {"budget": {"llm_calls": 0, "llm_tokens_used": 0, "total_iter_time_s": 0.1}}},
        {"iter_idx": 1, "prompt": "p1", "artifact_path": "a1", "critic_results": [], "fused_scores": {}, "belief_probs": {}, "overall_score": 0.6, "meta": {"budget": {"llm_calls": 1, "llm_tokens_used": 10, "total_iter_time_s": 0.2}}},
    ]
    with trace.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    md = build_dashboard(run_dir=run_dir)
    assert md.exists()
    text = md.read_text(encoding="utf-8")
    assert "Best score" in text
