from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from tribrain.memory.episodic import Episode, EpisodeStore


def _run_cli(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "tribrain.cli", *args],
        cwd=str(cwd) if cwd is not None else None,
        check=False,
        capture_output=True,
        text=True,
    )


def test_cli_help_works() -> None:
    cp = _run_cli(["--help"])
    assert cp.returncode == 0
    assert "TriBrain" in cp.stdout


def test_cli_export_pairs_and_replay_update(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Minimal trace with a score improvement so preference pairs can be built.
    trace = run_dir / "trace.jsonl"
    rows = [
        {
            "iter_idx": 0,
            "prompt": "p0",
            "artifact_path": "a0.mp4",
            "critic_results": [
                {"critic_name": "c", "scores": {"task": 0.6}, "notes": "", "extra": {}}
            ],
            "fused_scores": {"task": 0.6},
            "belief_probs": {"task": 1.0},
            "overall_score": 0.4,
            "meta": {"goal_instruction": "do X"},
        },
        {
            "iter_idx": 1,
            "prompt": "p1",
            "artifact_path": "a1.mp4",
            "critic_results": [
                {"critic_name": "c", "scores": {"task": 0.7}, "notes": "", "extra": {}}
            ],
            "fused_scores": {"task": 0.7},
            "belief_probs": {"task": 1.0},
            "overall_score": 0.5,
            "meta": {"goal_instruction": "do X"},
        },
    ]
    trace.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    # Minimal episodes DB for replay-update.
    store = EpisodeStore(run_dir / "episodes.sqlite", backup=False)
    store.add(
        Episode(
            ts=1.0,
            run_id="r",
            choice_iter_idx=0,
            ctx_vec=np.zeros((4,), dtype=np.float32),
            meta_x=np.zeros((6,), dtype=np.float32),
            arm="task",
            fused_scores_next={"task": 0.6},
            overall_score_next=0.5,
            delta_score=0.1,
            obs_reward=0.1,
            smoothed_reward=0.1,
            drift_flag_next=False,
            plateau_steps_next=0,
            iter_time_s_next=0.01,
            llm_calls_total_next=0,
            llm_tokens_total_next=0,
        )
    )

    out_pairs = tmp_path / "pairs.jsonl"
    cp = _run_cli([
        "export-preference-pairs",
        "--run-dir",
        str(run_dir),
        "--out-path",
        str(out_pairs),
        "--min-gap",
        "0.01",
        "--top-k",
        "1",
    ])
    assert cp.returncode == 0, cp.stderr
    assert out_pairs.exists()
    assert out_pairs.read_text(encoding="utf-8").strip()

    cp2 = _run_cli([
        "replay-update",
        "--run-dir",
        str(run_dir),
        "--episodes-db",
        "episodes.sqlite",
        "--k",
        "1",
        "--strategy",
        "uniform",
    ])
    assert cp2.returncode == 0, cp2.stderr
    assert "episodes_updated" in cp2.stdout