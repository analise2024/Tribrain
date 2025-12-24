from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from tribrain.memory.episodic import Episode, EpisodeStore


def _table_cols(db: Path, table: str) -> list[str]:
    with sqlite3.connect(str(db)) as con:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(r[1]) for r in rows]


def _get_schema_version(db: Path) -> int:
    with sqlite3.connect(str(db)) as con:
        try:
            row = con.execute("SELECT v FROM __schema_version LIMIT 1").fetchone()
        except Exception:
            return 0
    return int(row[0]) if row and row[0] is not None else 0


def _create_v1_db(db: Path) -> None:
    """Create a minimal legacy DB (v1) missing most columns."""
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db)) as con:
        con.execute(
            """
            CREATE TABLE episodes (
                ts REAL NOT NULL,
                run_id TEXT NOT NULL,
                choice_iter_idx INTEGER NOT NULL,
                arm TEXT NOT NULL,
                overall_score_next REAL NOT NULL,
                delta_score REAL NOT NULL
            )
            """
        )
        con.execute(
            "INSERT INTO episodes (ts,run_id,choice_iter_idx,arm,overall_score_next,delta_score) VALUES (?,?,?,?,?,?)",
            (1.0, "runA", 0, "task", 0.4, 0.1),
        )


def _create_v2_db(db: Path) -> None:
    """Legacy DB (v2): adds vectors but no meta/reliability columns."""
    db.parent.mkdir(parents=True, exist_ok=True)
    ctx = np.zeros((4,), dtype=np.float32).tobytes()
    mx = np.ones((3,), dtype=np.float32).tobytes()
    with sqlite3.connect(str(db)) as con:
        con.execute("CREATE TABLE IF NOT EXISTS __schema_version (v INTEGER NOT NULL)")
        con.execute("DELETE FROM __schema_version")
        con.execute("INSERT INTO __schema_version (v) VALUES (1)")
        con.execute(
            """
            CREATE TABLE episodes (
                ts REAL NOT NULL,
                run_id TEXT NOT NULL,
                choice_iter_idx INTEGER NOT NULL,
                ctx_vec BLOB NOT NULL,
                ctx_dim INTEGER NOT NULL,
                meta_x BLOB NOT NULL,
                meta_dim INTEGER NOT NULL,
                arm TEXT NOT NULL,
                overall_score_next REAL NOT NULL,
                delta_score REAL NOT NULL
            )
            """
        )
        con.execute(
            """
            INSERT INTO episodes (ts,run_id,choice_iter_idx,ctx_vec,ctx_dim,meta_x,meta_dim,arm,overall_score_next,delta_score)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            (2.0, "runB", 1, sqlite3.Binary(ctx), 4, sqlite3.Binary(mx), 3, "physics", 0.6, -0.05),
        )


def test_migration_v1_to_latest_is_idempotent(tmp_path: Path) -> None:
    db = tmp_path / "episodes.sqlite"
    _create_v1_db(db)

    # First open triggers migration.
    store = EpisodeStore(db, backup=False)
    assert store.count() == 1
    cols = _table_cols(db, "episodes")
    # Key new columns exist.
    for c in [
        "ctx_vec",
        "ctx_dim",
        "meta_x",
        "meta_dim",
        "fused_scores_next",
        "obs_reward",
        "smoothed_reward",
        "llm_calls_total_next",
        "llm_tokens_total_next",
    ]:
        assert c in cols
    assert _get_schema_version(db) >= 3

    # Data preserved for common columns.
    with sqlite3.connect(str(db)) as con:
        row = con.execute(
            "SELECT run_id,choice_iter_idx,arm,overall_score_next,delta_score FROM episodes ORDER BY ts"
        ).fetchone()
    assert row == ("runA", 0, "task", 0.4, 0.1)

    # Re-run migration: should be a no-op (no errors, no data loss).
    store2 = EpisodeStore(db, backup=False)
    assert store2.count() == 1
    assert _table_cols(db, "episodes") == cols


def test_migration_v2_to_latest_preserves_vectors(tmp_path: Path) -> None:
    db = tmp_path / "episodes.sqlite"
    _create_v2_db(db)

    store = EpisodeStore(db, backup=False)
    assert store.count() == 1
    cols = _table_cols(db, "episodes")
    assert "fused_scores_next" in cols
    assert _get_schema_version(db) >= 3

    # Existing vectors should survive rebuild.
    with sqlite3.connect(str(db)) as con:
        row = con.execute(
            "SELECT ctx_dim,meta_dim,arm,overall_score_next,delta_score FROM episodes ORDER BY ts"
        ).fetchone()
    assert row == (4, 3, "physics", 0.6, -0.05)

    # Insert using current writer to ensure new schema is writable.
    ep = Episode(
        ts=3.0,
        run_id="runC",
        choice_iter_idx=2,
        ctx_vec=np.zeros((4,), dtype=np.float32),
        meta_x=np.zeros((5,), dtype=np.float32),
        arm="quality",
        fused_scores_next={"task": 0.1},
        overall_score_next=0.2,
        delta_score=0.0,
        obs_reward=0.0,
        smoothed_reward=0.0,
        drift_flag_next=False,
        plateau_steps_next=0,
        iter_time_s_next=0.1,
        llm_calls_total_next=0,
        llm_tokens_total_next=0,
    )
    store.add(ep)
    assert store.count() == 2
