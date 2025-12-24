from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..storage.sqlite_migrator import SQLiteMigrator
from ..storage.sqlite_schema import EPISODES_SCHEMA_V3


@dataclass(frozen=True)
class Episode:
    """One decision/outcome step for replay.

    We align episodes to the credit-assignment step:
    - ctx_vec / meta_x: the context at time of decision (t)
    - arm: chosen intervention/focus
    - delta_score: observed improvement at next step (t+1)
    - obs_reward: bounded transform used for stable RL updates
    - costs: time/tokens/calls (used for energy-aware learning)
    """

    ts: float
    run_id: str
    choice_iter_idx: int

    ctx_vec: np.ndarray
    meta_x: np.ndarray
    arm: str

    fused_scores_next: dict[str, float]
    overall_score_next: float
    delta_score: float
    obs_reward: float
    smoothed_reward: float

    drift_flag_next: bool
    plateau_steps_next: int

    iter_time_s_next: float
    llm_calls_total_next: int
    llm_tokens_total_next: int

    def to_row(self) -> tuple:
        return (
            float(self.ts),
            str(self.run_id),
            int(self.choice_iter_idx),
            sqlite3.Binary(self.ctx_vec.astype(np.float32).tobytes()),
            int(self.ctx_vec.size),
            sqlite3.Binary(self.meta_x.astype(np.float32).tobytes()),
            int(self.meta_x.size),
            str(self.arm),
            json.dumps(self.fused_scores_next, ensure_ascii=False),
            float(self.overall_score_next),
            float(self.delta_score),
            float(self.obs_reward),
            float(self.smoothed_reward),
            1 if self.drift_flag_next else 0,
            int(self.plateau_steps_next),
            float(self.iter_time_s_next),
            int(self.llm_calls_total_next),
            int(self.llm_tokens_total_next),
        )

    @staticmethod
    def from_row(row: Sequence) -> Episode:
        (
            ts,
            run_id,
            choice_iter_idx,
            ctx_blob,
            ctx_dim,
            meta_blob,
            meta_dim,
            arm,
            fused_json,
            overall_score_next,
            delta_score,
            obs_reward,
            smoothed_reward,
            drift_flag_next,
            plateau_steps_next,
            iter_time_s_next,
            llm_calls_total_next,
            llm_tokens_total_next,
        ) = row
        ctx = np.frombuffer(ctx_blob, dtype=np.float32, count=int(ctx_dim))
        mx = np.frombuffer(meta_blob, dtype=np.float32, count=int(meta_dim))
        fused = json.loads(fused_json)
        return Episode(
            ts=float(ts),
            run_id=str(run_id),
            choice_iter_idx=int(choice_iter_idx),
            ctx_vec=ctx,
            meta_x=mx,
            arm=str(arm),
            fused_scores_next={str(k): float(v) for k, v in fused.items()},
            overall_score_next=float(overall_score_next),
            delta_score=float(delta_score),
            obs_reward=float(obs_reward),
            smoothed_reward=float(smoothed_reward),
            drift_flag_next=bool(int(drift_flag_next)),
            plateau_steps_next=int(plateau_steps_next),
            iter_time_s_next=float(iter_time_s_next),
            llm_calls_total_next=int(llm_calls_total_next),
            llm_tokens_total_next=int(llm_tokens_total_next),
        )


class EpisodeStore:
    """SQLite-backed episodic memory.

    This is intentionally "boring tech": durable, portable, and good enough for large runs.
    """

    def __init__(self, path: Path, *, backup: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._backup = bool(backup)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Connect with guaranteed close.

        Note: sqlite3.Connection as a context manager *does not* close the connection;
        it only commits/rolls back. We close explicitly in finally.
        """
        con = sqlite3.connect(str(self.path), timeout=30.0)
        try:
            con.execute("PRAGMA journal_mode=WAL;")
            con.execute("PRAGMA synchronous=NORMAL;")
            yield con
            con.commit()
        except Exception:
            with suppress(Exception):
                con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        # Idempotent, atomic migration.
        # This avoids fragile repeated ALTER TABLE when upgrading old DBs.
        SQLiteMigrator(self.path, backup=self._backup).migrate(EPISODES_SCHEMA_V3)

    def add(self, ep: Episode) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                ep.to_row(),
            )

    def add_many(self, eps: Iterable[Episode]) -> None:
        rows = [ep.to_row() for ep in eps]
        if not rows:
            return
        with self._connect() as con:
            con.executemany("INSERT INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def count(self) -> int:
        with self._connect() as con:
            (n,) = con.execute("SELECT COUNT(*) FROM episodes").fetchone()
            return int(n)

    def sample(self, k: int, seed: int | None = None) -> list[Episode]:
        """Uniform random sample."""
        rng = np.random.default_rng(seed)
        n = self.count()
        if n == 0 or k <= 0:
            return []
        k = int(min(k, n))
        idx_arr = rng.choice(n, size=k, replace=False)
        idx = [int(x) for x in idx_arr.tolist()]
        with self._connect() as con:
            rows = con.execute(
                "SELECT ts,run_id,choice_iter_idx,ctx_vec,ctx_dim,meta_x,meta_dim,arm,fused_scores_next,overall_score_next,delta_score,obs_reward,smoothed_reward,drift_flag_next,plateau_steps_next,iter_time_s_next,llm_calls_total_next,llm_tokens_total_next FROM episodes ORDER BY ts"
            ).fetchall()
        return [Episode.from_row(rows[i]) for i in idx]

    def nearest(self, ctx_vec: np.ndarray, k: int = 32) -> list[Episode]:
        """Brute-force cosine nearest neighbors."""
        ctx_vec = np.asarray(ctx_vec, dtype=np.float32).reshape(-1)
        with self._connect() as con:
            rows = con.execute(
                "SELECT ts,run_id,choice_iter_idx,ctx_vec,ctx_dim,meta_x,meta_dim,arm,fused_scores_next,overall_score_next,delta_score,obs_reward,smoothed_reward,drift_flag_next,plateau_steps_next,iter_time_s_next,llm_calls_total_next,llm_tokens_total_next FROM episodes"
            ).fetchall()
        eps = [Episode.from_row(r) for r in rows]
        if not eps:
            return []
        mats = np.stack([e.ctx_vec for e in eps], axis=0)
        denom = (np.linalg.norm(mats, axis=1) * (np.linalg.norm(ctx_vec) + 1e-8) + 1e-8)
        sims = (mats @ ctx_vec) / denom
        order = np.argsort(-sims)[: int(max(1, k))]
        return [eps[int(i)] for i in order.tolist()]


def make_run_id(*, seed: int | None = None, instruction: str | None = None) -> str:
    """Generate a run id.

    Requirements:
      - readable
      - includes the seed when provided (for reproducibility/debugging)
      - stable, short instruction hash (avoid leaking full prompt in filenames)
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    s = int(seed) if seed is not None else -1
    h = ""
    if instruction:
        import hashlib

        h = hashlib.sha1(instruction.encode("utf-8", errors="ignore")).hexdigest()[:8]
    if h:
        return f"{ts}_seed{s}_{h}"
    return f"{ts}_seed{s}"
