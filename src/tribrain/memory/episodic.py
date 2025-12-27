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
            1 if bool(self.drift_flag_next) else 0,
            int(self.plateau_steps_next),
            float(self.iter_time_s_next),
            int(self.llm_calls_total_next),
            int(self.llm_tokens_total_next),
        )

    @staticmethod
    def from_row(row: Sequence) -> "Episode":
        (
            ts,
            run_id,
            choice_iter_idx,
            ctx_blob,
            ctx_dim,
            meta_blob,
            meta_dim,
            arm,
            fused_scores_next,
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

        ctx = np.frombuffer(ctx_blob, dtype=np.float32, count=int(ctx_dim)).copy()
        meta = np.frombuffer(meta_blob, dtype=np.float32, count=int(meta_dim)).copy()

        return Episode(
            ts=float(ts),
            run_id=str(run_id),
            choice_iter_idx=int(choice_iter_idx),
            ctx_vec=ctx,
            meta_x=meta,
            arm=str(arm),
            fused_scores_next=json.loads(str(fused_scores_next) or "{}"),
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
    def __init__(self, path: Path, *, backup: bool = False):
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
            con.execute("PRAGMA foreign_keys=ON;")
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
        self._ensure_replay_table()

    def add(self, ep: Episode) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO episodes(
                  ts,run_id,choice_iter_idx,
                  ctx_vec,ctx_dim,
                  meta_x,meta_dim,
                  arm,
                  fused_scores_next,overall_score_next,
                  delta_score,obs_reward,smoothed_reward,
                  drift_flag_next,plateau_steps_next,
                  iter_time_s_next,llm_calls_total_next,llm_tokens_total_next
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                ep.to_row(),
            )

    def add_many(self, eps: Iterable[Episode]) -> int:
        rows = [e.to_row() for e in eps]
        if not rows:
            return 0
        with self._connect() as con:
            con.executemany(
                """
                INSERT INTO episodes(
                  ts,run_id,choice_iter_idx,
                  ctx_vec,ctx_dim,
                  meta_x,meta_dim,
                  arm,
                  fused_scores_next,overall_score_next,
                  delta_score,obs_reward,smoothed_reward,
                  drift_flag_next,plateau_steps_next,
                  iter_time_s_next,llm_calls_total_next,llm_tokens_total_next
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
        return len(rows)

    def count(self) -> int:
        with self._connect() as con:
            row = con.execute("SELECT COUNT(*) FROM episodes").fetchone()
        return int(row[0]) if row else 0

    def sample(self, k: int, *, seed: int = 0) -> list[Episode]:
        k = int(max(1, k))
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT ts,run_id,choice_iter_idx,ctx_vec,ctx_dim,meta_x,meta_dim,arm,
                       fused_scores_next,overall_score_next,delta_score,obs_reward,smoothed_reward,
                       drift_flag_next,plateau_steps_next,iter_time_s_next,llm_calls_total_next,llm_tokens_total_next
                FROM episodes
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (k,),
            ).fetchall()
        return [Episode.from_row(r) for r in rows]

    def nearest(self, ctx_vec: np.ndarray, k: int) -> list[Episode]:
        k = int(max(1, k))
        with self._connect() as con:
            rows = con.execute(
                """
                SELECT ts,run_id,choice_iter_idx,ctx_vec,ctx_dim,meta_x,meta_dim,arm,
                       fused_scores_next,overall_score_next,delta_score,obs_reward,smoothed_reward,
                       drift_flag_next,plateau_steps_next,iter_time_s_next,llm_calls_total_next,llm_tokens_total_next
                FROM episodes
                """
            ).fetchall()
        eps = [Episode.from_row(r) for r in rows]
        if not eps:
            return []
        mats = np.stack([e.ctx_vec for e in eps], axis=0)
        denom = (np.linalg.norm(mats, axis=1) * (np.linalg.norm(ctx_vec) + 1e-8) + 1e-8)
        sims = (mats @ ctx_vec) / denom
        order = np.argsort(-sims)[: int(max(1, k))]
        return [eps[int(i)] for i in order.tolist()]

    def _ensure_replay_table(self) -> None:
        """Create the replay-count side table (idempotent).

        We do NOT mutate the episodes schema for this: we track replay counts in a
        separate table keyed by the stable episode identity (ts, run_id, choice_iter_idx).
        """
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS episode_replay (
                    ts REAL NOT NULL,
                    run_id TEXT NOT NULL,
                    choice_iter_idx INTEGER NOT NULL,
                    replay_count INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (ts, run_id, choice_iter_idx)
                )
                """
            )

    def latest_run_id(self) -> str | None:
        with self._connect() as con:
            row = con.execute("SELECT run_id FROM episodes ORDER BY ts DESC LIMIT 1").fetchone()
        if not row:
            return None
        return str(row[0])

    def max_iter(self, *, run_id: str | None = None) -> int:
        with self._connect() as con:
            if run_id is None:
                row = con.execute("SELECT MAX(choice_iter_idx) FROM episodes").fetchone()
            else:
                row = con.execute("SELECT MAX(choice_iter_idx) FROM episodes WHERE run_id=?", (str(run_id),)).fetchone()
        v = row[0] if row else None
        return int(v) if v is not None else 0

    def fetch_recent(
        self,
        *,
        run_id: str | None,
        current_iter: int | None,
        max_age_iters: int | None,
        limit: int,
    ) -> list[Episode]:
        """Fetch recent episodes, optionally restricted to a run and iteration window."""
        limit_i = int(max(1, limit))
        sql = (
            "SELECT ts,run_id,choice_iter_idx,ctx_vec,ctx_dim,meta_x,meta_dim,arm,"
            "fused_scores_next,overall_score_next,delta_score,obs_reward,smoothed_reward,"
            "drift_flag_next,plateau_steps_next,iter_time_s_next,llm_calls_total_next,llm_tokens_total_next "
            "FROM episodes"
        )
        params: list[object] = []
        where: list[str] = []
        if run_id is not None:
            where.append("run_id=?")
            params.append(str(run_id))
        if current_iter is not None and max_age_iters is not None:
            min_iter = int(current_iter) - int(max_age_iters)
            where.append("choice_iter_idx>=?")
            params.append(int(min_iter))
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts DESC LIMIT ?"
        params.append(limit_i)
        with self._connect() as con:
            rows = con.execute(sql, tuple(params)).fetchall()
        return [Episode.from_row(r) for r in rows]

    def fetch_all(self, *, limit: int | None = None) -> list[Episode]:
        sql = (
            "SELECT ts,run_id,choice_iter_idx,ctx_vec,ctx_dim,meta_x,meta_dim,arm,"
            "fused_scores_next,overall_score_next,delta_score,obs_reward,smoothed_reward,"
            "drift_flag_next,plateau_steps_next,iter_time_s_next,llm_calls_total_next,llm_tokens_total_next "
            "FROM episodes ORDER BY ts DESC"
        )
        params: tuple[object, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(max(1, int(limit))),)
        with self._connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [Episode.from_row(r) for r in rows]

    def get_replay_counts(self, keys: list[tuple[float, str, int]]) -> dict[tuple[float, str, int], int]:
        if not keys:
            return {}
        with self._connect() as con:
            # Use a temp table-like IN query; chunk to avoid huge SQL.
            out: dict[tuple[float, str, int], int] = {}
            chunk = 250
            for i in range(0, len(keys), chunk):
                ks = keys[i : i + chunk]
                placeholders = ",".join(["(?,?,?)"] * len(ks))
                flat: list[object] = []
                for ts, run_id, it in ks:
                    flat.extend([float(ts), str(run_id), int(it)])
                rows = con.execute(
                    f"SELECT ts,run_id,choice_iter_idx,replay_count FROM episode_replay WHERE (ts,run_id,choice_iter_idx) IN ({placeholders})",
                    tuple(flat),
                ).fetchall()
                for ts, rid, it, c in rows:
                    out[(float(ts), str(rid), int(it))] = int(c)
        return out

    def bump_replay_counts(self, keys: list[tuple[float, str, int]], inc: int = 1) -> None:
        if not keys:
            return
        inc_i = int(max(1, inc))
        with self._connect() as con:
            for ts, rid, it in keys:
                con.execute(
                    "INSERT OR IGNORE INTO episode_replay(ts,run_id,choice_iter_idx,replay_count) VALUES (?,?,?,0)",
                    (float(ts), str(rid), int(it)),
                )
                con.execute(
                    "UPDATE episode_replay SET replay_count = replay_count + ? WHERE ts=? AND run_id=? AND choice_iter_idx=?",
                    (inc_i, float(ts), str(rid), int(it)),
                )


def make_run_id(*, seed: int | None = None, instruction: str | None = None) -> str:
    now = int(time.time() * 1000)
    s = "seed" if seed is None else f"seed{int(seed)}"
    if instruction:
        h = abs(hash(instruction)) % 1_000_000
        return f"{now}_{s}_h{h}"
    return f"{now}_{s}"
