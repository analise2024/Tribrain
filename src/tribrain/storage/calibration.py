from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from ..brains.belief import BetaReliability


@dataclass
class CalibrationStore:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            yield con
            con.commit()
        except Exception:
            with suppress(Exception):
                con.rollback()
            raise
        finally:
            con.close()

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS critic_calibration (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts INTEGER NOT NULL,
                    critic_name TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    weight REAL NOT NULL,
                    tags_json TEXT NOT NULL
                );
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_critic_name ON critic_calibration(critic_name);")

    def add_observation(
        self,
        critic_name: str,
        success: bool,
        weight: float = 1.0,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        tags = dict(tags or {})
        with self._connect() as con:
            con.execute(
                "INSERT INTO critic_calibration (ts, critic_name, success, weight, tags_json) VALUES (?, ?, ?, ?, ?)",
                (int(time.time()), critic_name, int(bool(success)), float(weight), json.dumps(tags)),
            )

    def reliability(self, critic_name: str) -> BetaReliability:
        # Aggregate as Beta(1+success_weight, 1+fail_weight)
        with self._connect() as con:
            rows = con.execute(
                "SELECT success, weight FROM critic_calibration WHERE critic_name = ?",
                (critic_name,),
            ).fetchall()
        a, b = 1.0, 1.0
        for success, w in rows:
            if int(success) == 1:
                a += float(w)
            else:
                b += float(w)
        return BetaReliability(a=a, b=b)

    def list_critics(self) -> list[str]:
        with self._connect() as con:
            rows = con.execute("SELECT DISTINCT critic_name FROM critic_calibration").fetchall()
        return [str(r[0]) for r in rows if r and r[0] is not None]
