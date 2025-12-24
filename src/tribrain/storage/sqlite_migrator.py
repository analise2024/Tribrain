"""Idempotent, atomic SQLite migration helper.

Why this exists:
  - Repeated ALTER TABLE migrations are fragile and often break in the real world
    ("duplicate column" / partially applied schema / locked DB).
  - The robust pattern is *rebuild migrations* inside a transaction:
      CREATE new_table
      INSERT common columns from old_table
      DROP old_table
      ALTER TABLE new_table RENAME TO old_table

Properties:
  - Works for new DBs and old DBs
  - Transactional (BEGIN IMMEDIATE)
  - PRAGMA-based schema inspection
  - Optional automatic backup
"""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from .sqlite_schema import TableSchema


@dataclass(frozen=True)
class MigrationResult:
    changed: bool
    from_version: int
    to_version: int


class SQLiteMigrator:
    def __init__(self, db_path: Path, *, backup: bool = True, timeout_s: float = 30.0):
        self.db_path = Path(db_path)
        self.backup = bool(backup)
        self.timeout_s = float(timeout_s)

    def migrate(self, target: TableSchema) -> MigrationResult:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.db_path.exists():
            with self._connect() as con:
                self._ensure_version_table(con)
                self._create_table(con, target)
                self._set_version(con, target.version)
            return MigrationResult(changed=True, from_version=0, to_version=target.version)

        with self._connect() as con:
            self._ensure_version_table(con)
            cur_version = self._get_version(con)

            # Fast-path: already at version and schema matches.
            if cur_version == target.version and self._schema_matches(con, target):
                return MigrationResult(changed=False, from_version=cur_version, to_version=target.version)

        # Only back up if a migration is needed.
        if self.backup:
            bak = self.db_path.with_suffix(self.db_path.suffix + ".bak")
            # Backup is best-effort; continue on errors.
            with suppress(Exception):
                shutil.copy2(self.db_path, bak)

        with self._connect() as con:
            self._ensure_version_table(con)
            cur_version = self._get_version(con)

            # Atomic migration.
            con.execute("PRAGMA foreign_keys=OFF")
            con.execute("BEGIN IMMEDIATE")
            try:
                # If table missing, just create.
                if not self._table_exists(con, target.name):
                    self._create_table(con, target)
                elif not self._schema_matches(con, target):
                    self._rebuild_table(con, target)

                self._set_version(con, target.version)
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise

        return MigrationResult(changed=True, from_version=cur_version, to_version=target.version)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(str(self.db_path), timeout=self.timeout_s)
        try:
            # WAL is a good default for concurrent reads while writing.
            with suppress(Exception):
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

    @staticmethod
    def _ensure_version_table(con: sqlite3.Connection) -> None:
        con.execute("CREATE TABLE IF NOT EXISTS __schema_version (v INTEGER NOT NULL)")
        # Ensure single-row invariant.
        row = con.execute("SELECT COUNT(*) FROM __schema_version").fetchone()
        if row and int(row[0]) == 0:
            con.execute("INSERT INTO __schema_version (v) VALUES (0)")

    @staticmethod
    def _get_version(con: sqlite3.Connection) -> int:
        row = con.execute("SELECT v FROM __schema_version LIMIT 1").fetchone()
        return int(row[0]) if row and row[0] is not None else 0

    @staticmethod
    def _set_version(con: sqlite3.Connection, v: int) -> None:
        con.execute("DELETE FROM __schema_version")
        con.execute("INSERT INTO __schema_version (v) VALUES (?)", (int(v),))

    @staticmethod
    def _table_exists(con: sqlite3.Connection, name: str) -> bool:
        row = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (str(name),)
        ).fetchone()
        return row is not None

    @staticmethod
    def _existing_columns(con: sqlite3.Connection, table: str) -> dict[str, str]:
        rows = con.execute(f"PRAGMA table_info({table})").fetchall()
        # table_info: cid, name, type, notnull, dflt_value, pk
        return {str(r[1]): str(r[2]) for r in rows}

    def _schema_matches(self, con: sqlite3.Connection, schema: TableSchema) -> bool:
        if not self._table_exists(con, schema.name):
            return False
        cols = self._existing_columns(con, schema.name)
        expected = {c.name: c.col_type.upper() for c in schema.columns}
        for name, col_type in expected.items():
            if name not in cols:
                return False
            # Type affinity is fuzzy in SQLite; do a conservative check.
            # Allow empty types in legacy DBs.
            if cols[name].upper() != col_type and cols[name].strip() != "":
                return False
        return True

    @staticmethod
    def _create_table(con: sqlite3.Connection, schema: TableSchema, *, name_override: str | None = None) -> None:
        con.execute(schema.create_table_sql(name_override=name_override))
        # Create indexes only for the canonical name.
        if name_override is None:
            for idx in schema.indexes:
                con.execute(idx)

    def _rebuild_table(self, con: sqlite3.Connection, schema: TableSchema) -> None:
        tmp = f"{schema.name}__new"
        con.execute(f"DROP TABLE IF EXISTS {tmp}")
        self._create_table(con, schema, name_override=tmp)

        existing = self._existing_columns(con, schema.name)
        common = [c.name for c in schema.columns if c.name in existing]
        if common:
            cols_sql = ",".join(common)
            con.execute(
                f"INSERT INTO {tmp} ({cols_sql}) SELECT {cols_sql} FROM {schema.name}"
            )

        con.execute(f"DROP TABLE IF EXISTS {schema.name}")
        con.execute(f"ALTER TABLE {tmp} RENAME TO {schema.name}")
        for idx in schema.indexes:
            con.execute(idx)
