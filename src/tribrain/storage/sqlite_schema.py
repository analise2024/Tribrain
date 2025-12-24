"""SQLite schema definitions.

TriBrain is local-first and uses SQLite for episodic memory.

We keep migrations deliberately boring and production-grade:
  - idempotent
  - atomic (single transaction)
  - no repeated ALTER TABLE
  - schema verified via PRAGMA
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnDef:
    name: str
    col_type: str
    not_null: bool = True
    # Raw SQL default (e.g., "0", "'{}'", "X''").
    default_sql: str | None = None


@dataclass(frozen=True)
class TableSchema:
    name: str
    version: int
    columns: Sequence[ColumnDef]
    indexes: Sequence[str] = ()

    def create_table_sql(self, name_override: str | None = None) -> str:
        tn = name_override or self.name
        parts: list[str] = []
        for c in self.columns:
            s = f"{c.name} {c.col_type}"
            if c.not_null:
                s += " NOT NULL"
            if c.default_sql is not None:
                s += f" DEFAULT {c.default_sql}"
            parts.append(s)
        return f"CREATE TABLE {tn} ({', '.join(parts)})"


# Episodic memory schema (v3)
EPISODES_SCHEMA_V3 = TableSchema(
    name="episodes",
    version=3,
    columns=(
        ColumnDef("ts", "REAL", not_null=True, default_sql="0"),
        ColumnDef("run_id", "TEXT", not_null=True, default_sql="''"),
        ColumnDef("choice_iter_idx", "INTEGER", not_null=True, default_sql="0"),
        ColumnDef("ctx_vec", "BLOB", not_null=True, default_sql="X''"),
        ColumnDef("ctx_dim", "INTEGER", not_null=True, default_sql="0"),
        ColumnDef("meta_x", "BLOB", not_null=True, default_sql="X''"),
        ColumnDef("meta_dim", "INTEGER", not_null=True, default_sql="0"),
        ColumnDef("arm", "TEXT", not_null=True, default_sql="''"),
        ColumnDef("fused_scores_next", "TEXT", not_null=True, default_sql="'{}'"),
        ColumnDef("overall_score_next", "REAL", not_null=True, default_sql="0"),
        ColumnDef("delta_score", "REAL", not_null=True, default_sql="0"),
        ColumnDef("obs_reward", "REAL", not_null=True, default_sql="0"),
        ColumnDef("smoothed_reward", "REAL", not_null=True, default_sql="0"),
        ColumnDef("drift_flag_next", "INTEGER", not_null=True, default_sql="0"),
        ColumnDef("plateau_steps_next", "INTEGER", not_null=True, default_sql="0"),
        ColumnDef("iter_time_s_next", "REAL", not_null=True, default_sql="0"),
        ColumnDef("llm_calls_total_next", "INTEGER", not_null=True, default_sql="0"),
        ColumnDef("llm_tokens_total_next", "INTEGER", not_null=True, default_sql="0"),
    ),
    indexes=(
        "CREATE INDEX IF NOT EXISTS idx_ep_time ON episodes(ts)",
        "CREATE INDEX IF NOT EXISTS idx_ep_arm ON episodes(arm)",
        "CREATE INDEX IF NOT EXISTS idx_ep_run ON episodes(run_id)",
        "CREATE INDEX IF NOT EXISTS idx_ep_choice_iter ON episodes(choice_iter_idx)",
    ),
)
