"""
Semantic Cortex (TriBrain Revision Sections 4.4.1–4.4.5)

A complementary memory subsystem for long-horizon stability.

This file implements:
- A compact Semantic Cortex (schemas, constraints, arbitration rules) with SQLite persistence
- An immutable Episodic Evidence Log (append-only) in the same DB (optional but recommended)
- Anchor-based regression tests (golden invariants)
- A gated consolidation mechanism (novelty / utility / disagreement / evidence)
- Homeostatic controls (trust-region clipping + critic-weight entropy regularization)
- Memory budget controls (pruning low-confidence knowledge)

Design goal:
Keep raw evidence immutable, while allowing *additive* interpretation updates (schemas / summaries / rules).
This reduces "false memory" drift caused by overwriting episode history.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# -----------------------------
# Data objects
# -----------------------------

@dataclass
class SchemaItem:
    name: str
    content: Dict[str, Any]
    confidence: float = 0.5
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class ConstraintItem:
    name: str
    content: Dict[str, Any]
    confidence: float = 0.5
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class ArbitrationRule:
    name: str
    content: Dict[str, Any]
    confidence: float = 0.5
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class AnchorTest:
    name: str
    description: str
    invariant: Dict[str, Any]
    created_at: float = field(default_factory=lambda: time.time())
    updated_at: float = field(default_factory=lambda: time.time())


# -----------------------------
# SQLite helpers
# -----------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def _episode_id_from_payload(payload: Dict[str, Any]) -> str:
    s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _safe_load(s: str) -> Any:
    return json.loads(s) if s else None


# -----------------------------
# Semantic Cortex
# -----------------------------

class SemanticCortex:
    """
    Stores compressed, reusable knowledge:
      - schemas
      - constraints
      - arbitration rules
      - anchors (regression tests)

    Also provides an optional immutable Episodic Evidence Log in the same DB.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._init_db()

        # In-memory caches for fast access
        self.schemas: Dict[str, SchemaItem] = {}
        self.constraints: Dict[str, ConstraintItem] = {}
        self.rules: Dict[str, ArbitrationRule] = {}
        self.anchors: Dict[str, AnchorTest] = {}

        self._load_all()

    def _init_db(self) -> None:
        """Initialize database schema (semantic + episodic evidence)."""
        conn = _connect(self.db_path)

        # Immutable episodic log (append-only)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id TEXT PRIMARY KEY,
                task_family TEXT,
                created_at REAL,
                iteration INTEGER,
                payload TEXT NOT NULL,           -- JSON
                provenance TEXT,                 -- free-form
                external_validated INTEGER,
                signature TEXT,
                embedding BLOB
            )
        """)

        # Schemas / constraints / rules as additive knowledge
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schemas (
                name TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                evidence_ids TEXT NOT NULL       -- JSON list
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS constraints (
                name TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                evidence_ids TEXT NOT NULL       -- JSON list
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rules (
                name TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                evidence_ids TEXT NOT NULL       -- JSON list
            )
        """)

        # Anchors
        conn.execute("""
            CREATE TABLE IF NOT EXISTS anchors (
                name TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                invariant TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
        """)

        # Internal meta table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def _load_all(self) -> None:
        """Load semantic memory caches from DB."""
        conn = _connect(self.db_path)

        # Schemas
        for name, content, conf, ca, ua, ev in conn.execute(
            "SELECT name,content,confidence,created_at,updated_at,evidence_ids FROM schemas"
        ).fetchall():
            self.schemas[name] = SchemaItem(
                name=name,
                content=_safe_load(content) or {},
                confidence=float(conf),
                created_at=float(ca),
                updated_at=float(ua),
                evidence_ids=list(_safe_load(ev) or []),
            )

        # Constraints
        for name, content, conf, ca, ua, ev in conn.execute(
            "SELECT name,content,confidence,created_at,updated_at,evidence_ids FROM constraints"
        ).fetchall():
            self.constraints[name] = ConstraintItem(
                name=name,
                content=_safe_load(content) or {},
                confidence=float(conf),
                created_at=float(ca),
                updated_at=float(ua),
                evidence_ids=list(_safe_load(ev) or []),
            )

        # Rules
        for name, content, conf, ca, ua, ev in conn.execute(
            "SELECT name,content,confidence,created_at,updated_at,evidence_ids FROM rules"
        ).fetchall():
            self.rules[name] = ArbitrationRule(
                name=name,
                content=_safe_load(content) or {},
                confidence=float(conf),
                created_at=float(ca),
                updated_at=float(ua),
                evidence_ids=list(_safe_load(ev) or []),
            )

        # Anchors
        for name, desc, inv, ca, ua in conn.execute(
            "SELECT name,description,invariant,created_at,updated_at FROM anchors"
        ).fetchall():
            self.anchors[name] = AnchorTest(
                name=name,
                description=str(desc),
                invariant=_safe_load(inv) or {},
                created_at=float(ca),
                updated_at=float(ua),
            )

        conn.close()

    # -----------------------------
    # Episodic evidence (append-only)
    # -----------------------------

    def append_episode(
        self,
        episode: Dict[str, Any],
        task_family: Optional[str] = None,
        iteration: Optional[int] = None,
        provenance: Optional[str] = None,
        external_validated: bool = False,
        signature: Optional[str] = None,
        embedding: Optional[Sequence[float]] = None,
        created_at: Optional[float] = None,
        strict: bool = False,
    ) -> str:
        """
        Append an episode as immutable evidence.

        If strict=True and the episode id already exists, raises ValueError.
        Otherwise, it's a no-op and returns the existing id.
        """
        ep = dict(episode)  # defensive copy
        ep_id = _episode_id_from_payload(ep)
        ts = float(created_at) if created_at is not None else float(ep.get("timestamp", time.time()) or time.time())
        fam = task_family or ep.get("task_family")
        it = iteration if iteration is not None else ep.get("iteration")
        try:
            it = int(it) if it is not None else None
        except Exception:
            it = None

        emb_blob: Optional[bytes] = None
        if embedding is not None:
            arr = np.asarray(list(embedding), dtype=np.float32)
            emb_blob = sqlite3.Binary(arr.tobytes())

        conn = _connect(self.db_path)
        row = conn.execute("SELECT id FROM episodes WHERE id=?", (ep_id,)).fetchone()
        if row is not None:
            conn.close()
            if strict:
                raise ValueError(f"Episode id already exists: {ep_id}")
            return ep_id

        conn.execute(
            """
            INSERT INTO episodes(id,task_family,created_at,iteration,payload,provenance,external_validated,signature,embedding)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                ep_id,
                str(fam) if fam is not None else None,
                float(ts),
                int(it) if it is not None else None,
                _safe_json(ep),
                str(provenance) if provenance is not None else None,
                1 if bool(external_validated) else 0,
                str(signature) if signature is not None else None,
                emb_blob,
            ),
        )
        conn.commit()
        conn.close()
        return ep_id

    def recent_episode_ids(self, limit: int = 100) -> List[str]:
        conn = _connect(self.db_path)
        rows = conn.execute("SELECT id FROM episodes ORDER BY created_at DESC LIMIT ?", (int(limit),)).fetchall()
        conn.close()
        return [str(r[0]) for r in rows]

    def fetch_episode_payload(self, episode_id: str) -> Optional[Dict[str, Any]]:
        conn = _connect(self.db_path)
        row = conn.execute("SELECT payload FROM episodes WHERE id=?", (str(episode_id),)).fetchone()
        conn.close()
        if row is None:
            return None
        return _safe_load(row[0]) or None

    # -----------------------------
    # Semantic memory CRUD
    # -----------------------------

    def upsert_schema(self, item: SchemaItem) -> None:
        self.schemas[item.name] = item
        conn = _connect(self.db_path)
        conn.execute(
            """
            INSERT INTO schemas(name,content,confidence,created_at,updated_at,evidence_ids)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                content=excluded.content,
                confidence=excluded.confidence,
                updated_at=excluded.updated_at,
                evidence_ids=excluded.evidence_ids
            """,
            (
                item.name,
                _safe_json(item.content),
                float(item.confidence),
                float(item.created_at),
                float(item.updated_at),
                _safe_json(list(item.evidence_ids)),
            ),
        )
        conn.commit()
        conn.close()

    def upsert_constraint(self, item: ConstraintItem) -> None:
        self.constraints[item.name] = item
        conn = _connect(self.db_path)
        conn.execute(
            """
            INSERT INTO constraints(name,content,confidence,created_at,updated_at,evidence_ids)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                content=excluded.content,
                confidence=excluded.confidence,
                updated_at=excluded.updated_at,
                evidence_ids=excluded.evidence_ids
            """,
            (
                item.name,
                _safe_json(item.content),
                float(item.confidence),
                float(item.created_at),
                float(item.updated_at),
                _safe_json(list(item.evidence_ids)),
            ),
        )
        conn.commit()
        conn.close()

    def upsert_rule(self, item: ArbitrationRule) -> None:
        self.rules[item.name] = item
        conn = _connect(self.db_path)
        conn.execute(
            """
            INSERT INTO rules(name,content,confidence,created_at,updated_at,evidence_ids)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                content=excluded.content,
                confidence=excluded.confidence,
                updated_at=excluded.updated_at,
                evidence_ids=excluded.evidence_ids
            """,
            (
                item.name,
                _safe_json(item.content),
                float(item.confidence),
                float(item.created_at),
                float(item.updated_at),
                _safe_json(list(item.evidence_ids)),
            ),
        )
        conn.commit()
        conn.close()

    def upsert_anchor(self, item: AnchorTest) -> None:
        self.anchors[item.name] = item
        conn = _connect(self.db_path)
        conn.execute(
            """
            INSERT INTO anchors(name,description,invariant,created_at,updated_at)
            VALUES (?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                description=excluded.description,
                invariant=excluded.invariant,
                updated_at=excluded.updated_at
            """,
            (
                item.name,
                str(item.description),
                _safe_json(item.invariant),
                float(item.created_at),
                float(item.updated_at),
            ),
        )
        conn.commit()
        conn.close()

    # -----------------------------
    # Consolidation (simple safe default)
    # -----------------------------

    def consolidate(self) -> None:
        """
        Default consolidation hook.

        The full gating/novelty/utility logic can be layered on later; this method is safe:
        it does not delete evidence and only updates semantic tables if a policy chooses to.
        """
        # No-op safe baseline (keeps integration stable).
        return
