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

NOTE:
This module does not depend on the rest of TriBrain; you can use it standalone.
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
class Schema:
    """Reusable refinement pattern extracted from episodes."""
    id: str
    pattern: str  # e.g., "clarity_improve"
    success_count: int = 0
    failure_count: int = 0
    avg_delta: float = 0.0
    task_families: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    confidence: float = 0.5


@dataclass
class Constraint:
    """Invariant that prevents regressions."""
    id: str
    condition: str  # e.g., "physics_score >= 0.7"
    violation_action: str  # "revert", "arbitrate", "delay"
    violations: int = 0
    enforced: bool = True
    created_at: float = field(default_factory=time.time)


@dataclass
class ArbitrationRule:
    """Stable trade-off among critic dimensions."""
    id: str
    critics: List[str]                 # e.g., ["physics", "task"]
    weights: Dict[str, float]          # Pareto-optimal weights
    contexts: List[str] = field(default_factory=list)  # When this rule applies
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)


@dataclass
class Anchor:
    """Golden regression test (invariant that must not degrade)."""
    id: str
    prompt: str
    expected_scores: Dict[str, float]  # Minimum expected scores per critic
    tolerance: float = 0.05
    violated: bool = False
    last_check: float = field(default_factory=time.time)


@dataclass
class ConsolidationCandidate:
    """Episode being considered for promotion to semantic cortex."""
    episode_id: str
    novelty_score: float
    utility_score: float
    disagreement_score: float
    evidence_score: float
    should_consolidate: bool = False


# -----------------------------
# SQLite utilities
# -----------------------------

def _connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_hash(obj: Any) -> str:
    payload = _json(obj).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _episode_id_from_payload(payload: Dict[str, Any]) -> str:
    for k in ("id", "episode_id", "uuid"):
        v = payload.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return "sha256:" + _canonical_hash(payload)


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
        self.schemas: Dict[str, Schema] = {}
        self.constraints: Dict[str, Constraint] = {}
        self.arbitration_rules: Dict[str, ArbitrationRule] = {}
        self.anchors: Dict[str, Anchor] = {}

        self._load_from_db()

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
                external_validated INTEGER DEFAULT 0,
                signature TEXT,
                embedding TEXT,                  -- JSON list (optional)
                sha256 TEXT NOT NULL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_episodes_family_time ON episodes(task_family, created_at)")

        # Semantic stores
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schemas (
                id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_delta REAL DEFAULT 0.0,
                task_families TEXT,              -- JSON array
                created_at REAL,
                confidence REAL DEFAULT 0.5
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS constraints (
                id TEXT PRIMARY KEY,
                condition TEXT NOT NULL,
                violation_action TEXT NOT NULL,
                violations INTEGER DEFAULT 0,
                enforced INTEGER DEFAULT 1,
                created_at REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS arbitration_rules (
                id TEXT PRIMARY KEY,
                critics TEXT NOT NULL,           -- JSON array
                weights TEXT NOT NULL,           -- JSON dict
                contexts TEXT,                   -- JSON array
                confidence REAL DEFAULT 0.5,
                created_at REAL
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS anchors (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                expected_scores TEXT NOT NULL,   -- JSON dict
                tolerance REAL DEFAULT 0.05,
                violated INTEGER DEFAULT 0,
                last_check REAL
            )
        """)

        conn.commit()
        conn.close()

    def _load_from_db(self) -> None:
        """Load all semantic knowledge into memory."""
        conn = _connect(self.db_path)

        for row in conn.execute("SELECT * FROM schemas"):
            self.schemas[row[0]] = Schema(
                id=row[0],
                pattern=row[1],
                success_count=int(row[2]),
                failure_count=int(row[3]),
                avg_delta=float(row[4]),
                task_families=json.loads(row[5]) if row[5] else [],
                created_at=float(row[6]) if row[6] is not None else time.time(),
                confidence=float(row[7]),
            )

        for row in conn.execute("SELECT * FROM constraints"):
            self.constraints[row[0]] = Constraint(
                id=row[0],
                condition=row[1],
                violation_action=row[2],
                violations=int(row[3]),
                enforced=bool(row[4]),
                created_at=float(row[5]) if row[5] is not None else time.time(),
            )

        for row in conn.execute("SELECT * FROM arbitration_rules"):
            self.arbitration_rules[row[0]] = ArbitrationRule(
                id=row[0],
                critics=json.loads(row[1]),
                weights=json.loads(row[2]),
                contexts=json.loads(row[3]) if row[3] else [],
                confidence=float(row[4]),
                created_at=float(row[5]) if row[5] is not None else time.time(),
            )

        for row in conn.execute("SELECT * FROM anchors"):
            self.anchors[row[0]] = Anchor(
                id=row[0],
                prompt=row[1],
                expected_scores=json.loads(row[2]),
                tolerance=float(row[3]),
                violated=bool(row[4]),
                last_check=float(row[5]) if row[5] is not None else time.time(),
            )

        conn.close()

    # -----------------------------
    # Episodic evidence log (append-only)
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

        payload = _json(ep)
        sha = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        conn = _connect(self.db_path)
        try:
            cur = conn.execute("SELECT 1 FROM episodes WHERE id = ?", (ep_id,))
            exists = cur.fetchone() is not None
            if exists:
                if strict:
                    raise ValueError(f"Episode id already exists: {ep_id}")
                return ep_id

            conn.execute("""
                INSERT INTO episodes
                (id, task_family, created_at, iteration, payload, provenance, external_validated, signature, embedding, sha256)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ep_id,
                str(fam) if fam is not None else None,
                ts,
                it,
                payload,
                provenance,
                1 if external_validated else 0,
                signature,
                _json(list(embedding)) if embedding is not None else None,
                sha,
            ))
            conn.commit()
            return ep_id
        finally:
            conn.close()

    def get_recent_episodes(
        self,
        task_family: Optional[str] = None,
        limit: int = 100,
        since_time: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch recent episodes (immutable evidence) by time descending."""
        limit = max(0, int(limit))
        if limit == 0:
            return []

        conn = _connect(self.db_path)
        try:
            q = "SELECT payload FROM episodes"
            params: List[Any] = []
            where: List[str] = []
            if task_family is not None:
                where.append("task_family = ?")
                params.append(task_family)
            if since_time is not None:
                where.append("created_at >= ?")
                params.append(float(since_time))
            if where:
                q += " WHERE " + " AND ".join(where)
            q += " ORDER BY created_at DESC LIMIT ?"
            params.append(limit)

            out: List[Dict[str, Any]] = []
            for (payload,) in conn.execute(q, tuple(params)):
                try:
                    out.append(json.loads(payload))
                except Exception:
                    continue
            return out
        finally:
            conn.close()

    # -----------------------------
    # Semantic knowledge CRUD
    # -----------------------------

    def add_schema(self, schema: Schema) -> None:
        """Add or update a schema."""
        self.schemas[schema.id] = schema
        conn = _connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO schemas
                (id, pattern, success_count, failure_count, avg_delta, task_families, created_at, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                schema.id, schema.pattern, schema.success_count, schema.failure_count,
                schema.avg_delta, _json(schema.task_families), schema.created_at, schema.confidence
            ))
            conn.commit()
        finally:
            conn.close()

    def add_constraint(self, constraint: Constraint) -> None:
        """Add or update a constraint."""
        self.constraints[constraint.id] = constraint
        conn = _connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO constraints
                (id, condition, violation_action, violations, enforced, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                constraint.id, constraint.condition, constraint.violation_action,
                constraint.violations, int(constraint.enforced), constraint.created_at
            ))
            conn.commit()
        finally:
            conn.close()

    def add_arbitration_rule(self, rule: ArbitrationRule) -> None:
        """Add or update an arbitration rule."""
        # Normalize weights (avoid negative, ensure sum=1)
        weights = {k: max(0.0, float(v)) for k, v in rule.weights.items()}
        s = sum(weights.values())
        if s <= 0:
            # fallback uniform over provided critics
            if rule.critics:
                w = 1.0 / len(rule.critics)
                weights = {c: w for c in rule.critics}
            else:
                weights = {}
        else:
            weights = {k: v / s for k, v in weights.items()}
        rule.weights = weights

        self.arbitration_rules[rule.id] = rule
        conn = _connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO arbitration_rules
                (id, critics, weights, contexts, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                rule.id, _json(rule.critics), _json(rule.weights), _json(rule.contexts),
                rule.confidence, rule.created_at
            ))
            conn.commit()
        finally:
            conn.close()

    def add_anchor(self, anchor: Anchor) -> None:
        """Add or update an anchor regression test."""
        self.anchors[anchor.id] = anchor
        conn = _connect(self.db_path)
        try:
            conn.execute("""
                INSERT OR REPLACE INTO anchors
                (id, prompt, expected_scores, tolerance, violated, last_check)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                anchor.id, anchor.prompt, _json(anchor.expected_scores),
                anchor.tolerance, int(anchor.violated), anchor.last_check
            ))
            conn.commit()
        finally:
            conn.close()

    # -----------------------------
    # Retrieval utilities
    # -----------------------------

    def get_relevant_schemas(self, task_family: str, top_k: int = 3) -> List[Schema]:
        """Retrieve most relevant schemas for a task family."""
        top_k = max(0, int(top_k))
        if top_k == 0:
            return []

        relevant = [s for s in self.schemas.values() if (task_family in s.task_families) or (not s.task_families)]
        relevant.sort(key=lambda s: (s.confidence, s.avg_delta, s.success_count), reverse=True)
        return relevant[:top_k]

    def get_enforced_constraints(self) -> List[Constraint]:
        """Get all active constraints."""
        return [c for c in self.constraints.values() if c.enforced]

    def recommend_arbitration_weights(self, critics: Sequence[str], context: Optional[str] = None) -> Dict[str, float]:
        """
        Select an arbitration rule that matches the requested critics (and optional context).
        Falls back to uniform weights.
        """
        crit_set = set(critics)
        best: Optional[ArbitrationRule] = None
        best_score = -1.0

        for rule in self.arbitration_rules.values():
            if set(rule.critics) != crit_set:
                continue
            if context is not None and rule.contexts and (context not in rule.contexts):
                continue
            # score by confidence and recency
            score = rule.confidence + 0.000001 * rule.created_at
            if score > best_score:
                best_score = score
                best = rule

        if best is None:
            if not critics:
                return {}
            w = 1.0 / len(critics)
            return {c: w for c in critics}
        return dict(best.weights)

    # -----------------------------
    # Anchor checking
    # -----------------------------

    def check_anchors(self, current_scores: Dict[str, float]) -> List[str]:
        """
        Check if current scores violate any anchors.

        Returns list of violated anchor IDs.
        """
        violations: List[str] = []
        now = time.time()

        for anchor_id, anchor in self.anchors.items():
            for critic, expected in anchor.expected_scores.items():
                if critic not in current_scores:
                    continue
                try:
                    actual = float(current_scores[critic])
                except Exception:
                    continue
                if actual < float(expected) - float(anchor.tolerance):
                    violations.append(anchor_id)
                    anchor.violated = True
                    anchor.last_check = now
                    self._update_anchor_violation(anchor_id, now)
                    break

        return violations

    def _update_anchor_violation(self, anchor_id: str, now: float) -> None:
        conn = _connect(self.db_path)
        try:
            conn.execute("UPDATE anchors SET violated = 1, last_check = ? WHERE id = ?", (float(now), anchor_id))
            conn.commit()
        finally:
            conn.close()

    # -----------------------------
    # Memory budget control
    # -----------------------------

    def prune_low_confidence_knowledge(self, threshold: float = 0.3) -> None:
        """Remove schemas and rules with very low confidence."""
        threshold = float(threshold)
        to_remove: List[Tuple[str, str]] = []

        for schema_id, schema in list(self.schemas.items()):
            if schema.confidence < threshold:
                to_remove.append(("schemas", schema_id))

        for rule_id, rule in list(self.arbitration_rules.items()):
            if rule.confidence < threshold:
                to_remove.append(("arbitration_rules", rule_id))

        if not to_remove:
            return

        conn = _connect(self.db_path)
        try:
            for table, id_ in to_remove:
                if table not in ("schemas", "arbitration_rules"):
                    continue
                conn.execute(f"DELETE FROM {table} WHERE id = ?", (id_,))
                if table == "schemas":
                    self.schemas.pop(id_, None)
                elif table == "arbitration_rules":
                    self.arbitration_rules.pop(id_, None)
            conn.commit()
        finally:
            conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get cortex statistics."""
        avg_conf = float(np.mean([s.confidence for s in self.schemas.values()])) if self.schemas else 0.0
        return {
            "num_schemas": len(self.schemas),
            "num_constraints": len(self.constraints),
            "num_arbitration_rules": len(self.arbitration_rules),
            "num_anchors": len(self.anchors),
            "avg_schema_confidence": avg_conf,
            "active_constraints": sum(1 for c in self.constraints.values() if c.enforced),
            "violated_anchors": sum(1 for a in self.anchors.values() if a.violated),
        }


# -----------------------------
# Consolidation gate
# -----------------------------

class ConsolidationGate:
    """
    Gated consolidation mechanism (Section 4.4.2).

    Promotes episodes to semantic cortex only when they are:
    - Novel (different from existing schemas)
    - High-utility (transferable across tasks)
    - High-disagreement (persistent critic conflict)
    - High-evidence (validated or consistent)
    """

    def __init__(
        self,
        novelty_threshold: float = 0.6,
        utility_threshold: float = 0.5,
        disagreement_threshold: float = 0.3,
        evidence_threshold: float = 0.7,
    ):
        self.novelty_threshold = float(novelty_threshold)
        self.utility_threshold = float(utility_threshold)
        self.disagreement_threshold = float(disagreement_threshold)
        self.evidence_threshold = float(evidence_threshold)

    def should_consolidate(self, episode: Dict[str, Any], cortex: SemanticCortex) -> ConsolidationCandidate:
        """Compute consolidation signals for an episode and decide promotion."""
        ep_id = _episode_id_from_payload(episode)

        novelty = self._compute_novelty(episode, cortex)
        utility = self._compute_utility(episode)
        disagreement = self._compute_disagreement(episode)
        evidence = self._compute_evidence(episode)

        should_promote = (
            (novelty >= self.novelty_threshold)
            and (utility >= self.utility_threshold)
            and (disagreement >= self.disagreement_threshold)
            and (evidence >= self.evidence_threshold)
        )

        return ConsolidationCandidate(
            episode_id=ep_id,
            novelty_score=novelty,
            utility_score=utility,
            disagreement_score=disagreement,
            evidence_score=evidence,
            should_consolidate=should_promote,
        )

    def _compute_novelty(self, episode: Dict[str, Any], cortex: SemanticCortex) -> float:
        """Novelty based on dissimilarity to existing schema patterns."""
        pattern = self._extract_pattern(episode)
        if not cortex.schemas:
            return 1.0  # fully novel

        similarities = [self._pattern_similarity(pattern, s.pattern) for s in cortex.schemas.values()]
        max_sim = max(similarities) if similarities else 0.0
        return float(1.0 - max_sim)

    def _compute_utility(self, episode: Dict[str, Any]) -> float:
        """Utility proxy: magnitude of improvement (bounded)."""
        improvement = float(episode.get("improvement", episode.get("delta", 0.0)) or 0.0)
        return float(np.tanh(abs(improvement) / 0.1))

    def _compute_disagreement(self, episode: Dict[str, Any]) -> float:
        """
        Disagreement proxy: std over critic scores.

        Prefer raw critic_scores if available; fall back to fused_scores.
        """
        scores = episode.get("critic_scores") or episode.get("fused_scores") or {}
        if not isinstance(scores, dict) or not scores:
            return 0.0
        vals = []
        for v in scores.values():
            try:
                vals.append(float(v))
            except Exception:
                continue
        if len(vals) < 2:
            return 0.0
        return float(np.std(vals))

    def _compute_evidence(self, episode: Dict[str, Any]) -> float:
        """
        Evidence proxy: combine outcome consistency + optional external validation flag.

        If external_validated==True, boost evidence.
        """
        delta = float(episode.get("delta", 0.0) or 0.0)
        improvement = float(episode.get("improvement", delta) or 0.0)
        base = float(np.tanh(abs(improvement) / 0.1))
        consistent = 1.0 - float(np.tanh(abs(delta - improvement) / 0.1))
        external = 1.0 if bool(episode.get("external_validated", False)) else 0.0
        # Weighted mixture
        evidence = 0.55 * base + 0.35 * consistent + 0.10 * external
        return float(np.clip(evidence, 0.0, 1.0))

    def _extract_pattern(self, episode: Dict[str, Any]) -> str:
        """Extract refinement pattern from episode."""
        focus = str(episode.get("refinement_focus", episode.get("focus", "unknown")))
        delta = float(episode.get("delta", 0.0) or 0.0)
        direction = "improve" if delta > 0 else "degrade"
        return f"{focus}_{direction}"

    def _pattern_similarity(self, p1: str, p2: str) -> float:
        """Compute similarity between two patterns (token overlap)."""
        t1 = set(str(p1).split("_"))
        t2 = set(str(p2).split("_"))
        if not t1 or not t2:
            return 0.0
        return float(len(t1 & t2) / len(t1 | t2))


# -----------------------------
# Homeostatic controls
# -----------------------------

class HomeostaticController:
    """
    Homeostatic controls (Section 4.4.4).

    Prevents runaway optimization through:
    - Bounded update magnitudes (trust regions)
    - Entropy regularization of critic weights
    - Memory budget enforcement
    """

    def __init__(
        self,
        max_update_magnitude: float = 0.1,
        min_weight_entropy: float = 0.5,
        max_memory_size: int = 10000,
        mix_uniform: float = 0.3,
    ):
        self.max_update_magnitude = float(max_update_magnitude)
        self.min_weight_entropy = float(min_weight_entropy)
        self.max_memory_size = int(max_memory_size)
        self.mix_uniform = float(mix_uniform)

    def clip_update(self, old_value: float, new_value: float) -> float:
        """Clip update to a trust region around old_value."""
        old_value = float(old_value)
        new_value = float(new_value)
        delta = new_value - old_value
        clipped_delta = float(np.clip(delta, -self.max_update_magnitude, self.max_update_magnitude))
        return old_value + clipped_delta

    def regularize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        Ensure critic weights have sufficient entropy.

        - If weights is empty: return {}.
        - If only one critic: return {critic: 1.0}.
        """
        if not weights:
            return {}

        keys = list(weights.keys())
        vals = np.array([max(0.0, float(weights[k])) for k in keys], dtype=float)
        s = float(vals.sum())
        if s <= 0.0:
            # fallback uniform
            u = 1.0 / len(keys)
            return {k: u for k in keys}

        vals = vals / s

        if len(vals) == 1:
            return {keys[0]: 1.0}

        entropy = float(-np.sum(vals * np.log(vals + 1e-12)))
        max_entropy = float(np.log(len(vals)))

        # If entropy too low, mix with uniform distribution
        if max_entropy > 0 and (entropy / max_entropy) < self.min_weight_entropy:
            uniform = np.ones(len(vals), dtype=float) / len(vals)
            m = float(np.clip(self.mix_uniform, 0.0, 1.0))
            vals = (1.0 - m) * vals + m * uniform
            vals = vals / float(vals.sum())

        return {k: float(v) for k, v in zip(keys, vals)}

    def enforce_memory_budget(self, current_size: int) -> bool:
        """Return True if memory pruning is needed."""
        return int(current_size) > self.max_memory_size
