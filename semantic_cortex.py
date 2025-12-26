"""
Semantic Cortex - Complementary memory system for long-horizon stability.

Implements Section 4.4.1-4.4.5 from the revised TriBrain paper:
- Immutable episodic evidence log (append-only)
- Compact semantic cortex (schemas, constraints, arbitration rules)
- Gated consolidation (novelty, utility, disagreement)
- Bi-phasic replay (online micro-updates + offline consolidation)
- Homeostatic controls (bounded updates, entropy regularization)
- Anchor-based regression tests (golden invariants)
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class Schema:
    """Reusable refinement pattern extracted from episodes."""
    id: str
    pattern: str  # e.g., "physics_failure → stable_grasp_constraint"
    success_count: int = 0
    failure_count: int = 0
    avg_delta: float = 0.0
    task_families: list[str] = field(default_factory=list)
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
    critics: list[str]  # e.g., ["physics", "task"]
    weights: dict[str, float]  # Pareto-optimal weights
    contexts: list[str]  # When this rule applies
    confidence: float = 0.5
    created_at: float = field(default_factory=time.time)


@dataclass
class Anchor:
    """Golden regression test (invariant that must not degrade)."""
    id: str
    prompt: str
    expected_scores: dict[str, float]  # Min expected scores per critic
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


class SemanticCortex:
    """
    Complementary memory system that stores compressed, reusable knowledge.
    
    Unlike the episodic store (immutable evidence log), the semantic cortex
    is slowly updated with schemas, constraints, and arbitration rules that
    are stable across tasks.
    """
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()
        
        # In-memory caches for fast access
        self.schemas: dict[str, Schema] = {}
        self.constraints: dict[str, Constraint] = {}
        self.arbitration_rules: dict[str, ArbitrationRule] = {}
        self.anchors: dict[str, Anchor] = {}
        
        self._load_from_db()
        
    def _init_db(self):
        """Initialize semantic cortex database schema."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS schemas (
                id TEXT PRIMARY KEY,
                pattern TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                avg_delta REAL DEFAULT 0.0,
                task_families TEXT,  -- JSON array
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
                critics TEXT NOT NULL,  -- JSON array
                weights TEXT NOT NULL,  -- JSON dict
                contexts TEXT,  -- JSON array
                confidence REAL DEFAULT 0.5,
                created_at REAL
            )
        """)
        
        conn.execute("""
            CREATE TABLE IF NOT EXISTS anchors (
                id TEXT PRIMARY KEY,
                prompt TEXT NOT NULL,
                expected_scores TEXT NOT NULL,  -- JSON dict
                tolerance REAL DEFAULT 0.05,
                violated INTEGER DEFAULT 0,
                last_check REAL
            )
        """)
        
        conn.commit()
        conn.close()
        
    def _load_from_db(self):
        """Load all semantic knowledge into memory."""
        conn = sqlite3.connect(self.db_path)
        
        # Load schemas
        for row in conn.execute("SELECT * FROM schemas"):
            schema = Schema(
                id=row[0],
                pattern=row[1],
                success_count=row[2],
                failure_count=row[3],
                avg_delta=row[4],
                task_families=json.loads(row[5]) if row[5] else [],
                created_at=row[6],
                confidence=row[7],
            )
            self.schemas[schema.id] = schema
            
        # Load constraints
        for row in conn.execute("SELECT * FROM constraints"):
            constraint = Constraint(
                id=row[0],
                condition=row[1],
                violation_action=row[2],
                violations=row[3],
                enforced=bool(row[4]),
                created_at=row[5],
            )
            self.constraints[constraint.id] = constraint
            
        # Load arbitration rules
        for row in conn.execute("SELECT * FROM arbitration_rules"):
            rule = ArbitrationRule(
                id=row[0],
                critics=json.loads(row[1]),
                weights=json.loads(row[2]),
                contexts=json.loads(row[3]) if row[3] else [],
                confidence=row[4],
                created_at=row[5],
            )
            self.arbitration_rules[rule.id] = rule
            
        # Load anchors
        for row in conn.execute("SELECT * FROM anchors"):
            anchor = Anchor(
                id=row[0],
                prompt=row[1],
                expected_scores=json.loads(row[2]),
                tolerance=row[3],
                violated=bool(row[4]),
                last_check=row[5],
            )
            self.anchors[anchor.id] = anchor
            
        conn.close()
        
    def add_schema(self, schema: Schema):
        """Add or update a schema in the cortex."""
        self.schemas[schema.id] = schema
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO schemas
            (id, pattern, success_count, failure_count, avg_delta, task_families, created_at, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            schema.id, schema.pattern, schema.success_count, schema.failure_count,
            schema.avg_delta, json.dumps(schema.task_families), schema.created_at, schema.confidence
        ))
        conn.commit()
        conn.close()
        
    def add_constraint(self, constraint: Constraint):
        """Add a new constraint to the cortex."""
        self.constraints[constraint.id] = constraint
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO constraints
            (id, condition, violation_action, violations, enforced, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            constraint.id, constraint.condition, constraint.violation_action,
            constraint.violations, int(constraint.enforced), constraint.created_at
        ))
        conn.commit()
        conn.close()
        
    def add_anchor(self, anchor: Anchor):
        """Add a regression test anchor."""
        self.anchors[anchor.id] = anchor
        
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO anchors
            (id, prompt, expected_scores, tolerance, violated, last_check)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            anchor.id, anchor.prompt, json.dumps(anchor.expected_scores),
            anchor.tolerance, int(anchor.violated), anchor.last_check
        ))
        conn.commit()
        conn.close()
        
    def check_anchors(self, current_scores: dict[str, float]) -> list[str]:
        """
        Check if current scores violate any anchors.
        
        Returns list of violated anchor IDs.
        """
        violations = []
        
        for anchor_id, anchor in self.anchors.items():
            for critic, expected in anchor.expected_scores.items():
                if critic in current_scores:
                    actual = current_scores[critic]
                    if actual < expected - anchor.tolerance:
                        violations.append(anchor_id)
                        anchor.violated = True
                        self._update_anchor_violation(anchor_id)
                        break
                        
        return violations
        
    def _update_anchor_violation(self, anchor_id: str):
        """Mark anchor as violated in database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            UPDATE anchors SET violated = 1, last_check = ? WHERE id = ?
        """, (time.time(), anchor_id))
        conn.commit()
        conn.close()
        
    def get_relevant_schemas(self, task_family: str, top_k: int = 3) -> list[Schema]:
        """Retrieve most relevant schemas for a task family."""
        relevant = [
            s for s in self.schemas.values()
            if task_family in s.task_families or not s.task_families
        ]
        
        # Sort by confidence and success rate
        relevant.sort(
            key=lambda s: (s.confidence, s.avg_delta, s.success_count),
            reverse=True
        )
        
        return relevant[:top_k]
        
    def get_enforced_constraints(self) -> list[Constraint]:
        """Get all active constraints."""
        return [c for c in self.constraints.values() if c.enforced]
        
    def prune_low_confidence_knowledge(self, threshold: float = 0.3):
        """Remove schemas and rules with very low confidence (memory budget control)."""
        to_remove = []
        
        for schema_id, schema in self.schemas.items():
            if schema.confidence < threshold:
                to_remove.append(("schemas", schema_id))
                
        for rule_id, rule in self.arbitration_rules.items():
            if rule.confidence < threshold:
                to_remove.append(("arbitration_rules", rule_id))
                
        if to_remove:
            conn = sqlite3.connect(self.db_path)
            for table, id_ in to_remove:
                conn.execute(f"DELETE FROM {table} WHERE id = ?", (id_,))
                if table == "schemas":
                    del self.schemas[id_]
                elif table == "arbitration_rules":
                    del self.arbitration_rules[id_]
            conn.commit()
            conn.close()
            
    def get_stats(self) -> dict[str, Any]:
        """Get cortex statistics."""
        return {
            "num_schemas": len(self.schemas),
            "num_constraints": len(self.constraints),
            "num_arbitration_rules": len(self.arbitration_rules),
            "num_anchors": len(self.anchors),
            "avg_schema_confidence": np.mean([s.confidence for s in self.schemas.values()]) if self.schemas else 0.0,
            "active_constraints": sum(1 for c in self.constraints.values() if c.enforced),
            "violated_anchors": sum(1 for a in self.anchors.values() if a.violated),
        }


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
        self.novelty_threshold = novelty_threshold
        self.utility_threshold = utility_threshold
        self.disagreement_threshold = disagreement_threshold
        self.evidence_threshold = evidence_threshold
        
    def should_consolidate(
        self,
        episode: dict[str, Any],
        existing_schemas: Sequence[Schema],
    ) -> ConsolidationCandidate:
        """
        Determine if episode should be promoted to semantic cortex.
        """
        novelty = self._compute_novelty(episode, existing_schemas)
        utility = self._compute_utility(episode)
        disagreement = self._compute_disagreement(episode)
        evidence = self._compute_evidence(episode)
        
        # Consolidate if episode passes multiple thresholds
        should_promote = (
            (novelty > self.novelty_threshold) or
            (utility > self.utility_threshold and evidence > self.evidence_threshold) or
            (disagreement > self.disagreement_threshold and evidence > 0.5)
        )
        
        return ConsolidationCandidate(
            episode_id=episode.get("id", "unknown"),
            novelty_score=novelty,
            utility_score=utility,
            disagreement_score=disagreement,
            evidence_score=evidence,
            should_consolidate=should_promote,
        )
        
    def _compute_novelty(self, episode: dict, schemas: Sequence[Schema]) -> float:
        """Measure how different episode is from existing schemas."""
        if not schemas:
            return 1.0
            
        # Simple heuristic: check if episode's pattern matches any schema
        episode_pattern = self._extract_pattern(episode)
        
        max_similarity = 0.0
        for schema in schemas:
            sim = self._pattern_similarity(episode_pattern, schema.pattern)
            max_similarity = max(max_similarity, sim)
            
        return 1.0 - max_similarity
        
    def _compute_utility(self, episode: dict) -> float:
        """Estimate transferability across tasks."""
        # High utility if:
        # - Large improvement delta
        # - Simple pattern (likely to generalize)
        # - Works on multiple task families
        
        delta = episode.get("delta", 0.0)
        complexity = len(episode.get("refinement_focus", "").split())
        
        utility = np.tanh(delta / 0.1)  # Normalize delta
        utility *= (1.0 / (1.0 + complexity / 10.0))  # Prefer simpler patterns
        
        return float(np.clip(utility, 0.0, 1.0))
        
    def _compute_disagreement(self, episode: dict) -> float:
        """Measure persistent critic conflict."""
        scores = episode.get("fused_scores", {})
        if len(scores) < 2:
            return 0.0
            
        values = list(scores.values())
        return float(np.std(values))
        
    def _compute_evidence(self, episode: dict) -> float:
        """Quality of evidence (consistency, validation)."""
        # High evidence if:
        # - Multiple iterations confirmed improvement
        # - No reversals
        # - High critic agreement in final state
        
        improvement = episode.get("delta", 0.0)
        final_scores = episode.get("fused_scores", {})
        
        # High evidence if improvement is positive and scores are high
        evidence = np.tanh(improvement / 0.1) * np.mean(list(final_scores.values()) if final_scores else [0.5])
        
        return float(np.clip(evidence, 0.0, 1.0))
        
    def _extract_pattern(self, episode: dict) -> str:
        """Extract refinement pattern from episode."""
        # Simplified: focus + delta direction
        focus = episode.get("refinement_focus", "unknown")
        delta = episode.get("delta", 0.0)
        direction = "improve" if delta > 0 else "degrade"
        return f"{focus}_{direction}"
        
    def _pattern_similarity(self, p1: str, p2: str) -> float:
        """Compute similarity between two patterns."""
        # Simple token overlap
        t1 = set(p1.split("_"))
        t2 = set(p2.split("_"))
        if not t1 or not t2:
            return 0.0
        return len(t1 & t2) / len(t1 | t2)


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
    ):
        self.max_update_magnitude = max_update_magnitude
        self.min_weight_entropy = min_weight_entropy
        self.max_memory_size = max_memory_size
        
    def clip_update(self, old_value: float, new_value: float) -> float:
        """Clip update to trust region."""
        delta = new_value - old_value
        clipped_delta = np.clip(delta, -self.max_update_magnitude, self.max_update_magnitude)
        return old_value + clipped_delta
        
    def regularize_weights(self, weights: dict[str, float]) -> dict[str, float]:
        """Ensure minimum entropy in critic weights."""
        values = np.array(list(weights.values()))
        
        # Normalize
        values = values / (values.sum() + 1e-10)
        
        # Compute entropy
        entropy = -np.sum(values * np.log(values + 1e-10))
        max_entropy = np.log(len(values))
        
        # If entropy too low, add uniform noise
        if entropy / max_entropy < self.min_weight_entropy:
            uniform = np.ones(len(values)) / len(values)
            values = 0.7 * values + 0.3 * uniform
            values = values / values.sum()
            
        return {k: float(v) for k, v in zip(weights.keys(), values)}
        
    def enforce_memory_budget(self, current_size: int) -> bool:
        """Check if memory pruning is needed."""
        return current_size > self.max_memory_size
