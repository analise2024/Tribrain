from __future__ import annotations

import json
import os
import random
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .belief.bayesian_belief import BeliefConfig, BayesianBeliefBrain
from .budget.budget import BudgetConfig, BudgetController, BudgetState
from .controller.bandit import PromptEditBandit
from .controller.hierarchical_controller import HierarchicalContextualTS
from .critics.critic_team import CriticTeam
from .critics.simple_critics import InstructionFollowingCritic, PhysicsSanityCritic
from .dashboard.report import build_report
from .memory.episodic import Episode, EpisodeStore, make_run_id
from .memory.semantic_cortex import SemanticCortex
from .memory.replay import ReplayConfig, ReplayEngine
from .meta.meta_controller import MetaConfig, MetaController
from .meta.meta_value_model import MetaValueModel, MetaValueModelConfig
from .ontology.extractor import OntologyExtractor
from .reward.preference_dataset import build_preference_pairs
from .reward.reward_model import RewardModelConfig, SimpleRewardModel
from .storage.calibration_store import CalibrationStore
from .storage.checkpoint_store import CheckpointStore
from .storage.trace_writer import JsonlTraceWriter
from .types import GoalSpec, RunResult


@dataclass
class OrchestratorConfig:
    max_iters: int = 4
    target_score: float = 0.85
    patience: int = 2
    min_delta: float = 0.01
    drift_threshold: float = 0.15
    # Budgets: hard caps to prevent runaway inference cost.
    max_wall_time_s: float = 20 * 60
    max_iter_time_s: float = 10 * 60
    llm_max_calls: int = 4
    llm_max_tokens: int = 1200
    llm_policy: str = "adaptive"  # adaptive|always|never
    # RL / Controller
    controller: str = "hierarchical"  # hierarchical|bandit
    bow_dim: int = 32
    controller_noise_var: float = 0.10
    # Optional LLM refiner and ontology extractor (can be disabled).
    use_llm_refiner: bool = False
    use_ontology: bool = True
    # Continual learning via episodic replay
    use_episodic_memory: bool = True
    episodes_db: str = "episodes.sqlite"
    replay_k: int = 256
    replay_strategy: str = "uniform"  # uniform|nearest|biphase
    replay_seed: int = 0

    # Optional semantic cortex (complementary long-horizon memory)
    use_semantic_cortex: bool = False
    semantic_cortex_db: str = "semantic_cortex.sqlite"
    semantic_consolidate_every: int = 0  # 0=never, N=every N episodes

    # Metacognitive value model (learns when continuing is worth it)
    use_meta_value_model: bool = True
    meta_value_min_updates: int = 30
    meta_value_min_expected_delta: float = 0.005
    meta_value_cost_penalty: float = 0.15
    meta_value_min_value: float = 0.0


class TriBrainOrchestrator:
    def __init__(
        self,
        out_dir: Path,
        cfg: OrchestratorConfig,
        *,
        llm_refiner: Any | None = None,
        ontology_extractor: OntologyExtractor | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg

        # Storage
        self.calibration = CalibrationStore(self.out_dir / "calibration.sqlite3")
        self.checkpoints = CheckpointStore(self.out_dir / "checkpoint.json")

        self.trace = JsonlTraceWriter(self.out_dir / "trace.jsonl")
        # Run identity (set at run() time once the seed is known)
        self.run_id = ""

        # Episodic memory + replay (continual improvement without re-running expensive generations)
        self.episode_store: EpisodeStore | None = None
        self.replay_engine: ReplayEngine | None = None
        self.semantic_cortex: SemanticCortex | None = None
        if self.cfg.use_episodic_memory:
            self.episode_store = EpisodeStore(self.out_dir / self.cfg.episodes_db)
            self.replay_engine = ReplayEngine(
                self.episode_store,
                ReplayConfig(k=int(self.cfg.replay_k), strategy=str(self.cfg.replay_strategy), seed=int(self.cfg.replay_seed)),
            )

            if self.cfg.use_semantic_cortex:
                # Separate DB: does not affect episodic migration/tests.
                self.semantic_cortex = SemanticCortex(self.out_dir / self.cfg.semantic_cortex_db)

        # Metacognitive value model: learns marginal value of "one more iteration"
        mv_cfg = MetaValueModelConfig(
            min_updates=int(self.cfg.meta_value_min_updates),
            min_expected_delta=float(self.cfg.meta_value_min_expected_delta),
            cost_penalty=float(self.cfg.meta_value_cost_penalty),
            min_value=float(self.cfg.meta_value_min_value),
        )
        self.meta_value_model = MetaValueModel(mv_cfg) if self.cfg.use_meta_value_model else None

        # Critics
        critics = [InstructionFollowingCritic(), PhysicsSanityCritic()]
        self.critic_team = CriticTeam(critics)

        # Brains
        belief_cfg = BeliefConfig(drift_threshold=float(self.cfg.drift_threshold))
        self.belief_brain = BayesianBeliefBrain(belief_cfg)

        meta_cfg = MetaConfig(
            target_score=float(self.cfg.target_score),
            patience=int(self.cfg.patience),
            min_delta=float(self.cfg.min_delta),
            drift_threshold=float(self.cfg.drift_threshold),
        )
        self.meta_controller = MetaController(meta_cfg)

        # Reward model
        rm_cfg = RewardModelConfig()
        self.reward_model = SimpleRewardModel(rm_cfg)

        # Optional LLM refiner and ontology extractor
        self.llm_refiner = llm_refiner
        self.ontology_extractor = ontology_extractor

        self.budget = BudgetController(
            BudgetConfig(
                max_wall_time_s=self.cfg.max_wall_time_s,
                max_iter_time_s=self.cfg.max_iter_time_s,
                llm_max_calls=self.cfg.llm_max_calls,
                llm_max_tokens=self.cfg.llm_max_tokens,
            )
        )
        self.budget_state = BudgetState(start_time=time.time())

        # Controller
        self.bandit = None
        self.controller = None

        if self.cfg.controller == "bandit":
            self.bandit = PromptEditBandit(seed=0)
            self.bandit_path = self.out_dir / "rl" / "prompt_edit_bandit.json"
            self.bandit.load(self.bandit_path)
        else:
            # Context dimension depends on bow_dim and ContextFeatures design: 11 + 2*bow_dim
            ctx_dim = 11 + 2 * int(self.cfg.bow_dim)
            self.controller_path = self.out_dir / "rl" / "hierarchical_controller.json"
            if self.controller_path.exists():
                try:
                    self.controller = HierarchicalContextualTS.load(self.controller_path)
                except Exception:
                    self.controller = HierarchicalContextualTS(ctx_dim=ctx_dim, noise_var=float(self.cfg.controller_noise_var), seed=0)
            else:
                self.controller = HierarchicalContextualTS(ctx_dim=ctx_dim, noise_var=float(self.cfg.controller_noise_var), seed=0)

    def run(self, goal: GoalSpec, *, seed: int | None = None) -> RunResult:
        if seed is None:
            seed = int(time.time() * 1000) % (2**31 - 1)
        random.seed(int(seed))
        np.random.seed(int(seed))

        self.run_id = make_run_id(seed=int(seed), instruction=goal.instruction)

        meta = self.meta_controller.initial_state()
        best_score = -1.0
        best_iter = -1
        prev_score = 0.0
        last_context = None

        # pseudo-world-model loop placeholder (kept minimal for tests/demo)
        for it in range(int(self.cfg.max_iters)):
            # Budget checks
            if self.budget.should_stop(self.budget_state):
                break

            iter_start = time.time()

            # Dummy score curve for demo
            score = float(min(1.0, 0.4 + 0.1 * it))
            fused_scores = {"task": score}
            critic_results = [{"critic_name": "demo", "scores": fused_scores, "notes": "", "extra": {}}]

            # Belief update
            self.belief_brain.update(fused_scores)

            # Meta control update
            meta = self.meta_controller.update(meta, score=score)

            # Trace
            trace_row = {
                "iter_idx": it,
                "run_id": self.run_id,
                "prompt": f"demo-prompt-{it}",
                "artifact_path": f"artifact-{it}.mp4",
                "critic_results": critic_results,
                "fused_scores": fused_scores,
                "belief_probs": self.belief_brain.beliefs,
                "overall_score": score,
                "meta": {"goal_instruction": goal.instruction},
            }
            self.trace.write(trace_row)

            # Episode store (for replay)
            if self.episode_store is not None:
                ctx_vec = np.zeros((4,), dtype=np.float32)
                meta_x = np.zeros((6,), dtype=np.float32)
                ep = Episode(
                    ts=time.time(),
                    run_id=self.run_id,
                    choice_iter_idx=int(it),
                    ctx_vec=ctx_vec,
                    meta_x=meta_x,
                    arm="task",
                    fused_scores_next=dict(fused_scores),
                    overall_score_next=float(score),
                    delta_score=float(score - prev_score),
                    obs_reward=float(score - prev_score),
                    smoothed_reward=float(score - prev_score),
                    drift_flag_next=bool(meta.drift_flag),
                    plateau_steps_next=int(meta.plateau_steps),
                    iter_time_s_next=float(time.time() - iter_start),
                    llm_calls_total_next=int(self.budget_state.llm_calls),
                    llm_tokens_total_next=int(self.budget_state.llm_tokens_used),
                )
                with suppress(Exception):
                    self.episode_store.add(ep)
                if self.semantic_cortex is not None:
                    with suppress(Exception):
                        payload = {
                            "run_id": ep.run_id,
                            "iteration": int(ep.choice_iter_idx),
                            "arm": ep.arm,
                            "overall_score_next": float(ep.overall_score_next),
                            "delta_score": float(ep.delta_score),
                            "reward": float(ep.smoothed_reward),
                            "drift_flag_next": bool(ep.drift_flag_next),
                            "plateau_steps_next": int(ep.plateau_steps_next),
                            "critic_scores": dict(ep.fused_scores_next),
                        }
                        self.semantic_cortex.append_episode(
                            payload,
                            task_family=str(ep.run_id),
                            iteration=int(ep.choice_iter_idx),
                            provenance="tribrain",
                            external_validated=False,
                            created_at=float(ep.ts),
                        )
                        # Optional gated consolidation.
                        if int(self.cfg.semantic_consolidate_every) > 0 and int(ep.choice_iter_idx) % int(self.cfg.semantic_consolidate_every) == 0:
                            with suppress(Exception):
                                self.semantic_cortex.consolidate()

            prev_score = float(score)
            last_context = None

            if score > best_score:
                best_score = float(score)
                best_iter = int(it)

            if score >= float(self.cfg.target_score):
                break

        # Replay update at end (continual improvement)
        replay_updates = 0
        if self.replay_engine is not None and self.episode_store is not None:
            query = None
            with suppress(Exception):
                replay_updates = int(
                    self.replay_engine.replay_update(
                        controller=self.controller,
                        meta_value_model=self.meta_value_model,
                        query_ctx=query,
                    )
                )

        # Report
        report = build_report(self.out_dir)
        report_path = self.out_dir / "report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        return RunResult(
            run_id=self.run_id,
            out_dir=self.out_dir,
            best_score=float(best_score),
            best_iter=int(best_iter),
            iters=int(meta.iter_idx),
            stop_reason=str(meta.stop_reason),
        )
