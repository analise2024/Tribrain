from __future__ import annotations

import json
import shutil
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .brains.belief import BeliefState
from .brains.executive import ExecutiveRefiner
from .brains.executive_llm import ExecutiveLLMRefiner
from .brains.meta import MetaConfig, MetaController, MetaState
from .brains.meta_value import MetaValueConfig, MetaValueModel
from .budget import BudgetConfig, BudgetController, BudgetState
from .critics.base import Critic, CriticBundle
from .features.context import ContextFeatures
from .features.meta_context import meta_value_features
from .memory.episodic import Episode, EpisodeStore, make_run_id
from .memory.replay import ReplayConfig, ReplayEngine
from .ontology.compress import compress_ontology
from .ontology.extractor import OntologyExtractor
from .reward.model import BayesianRewardModel, reward_feature_vector
from .rl.bandit import PromptEditBandit
from .rl.hierarchical_contextual import HierarchicalContextualTS
from .storage.calibration import CalibrationStore
from .storage.checkpoint import CheckpointStore
from .storage.trace import JsonlTraceWriter
from .types import FailureMode, GenerationSpec, GoalSpec, WMArtifact


class WorldModel(Protocol):
    def predict(self, goal: GoalSpec, out_dir: Path, spec: GenerationSpec) -> WMArtifact:
        ...


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
    controller_prior_prec_global: float = 1.0
    controller_prior_prec_arm: float = 2.0
    # Reward model (stabilizes critic signals)
    use_reward_model: bool = True
    reward_noise_var: float = 0.05
    reward_prior_prec: float = 1.0
    # Self-healing / resilience
    resume: bool = True
    checkpoint_every_iter: bool = True



    # Continual learning via episodic replay
    use_episodic_memory: bool = True
    episodes_db: str = "episodes.sqlite"
    replay_k: int = 256
    replay_strategy: str = "uniform"  # uniform|nearest
    replay_seed: int = 0

    # Metacognitive value model (learns when continuing is worth it)
    use_meta_value_model: bool = True
    meta_value_min_updates: int = 30
    meta_value_min_expected_delta: float = 0.005
    meta_value_cost_penalty: float = 0.15
    meta_value_min_value: float = 0.0

class TriBrainOrchestrator:
    def __init__(
        self,
        world_model: WorldModel,
        critics: list[Critic],
        out_dir: Path,
        cfg: OrchestratorConfig | None = None,
        # Optional LLM refiner and ontology extractor (can be disabled).
        llm_refiner: ExecutiveLLMRefiner | None = None,
        ontology_extractor: OntologyExtractor | None = None,
    ):
        self.world_model = world_model
        self.bundle = CriticBundle(critics=critics)
        self.out_dir = out_dir
        self.cfg = cfg or OrchestratorConfig()

        meta_cfg = MetaConfig(
            target_score=self.cfg.target_score,
            patience=self.cfg.patience,
            min_delta=self.cfg.min_delta,
            max_iters=self.cfg.max_iters,
            drift_threshold=self.cfg.drift_threshold,
        )
        self.meta_controller = MetaController(meta_cfg)
        self.exec_refiner = ExecutiveRefiner()
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

        self.calibration = CalibrationStore(self.out_dir / "calibration.sqlite3")
        self.checkpoints = CheckpointStore(self.out_dir / "checkpoint.json")

        self.trace = JsonlTraceWriter(self.out_dir / "trace.jsonl")
        # Run identity (set at run() time once the seed is known)
        self.run_id = ""

        # Episodic memory + replay (continual improvement without re-running expensive generations)
        self.episode_store: EpisodeStore | None = None
        self.replay_engine: ReplayEngine | None = None
        if self.cfg.use_episodic_memory:
            self.episode_store = EpisodeStore(self.out_dir / self.cfg.episodes_db)
            self.replay_engine = ReplayEngine(
                self.episode_store,
                ReplayConfig(k=int(self.cfg.replay_k), strategy=str(self.cfg.replay_strategy), seed=int(self.cfg.replay_seed)),
            )

        # Metacognitive value model: learns marginal value of "one more iteration"
        self.meta_value_model: MetaValueModel | None = None
        self.meta_value_path = self.out_dir / "meta_value_model.json"
        if self.cfg.use_meta_value_model:
            ctx_dim = int(11 + 2 * int(self.cfg.bow_dim))  # ContextFeatures = 7 nums + 4 scores + 2*bow_dim
            meta_dim = int(ctx_dim + 12)  # + meta extras + bias
            mv_cfg = MetaValueConfig(
                dim=meta_dim,
                noise_var=float(self.cfg.controller_noise_var),
                prior_prec=float(self.cfg.controller_prior_prec_global),
                min_expected_delta=float(self.cfg.meta_value_min_expected_delta),
                cost_penalty=float(self.cfg.meta_value_cost_penalty),
                min_value=float(self.cfg.meta_value_min_value),
            )
            if self.meta_value_path.exists():
                try:
                    self.meta_value_model = MetaValueModel.load(self.meta_value_path)
                except Exception:
                    self.meta_value_model = MetaValueModel(cfg=mv_cfg)
            else:
                self.meta_value_model = MetaValueModel(cfg=mv_cfg)

        # RL: learn which refinement focus works for this world model/task distribution.
        # RL / controller state (learns which refinement focus saves compute and improves score).
        self.arm_names = ["physics", "task", "smoothness", "quality", "minimal"]

        # Controller persistence
        (self.out_dir / "rl").mkdir(parents=True, exist_ok=True)
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
                    self.controller = None
            if self.controller is None:
                self.controller = HierarchicalContextualTS(
                    dim=ctx_dim,
                    arm_names=self.arm_names,
                    noise_var=float(self.cfg.controller_noise_var),
                    prior_prec_global=float(self.cfg.controller_prior_prec_global),
                    prior_prec_arm=float(self.cfg.controller_prior_prec_arm),
                    seed=0,
                )

        # Reward model persistence (stabilizes noisy critic signals)
        (self.out_dir / "reward").mkdir(parents=True, exist_ok=True)
        self.reward_path = self.out_dir / "reward" / "reward_model.json"
        self.reward_updates = 0
        self.reward_model = None
        if self.cfg.use_reward_model:
            if self.reward_path.exists():
                try:
                    self.reward_model = BayesianRewardModel.load(self.reward_path)
                except Exception:
                    self.reward_model = None
            if self.reward_model is None:
                self.reward_model = BayesianRewardModel(dim=8, noise_var=float(self.cfg.reward_noise_var), prior_prec=float(self.cfg.reward_prior_prec))

    def run(self, goal: GoalSpec, spec: GenerationSpec | None = None) -> dict[str, Any]:
        spec = spec or GenerationSpec()
        seed = int(spec.seed) if spec.seed is not None else 0
        self.run_id = make_run_id(seed=seed, instruction=goal.instruction)

        # Determinism: seed all internal stochastic components.
        if self.controller is not None:
            with suppress(Exception):
                self.controller.set_seed(seed)
        if self.bandit is not None:
            with suppress(Exception):
                self.bandit.set_seed(seed)
        belief = BeliefState()
        meta = MetaState()
        prompt = goal.instruction

        # Resume if possible
        if self.cfg.resume:
            ckpt = self.checkpoints.load()
            if ckpt is not None:
                prompt = ckpt.get("prompt", prompt)
                meta = MetaState(**ckpt.get("meta", {}))
                belief.load_dict(ckpt.get("belief", {}))
                self.budget_state = BudgetState.from_dict(ckpt.get("budget", {}), start_time=time.time())

        # Pull critic reliability from calibration DB
        belief.sync_reliability_from_store(self.calibration)

        # Ontology extraction once per goal (cheap to cache; huge token saver later)
        ontology_obj = None
        ontology_text = ""
        if self.ontology_extractor is not None:
            try:
                ontology_obj = self.ontology_extractor.extract(goal.instruction)
                ontology_text = compress_ontology(ontology_obj)
            except Exception:
                ontology_obj = None
                ontology_text = ""

        best_artifact: WMArtifact | None = None
        best_score = -1.0
        prev_score: float | None = None
        last_arm: str | None = None
        last_context: ContextFeatures | None = None
        last_meta_x: np.ndarray | None = None
        last_choice_iter_idx: int | None = None

        for i in range(self.cfg.max_iters):
            if self.budget.should_stop(self.budget_state):
                meta.stop_reason = "budget_exhausted"
                break

            meta.iter_idx = i
            iter_dir = self.out_dir / f"iter_{i:02d}"
            iter_start = time.time()

            # Inject ontology (compact) into the instruction to stabilize planning while keeping tokens low.
            wm_instruction = prompt
            if ontology_text:
                wm_instruction = f"{prompt}\n\n[Ontology]\n{ontology_text}".strip()

            try:
                artifact = self.world_model.predict(
                    GoalSpec(instruction=wm_instruction, tags=goal.tags),
                    iter_dir,
                    spec,
                )
            except Exception as e:
                # Real-world behavior: stop gracefully, write a trace record + checkpoint.
                fused = {m: 0.0 for m in FailureMode}
                probs = belief.modes.probs()
                meta.stop_reason = f"wm_error:{type(e).__name__}"

                self.trace.append(
                    iter_idx=i,
                    prompt=wm_instruction,
                    artifact_path="",
                    critic_results=[],
                    fused_scores=fused,
                    belief_probs=probs,
                    overall_score=0.0,
                    meta={
                        "iter_idx": i,
                        "run_id": self.run_id,
                        "seed": seed,
                        "goal_instruction": goal.instruction,
                        "stop_reason": meta.stop_reason,
                        "error": str(e)[:2000],
                    },
                )
                if self.cfg.checkpoint_every_iter:
                    self.checkpoints.save(
                        {
                            "prompt": prompt,
                            "meta": meta.__dict__,
                            "belief": belief.to_dict(),
                            "budget": self.budget_state.to_dict(),
                        }
                    )
                break

            # Validate artifact existence (self-healing). Some generators can "succeed" but not emit output.
            if not getattr(artifact, "path", None) or not Path(artifact.path).exists():
                fused = {m: 0.0 for m in FailureMode}
                probs = belief.modes.probs()
                meta.stop_reason = "missing_artifact"
                self.trace.append(
                    iter_idx=i,
                    prompt=wm_instruction,
                    artifact_path="",
                    critic_results=[],
                    fused_scores=fused,
                    belief_probs=probs,
                    overall_score=0.0,
                    meta={
                        "iter_idx": i,
                        "run_id": self.run_id,
                        "seed": seed,
                        "stop_reason": meta.stop_reason,
                        "goal_instruction": goal.instruction,
                    },
                )
                if self.cfg.checkpoint_every_iter:
                    self.checkpoints.save(
                        {
                            "prompt": prompt,
                            "meta": meta.__dict__,
                            "belief": belief.to_dict(),
                            "budget": self.budget_state.to_dict(),
                        }
                    )
                break

            results = self.bundle.evaluate_all(artifact, goal)
            fused = belief.fuse(results)
            belief.update(results, fused)
            probs = belief.modes.probs()

            score = self.meta_controller.update(meta, fused)

            # RL update (credit assignment): the improvement observed at this iteration
            # is attributed to the refinement action chosen at the end of the previous iteration.
            if prev_score is not None and last_arm is not None and last_context is not None:
                delta = float(score - prev_score)
                # Observation reward: bounded, stable, and symmetric.
                obs_reward = float(np.tanh(delta / 0.05))

                # Reward model smoothing (optional)
                rm_reward = obs_reward
                if self.reward_model is not None:
                    x_rm = reward_feature_vector(
                        fused_scores=fused,
                        prev_score=float(prev_score),
                        prompt_len=len(prompt.split()),
                        drift_flag=bool(meta.drift_flag),
                    )
                    if self.reward_updates >= 3:
                        mu, var = self.reward_model.predict(x_rm)
                        # trust prediction more when uncertainty is low
                        mix = float(max(0.0, min(0.8, 0.2 + 0.6 * (1.0 / (1.0 + var)))))
                        rm_reward = float((1.0 - mix) * obs_reward + mix * mu)
                    # Update after computing smoothed reward
                    self.reward_model.update(x_rm, obs_reward)
                    self.reward_updates += 1
                    with suppress(Exception):
                        self.reward_model.save(self.reward_path)

                # Update controller/bandit with the (smoothed) reward
                if self.controller is not None:
                    self.controller.update(last_arm, last_context.vector, rm_reward)
                    with suppress(Exception):
                        self.controller.save(self.controller_path)
                elif self.bandit is not None:
                    # Map [-1,1] -> [0,1]
                    r01 = float(max(0.0, min(1.0, 0.5 + 0.5 * rm_reward)))
                    self.bandit.update(r01)
                    self.bandit.save(self.bandit_path)


                # Metacognitive value model update: learn expected *raw delta score* for stop decisions.
                if self.meta_value_model is not None and last_meta_x is not None and last_choice_iter_idx is not None:
                    with suppress(Exception):
                        self.meta_value_model.update(last_meta_x, float(delta))
                        if (self.meta_value_model.updates % 10) == 0:
                            self.meta_value_model.save(self.meta_value_path)

                # Episodic memory write (continual improvement via replay without re-running generations)
                if self.episode_store is not None and last_choice_iter_idx is not None and last_meta_x is not None:
                    iter_time_obs = float(max(0.0, time.time() - iter_start))
                    ep = Episode(
                        ts=time.time(),
                        run_id=self.run_id,
                        choice_iter_idx=int(last_choice_iter_idx),
                        ctx_vec=last_context.vector,
                        meta_x=last_meta_x,
                        arm=str(last_arm),
                        fused_scores_next={str(k): float(v) for k, v in fused.items()},
                        overall_score_next=float(score),
                        delta_score=float(delta),
                        obs_reward=float(obs_reward),
                        smoothed_reward=float(rm_reward),
                        drift_flag_next=bool(meta.drift_flag),
                        plateau_steps_next=int(meta.plateau_steps),
                        iter_time_s_next=iter_time_obs,
                        llm_calls_total_next=int(self.budget_state.llm_calls),
                        llm_tokens_total_next=int(self.budget_state.llm_tokens_used),
                    )
                    with suppress(Exception):
                        self.episode_store.add(ep)

            prev_score = float(score)
            meta_dict = {
                "iter_idx": meta.iter_idx,
                "goal_instruction": goal.instruction,
                "best_score": meta.best_score,
                "best_iter": meta.best_iter,
                "plateau_steps": meta.plateau_steps,
                "drift_flag": meta.drift_flag,
                "stop_reason": meta.stop_reason,
            }

            if score > best_score:
                best_score = score
                best_artifact = artifact
                self._snapshot_best(artifact, i, score, prompt)

            # Decide whether another iteration is worth it (metacognition).
            # This combines:
            # - classical stop conditions (target/plateau/drift) from MetaController
            # - a learned marginal-value model (expected next-iter improvement vs cost)
            stop_now = bool(self.meta_controller.should_stop(meta))
            arm = ""
            prompt_next = prompt
            ctx: ContextFeatures | None = None
            mx: np.ndarray | None = None
            meta_value_mu: float | None = None
            meta_value_var: float | None = None
            meta_value_value: float | None = None

            if not stop_now:
                # Contextual controller chooses *where to intervene* next (saves tokens/compute)
                ctx = ContextFeatures.from_state(
                    instruction=prompt,
                    ontology=ontology_obj,
                    compressed_ontology=ontology_text,
                    last_scores=fused,
                    drift_flag=bool(meta.drift_flag),
                    plateau_count=int(meta.plateau_steps),
                    bow_dim=int(self.cfg.bow_dim),
                )
                if self.controller is not None:
                    arm = self.controller.choose(ctx.vector)
                elif self.bandit is not None:
                    arm = self.bandit.choose()
                else:
                    arm = "task"

                # Remaining budget fractions (bounded to [0,1]) for energy-aware metacognition
                elapsed = float(max(0.0, time.time() - self.budget_state.start_time))
                rem_wall = float(max(0.0, self.cfg.max_wall_time_s - elapsed))
                rem_wall_frac = float(rem_wall / max(1e-6, self.cfg.max_wall_time_s))
                rem_calls = float(max(0.0, self.cfg.llm_max_calls - self.budget_state.llm_calls))
                rem_calls_frac = float(rem_calls / max(1.0, float(self.cfg.llm_max_calls)))
                rem_toks = float(max(0.0, self.cfg.llm_max_tokens - self.budget_state.llm_tokens_used))
                rem_toks_frac = float(rem_toks / max(1.0, float(self.cfg.llm_max_tokens)))

                mx = meta_value_features(
                    ctx_vec=ctx.vector,
                    fused_scores=fused,
                    overall_score=float(score),
                    iter_idx=int(i),
                    plateau_steps=int(meta.plateau_steps),
                    drift_flag=bool(meta.drift_flag),
                    remaining_wall_frac=rem_wall_frac,
                    remaining_llm_calls_frac=rem_calls_frac,
                    remaining_llm_tokens_frac=rem_toks_frac,
                )

                # Learned marginal value: if we expect negligible improvement, stop and save compute.
                if self.meta_value_model is not None and int(self.meta_value_model.updates) >= int(self.cfg.meta_value_min_updates):
                    with suppress(Exception):
                        meta_value_mu, meta_value_var = self.meta_value_model.predict(mx)
                        # Cost proxy for "one more iteration": time so far in this iter relative to allowed per-iter budget.
                        cost_est = float(
                            min(1.0, max(0.0, (time.time() - iter_start) / max(1e-6, self.cfg.max_iter_time_s)))
                        )
                        meta_value_value = float(meta_value_mu - float(self.cfg.meta_value_cost_penalty) * cost_est)
                        if float(meta_value_mu) < float(self.cfg.meta_value_min_expected_delta) or meta_value_value < float(self.cfg.meta_value_min_value):
                            meta.stop_reason = "low_marginal_value"
                            stop_now = True

                if not stop_now:
                    # Store for next-iteration credit assignment
                    last_arm = arm
                    last_context = ctx
                    last_meta_x = mx
                    last_choice_iter_idx = int(i)

                    preferred = {
                        "physics": [FailureMode.PHYSICS],
                        "task": [FailureMode.TASK],
                        "smoothness": [FailureMode.SMOOTHNESS],
                        "quality": [FailureMode.QUALITY],
                        "minimal": [],
                    }.get(arm, [])

                    # Cheap deterministic edits always.
                    prompt_next = self.exec_refiner.refine(
                        prompt=prompt,
                        belief_probs=probs,
                        fused_scores=fused,
                        drift_flag=meta.drift_flag,
                        preferred_modes=preferred,
                    )

                    # Decide if the LLM should intervene this iteration (immediate + bounded).
                    if self.cfg.llm_policy == "never":
                        llm_trigger = False
                    elif self.cfg.llm_policy == "always":
                        llm_trigger = True
                    else:
                        llm_trigger = bool(meta.drift_flag) or int(meta.plateau_steps) >= 1 or float(score) < (float(self.cfg.target_score) - 0.15)

                    if self.llm_refiner is not None and llm_trigger and self.budget.allow_llm_call(self.budget_state):
                        with suppress(Exception):
                            prompt_next = self.llm_refiner.refine(
                                prompt=prompt_next,
                                ontology_hint=ontology_text,
                                belief_probs=probs,
                                fused_scores=fused,
                                drift_flag=meta.drift_flag,
                                budget=self.budget,
                                budget_state=self.budget_state,
                            )

            prompt = prompt_next

            # Update iteration time budget
            iter_time = float(time.time() - iter_start)
            self.budget.note_iter_time(self.budget_state, float(iter_time))
            if self.budget.iter_time_exceeded(iter_time):
                meta.stop_reason = "iter_time_exceeded"
                stop_now = True

            # Save checkpoint for self-healing/resume
            if self.cfg.checkpoint_every_iter:
                self.checkpoints.save(
                    {
                        "prompt": prompt,
                        "meta": meta.__dict__,
                        "belief": belief.to_dict(),
                        "budget": self.budget_state.to_dict(),
                    }
                )


            # Trace: record generation + refinement decisions + budgets for auditability and continual learning.
            meta_dict_full = dict(meta_dict)
            meta_dict_full.update(
                {
                    "run_id": self.run_id,
                    "seed": seed,
                    "wm_instruction": wm_instruction,
                    "chosen_arm": arm,
                    "next_prompt": prompt,
                    "meta_value_mu": meta_value_mu,
                    "meta_value_var": meta_value_var,
                    "meta_value_value": meta_value_value,
                    "budget": self.budget_state.to_dict(),
                    "iter_time_s": float(iter_time),
                    "controller": self.cfg.controller,
                    "llm_policy": self.cfg.llm_policy,
                    "stop_now": bool(stop_now),
                }
            )
            self.trace.append(
                iter_idx=i,
                prompt=wm_instruction,
                artifact_path=artifact.path,
                critic_results=results,
                fused_scores=fused,
                belief_probs=probs,
                overall_score=score,
                meta=meta_dict_full,
            )

            if stop_now:
                break

        # Continual improvement: replay a sample of stored episodes to refine controller & meta-value model.
        replay_updates = 0
        if self.replay_engine is not None and self.episode_store is not None:
            query = last_context.vector if last_context is not None else None
            with suppress(Exception):
                replay_updates = int(
                    self.replay_engine.replay_update(
                        controller=self.controller,
                        meta_value_model=self.meta_value_model,
                        query_ctx=query,
                    )
                )
                if self.controller is not None:
                    self.controller.save(self.controller_path)
                if self.meta_value_model is not None:
                    self.meta_value_model.save(self.meta_value_path)


        return {
            "best_score": best_score,
            "best_artifact": str(best_artifact.path) if best_artifact else "",
            "stop_reason": meta.stop_reason,
            "out_dir": str(self.out_dir),
            "replay_updates": int(replay_updates),
        }

    def _snapshot_best(self, artifact: WMArtifact, iter_idx: int, score: float, prompt: str) -> None:
        best_dir = self.out_dir / "best"
        best_dir.mkdir(parents=True, exist_ok=True)
        # Copy best video to a stable path
        dst = best_dir / artifact.path.name
        with suppress(Exception):
            shutil.copy2(artifact.path, dst)
        meta = {
            "iter_idx": iter_idx,
            "score": score,
            "prompt": prompt,
            "artifact": str(artifact.path),
        }
        (best_dir / "best.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
