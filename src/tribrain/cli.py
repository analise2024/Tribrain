"""TriBrain CLI.

Design goals:
  - stable (argparse)
  - CPU-first
  - graceful degradation for optional features

Acceptance-critical commands (tests rely on these):
  - export-preference-pairs
  - replay-update

Additional production commands:
  - run-subprocess / run-wow
  - export-dataset
  - train-preference-rm (Bradley–Terry)
  - train-lora-rm (optional deps)
  - dashboard
  - calibrate-critic

Run:
  python -m tribrain.cli --help
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, Literal, cast

from tribrain.continual.dataset import export_preference_dataset
from tribrain.continual.preference_dataset import export_preference_pairs_jsonl
from tribrain.critics.video_metrics import FlickerCritic, FlowSmoothnessCritic, TemporalJitterCritic
from tribrain.dashboard.report import build_dashboard
from tribrain.integrations.subprocess_world_model import SubprocessWorldModel
from tribrain.integrations.wow import WowWorldModel
from tribrain.memory.episodic import EpisodeStore
from tribrain.memory.replay import ReplayConfig, ReplayEngine
from tribrain.ontology.extractor import OntologyExtractor
from tribrain.orchestrator import OrchestratorConfig, TriBrainOrchestrator
from tribrain.storage.calibration import CalibrationStore
from tribrain.trainers.bt_reward_model import BTTrainConfig, train_bt_text_reward_model
from tribrain.types import GenerationSpec, GoalSpec
from tribrain.utils.seeding import set_global_seed


def _p(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _cmd_export_preference_pairs(ns: argparse.Namespace) -> int:
    run_dir = _p(ns.run_dir)
    trace_path = _p(ns.trace) if ns.trace else (run_dir / "trace.jsonl")
    out_path = _p(ns.out_path)
    n = export_preference_pairs_jsonl(
        trace_path,
        out_path,
        min_gap=float(ns.min_gap),
        top_k=int(ns.top_k),
        strategy=cast(Literal["best_vs_rest", "adjacent"], str(ns.strategy)),
    )
    sys.stdout.write(json.dumps({"pairs_exported": int(n), "out_path": str(out_path)}) + "\n")
    return 0


def _load_optional_models(run_dir: Path) -> tuple[Any | None, Any | None]:
    """Load controller + meta-value model if available.

    We keep this optional so replay-update can run with just episodes.sqlite.
    """

    controller = None
    meta_value_model = None

    controller_path = run_dir / "rl" / "hierarchical_controller.json"
    if controller_path.exists():
        try:
            from tribrain.rl.hierarchical_contextual import HierarchicalContextualTS

            controller = HierarchicalContextualTS.load(controller_path)
        except Exception:
            controller = None

    mv_path = run_dir / "meta_value_model.json"
    if mv_path.exists():
        try:
            from tribrain.brains.meta_value import MetaValueModel

            meta_value_model = MetaValueModel.load(mv_path)
        except Exception:
            meta_value_model = None

    return controller, meta_value_model


def _save_optional_models(run_dir: Path, controller: Any | None, meta_value_model: Any | None) -> None:
    with suppress(Exception):
        if controller is not None:
            (run_dir / "rl").mkdir(parents=True, exist_ok=True)
            controller.save(run_dir / "rl" / "hierarchical_controller.json")
    with suppress(Exception):
        if meta_value_model is not None:
            meta_value_model.save(run_dir / "meta_value_model.json")


def _cmd_replay_update(ns: argparse.Namespace) -> int:
    run_dir = _p(ns.run_dir)
    db_path = run_dir / str(ns.episodes_db)
    store = EpisodeStore(db_path, backup=False)
    cfg = ReplayConfig(k=int(ns.k), strategy=str(ns.strategy), seed=int(ns.seed))
    engine = ReplayEngine(store, cfg)

    controller, meta_value_model = _load_optional_models(run_dir)

    updated = int(engine.replay_update(controller=controller, meta_value_model=meta_value_model))
    _save_optional_models(run_dir, controller, meta_value_model)
    sys.stdout.write(json.dumps({"episodes_updated": updated, "episodes_db": str(db_path)}) + "\n")
    return 0


def _build_base_critics(ns: argparse.Namespace) -> list[Any]:
    critics: list[Any] = []
    if not getattr(ns, "no_video_critics", False):
        critics.extend([FlickerCritic(), FlowSmoothnessCritic(), TemporalJitterCritic()])

    if getattr(ns, "use_vlm_critic", False):
        try:
            from tribrain.critics.vlm_critic import HFVlmCritic

            critics.append(
                HFVlmCritic(
                    model_name=str(getattr(ns, "vlm_model", "Qwen/Qwen2-VL-2B-Instruct")),
                    device=str(getattr(ns, "vlm_device", "auto")),
                    max_new_tokens=int(getattr(ns, "vlm_max_new_tokens", 256)),
                )
            )
        except Exception as e:
            sys.stderr.write(f"[tribrain] VLM critic not available: {e}\n")
    return critics


def _make_llm_refiner(ns: argparse.Namespace, run_dir: Path) -> Any | None:
    backend = str(getattr(ns, "llm_backend", "none"))
    if backend == "none":
        return None
    if backend != "ollama":
        raise ValueError(f"Unknown llm backend: {backend}")

    try:
        from tribrain.brains.executive_llm import ExecutiveLLMRefiner, LLMRefinerConfig
        from tribrain.llm.ollama import OllamaClient
    except Exception as e:
        raise RuntimeError("LLM backend requested but deps are missing.") from e

    cache_dir = Path(getattr(ns, "llm_cache", "") or (run_dir / "llm_cache")).resolve()
    cfg = LLMRefinerConfig(
        cache_path=cache_dir,
        max_tokens=int(getattr(ns, "llm_call_tokens", 350)),
        temperature=float(getattr(ns, "llm_temperature", 0.0)),
        timeout_s=float(getattr(ns, "llm_timeout", 20.0)),
    )
    client = OllamaClient(
        model=str(getattr(ns, "llm_model", "llama3.1:8b-instruct")),
        base_url=str(getattr(ns, "llm_url", "http://localhost:11434")),
    )
    return ExecutiveLLMRefiner(client, cfg)


def _run_loop(world_model: Any, ns: argparse.Namespace) -> int:
    run_dir = Path(getattr(ns, "out", "outputs/run"))
    run_dir = run_dir.expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = int(getattr(ns, "seed", 0))
    set_global_seed(seed)

    llm_refiner = None
    ontology = None
    try:
        llm_refiner = _make_llm_refiner(ns, run_dir)
    except Exception as e:
        sys.stderr.write(f"[tribrain] LLM disabled: {e}\n")
        llm_refiner = None

    if bool(getattr(ns, "use_ontology", False)):
        # Optionally let the ontology extractor use the same LLM (if available).
        ontology = OntologyExtractor(llm=getattr(llm_refiner, "llm", None) if llm_refiner else None)

    cfg = OrchestratorConfig(
        max_iters=int(getattr(ns, "iters", 4)),
        target_score=float(getattr(ns, "target_score", 0.85)),
        patience=int(getattr(ns, "patience", 2)),
        min_delta=float(getattr(ns, "min_delta", 0.01)),
        drift_threshold=float(getattr(ns, "drift_threshold", 0.15)),
        max_wall_time_s=float(getattr(ns, "max_wall", 20 * 60)),
        max_iter_time_s=float(getattr(ns, "max_iter", 10 * 60)),
        llm_max_calls=int(getattr(ns, "llm_max_calls", 4)),
        llm_max_tokens=int(getattr(ns, "llm_max_tokens", 1200)),
        llm_policy=str(getattr(ns, "llm_policy", "adaptive")),
        controller=str(getattr(ns, "controller", "hierarchical")),
        bow_dim=int(getattr(ns, "bow_dim", 32)),
        use_reward_model=not bool(getattr(ns, "no_reward_model", False)),
        resume=not bool(getattr(ns, "no_resume", False)),
        checkpoint_every_iter=not bool(getattr(ns, "no_checkpoint", False)),
        use_episodic_memory=not bool(getattr(ns, "no_episodic", False)),
        episodes_db=str(getattr(ns, "episodes_db", "episodes.sqlite")),
        replay_k=int(getattr(ns, "replay_k", 256)),
        replay_strategy=str(getattr(ns, "replay_strategy", "uniform")),
        replay_seed=int(getattr(ns, "replay_seed", 0)),
        use_meta_value_model=not bool(getattr(ns, "no_meta_value", False)),
    )

    critics = _build_base_critics(ns)
    orch = TriBrainOrchestrator(
        world_model=world_model,
        critics=critics,
        out_dir=run_dir,
        cfg=cfg,
        llm_refiner=llm_refiner,
        ontology_extractor=ontology,
    )

    goal = GoalSpec(instruction=str(getattr(ns, "prompt", "")))
    spec = GenerationSpec(seed=seed)

    result = orch.run(goal, spec)
    (run_dir / "run_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with suppress(Exception):
        build_dashboard(run_dir)

    sys.stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
    stop_reason = str(result.get("stop_reason", ""))
    return 2 if stop_reason.startswith("wm_error") else 0


def _cmd_run_subprocess(ns: argparse.Namespace) -> int:
    wm = SubprocessWorldModel(cmd_template=str(ns.cmd))
    return _run_loop(wm, ns)


def _cmd_run_wow(ns: argparse.Namespace) -> int:
    wm = WowWorldModel(wow_repo=Path(ns.wow_repo), model=str(ns.model), extra_args=str(ns.wow_extra_args))
    return _run_loop(wm, ns)


def _cmd_export_dataset(ns: argparse.Namespace) -> int:
    run_dir = _p(ns.run_dir)
    out_path = _p(ns.out_path)
    n = int(export_preference_dataset(run_dir, out_path, min_score=float(ns.min_score)))
    sys.stdout.write(json.dumps({"examples_exported": n, "out_path": str(out_path)}) + "\n")
    return 0


def _cmd_dashboard(ns: argparse.Namespace) -> int:
    run_dir = _p(ns.run_dir)
    out_dir = _p(ns.out_dir) if ns.out_dir else None
    md = build_dashboard(run_dir, out_dir=out_dir)
    sys.stdout.write(json.dumps({"report": str(md)}) + "\n")
    return 0


def _cmd_calibrate_critic(ns: argparse.Namespace) -> int:
    run_dir = _p(ns.run_dir)
    store = CalibrationStore(run_dir / "calibration.sqlite3")
    tags: dict[str, str] = {}
    for kv in (ns.tag or []):
        if "=" in kv:
            k, v = kv.split("=", 1)
            tags[k.strip()] = v.strip()
    store.add_observation(
        critic_name=str(ns.critic),
        success=bool(int(ns.success)),
        weight=float(ns.weight),
        tags=tags,
    )
    rel = store.reliability(str(ns.critic))
    sys.stdout.write(
        json.dumps({"critic": str(ns.critic), "reliability_mean": float(rel.mean())}, ensure_ascii=False) + "\n"
    )
    return 0


def _cmd_train_preference_rm(ns: argparse.Namespace) -> int:
    cfg = BTTrainConfig(
        bow_dim=int(ns.bow_dim),
        epochs=int(ns.epochs),
        lr=float(ns.lr),
        l2=float(ns.l2),
        seed=int(ns.seed),
    )
    out = train_bt_text_reward_model(_p(ns.pairs_jsonl), _p(ns.out_path), cfg)
    sys.stdout.write(json.dumps({"model": str(out)}) + "\n")
    return 0


def _cmd_train_lora_rm(ns: argparse.Namespace) -> int:
    try:
        from tribrain.trainers.lora_reward_model import LoRATrainConfig, train_text_reward_model_lora  # noqa: I001
    except Exception as e:
        sys.stderr.write(
            "[tribrain] LoRA trainer unavailable. Install extras: pip install -e '.[lora]'\n"
            f"Reason: {e}\n"
        )
        return 2

    cfg = LoRATrainConfig(
        base_model=str(ns.base_model),
        output_dir=str(ns.out_dir),
        epochs=int(ns.epochs),
        batch_size=int(ns.batch_size),
        grad_accum=int(ns.grad_accum),
        lr=float(ns.lr),
        max_length=int(ns.max_length),
    )
    out = train_text_reward_model_lora(_p(ns.pairs_jsonl), cfg)
    sys.stdout.write(json.dumps({"out_dir": str(out)}) + "\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tribrain",
        description="TriBrain 3-Brain world models controller (Executive + Bayesian + Metacognitive)",
    )
    sub = parser.add_subparsers(dest="command")

    p_pairs = sub.add_parser(
        "export-preference-pairs",
        help="Export pairwise preferences from a run trace.jsonl",
    )
    p_pairs.add_argument("--run-dir", required=True, help="Run directory containing trace.jsonl")
    p_pairs.add_argument(
        "--trace",
        default=None,
        help="Optional explicit path to trace JSONL (defaults to RUN_DIR/trace.jsonl)",
    )
    p_pairs.add_argument("--out-path", "--out", dest="out_path", required=True, help="Output JSONL path")
    p_pairs.add_argument("--min-gap", type=float, default=0.03)
    p_pairs.add_argument("--top-k", type=int, default=1)
    p_pairs.add_argument("--strategy", choices=["best_vs_rest", "adjacent"], default="best_vs_rest")
    p_pairs.set_defaults(_handler=_cmd_export_preference_pairs)

    p_replay = sub.add_parser(
        "replay-update",
        help="Replay-update controller/meta-value model from episodic memory",
    )
    p_replay.add_argument("--run-dir", required=True)
    p_replay.add_argument("--episodes-db", default="episodes.sqlite")
    p_replay.add_argument("--k", type=int, default=256)
    p_replay.add_argument("--strategy", choices=["uniform", "nearest"], default="uniform")
    p_replay.add_argument("--seed", type=int, default=0)
    p_replay.set_defaults(_handler=_cmd_replay_update)

    # --------------------- run commands ---------------------
    def add_run_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--prompt", required=True, help="Goal instruction")
        p.add_argument("--out", required=True, help="Run output directory")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--iters", type=int, default=4)
        p.add_argument("--target-score", type=float, default=0.85)
        p.add_argument("--patience", type=int, default=2)
        p.add_argument("--min-delta", type=float, default=0.01)
        p.add_argument("--drift-threshold", type=float, default=0.15)

        # Hard cost budgets
        p.add_argument("--max-wall", type=float, default=20 * 60, help="Max wall time (s)")
        p.add_argument("--max-iter", type=float, default=10 * 60, help="Max per-iter time (s)")
        p.add_argument("--llm-max-calls", type=int, default=4)
        p.add_argument("--llm-max-tokens", type=int, default=1200)

        p.add_argument("--llm-policy", choices=["adaptive", "always", "never"], default="adaptive")
        p.add_argument("--controller", choices=["hierarchical", "bandit"], default="hierarchical")
        p.add_argument("--bow-dim", type=int, default=32)
        p.add_argument("--no-reward-model", action="store_true")

        # Resilience
        p.add_argument("--no-resume", action="store_true")
        p.add_argument("--no-checkpoint", action="store_true")

        # Continual learning / replay
        p.add_argument("--no-episodic", action="store_true")
        p.add_argument("--episodes-db", default="episodes.sqlite")
        p.add_argument("--replay-k", type=int, default=256)
        p.add_argument("--replay-strategy", choices=["uniform", "nearest"], default="uniform")
        p.add_argument("--replay-seed", type=int, default=0)
        p.add_argument("--no-meta-value", action="store_true")

        # Critics
        p.add_argument("--no-video-critics", action="store_true")
        p.add_argument("--use-vlm-critic", action="store_true")
        p.add_argument("--vlm-model", default="Qwen/Qwen2-VL-2B-Instruct")
        p.add_argument("--vlm-device", choices=["auto", "cpu", "cuda"], default="auto")
        p.add_argument("--vlm-max-new-tokens", type=int, default=256)

        # Ontology extraction
        p.add_argument("--use-ontology", action="store_true")

        # LLM refiner (optional)
        p.add_argument("--llm-backend", choices=["none", "ollama"], default="none")
        p.add_argument("--llm-model", default="llama3.1:8b-instruct")
        p.add_argument("--llm-url", default="http://localhost:11434")
        p.add_argument("--llm-cache", default=None)
        p.add_argument("--llm-call-tokens", type=int, default=350)
        p.add_argument("--llm-temperature", type=float, default=0.0)
        p.add_argument("--llm-timeout", type=float, default=20.0)

    p_run_sub = sub.add_parser(
        "run-subprocess",
        help="Run TriBrain over an external world model command (no code changes to the world model)",
    )
    p_run_sub.add_argument(
        "--cmd",
        required=True,
        help="Command template with placeholders {prompt}, {out_dir}, {seed}",
    )
    add_run_common(p_run_sub)
    p_run_sub.set_defaults(_handler=_cmd_run_subprocess)

    p_run_wow = sub.add_parser(
        "run-wow",
        help="Run TriBrain via the WoW adapter (subprocess)",
    )
    p_run_wow.add_argument("--wow-repo", required=True, help="Path to WoW repo")
    p_run_wow.add_argument("--model", default="wow-mini")
    p_run_wow.add_argument("--wow-extra-args", default="")
    add_run_common(p_run_wow)
    p_run_wow.set_defaults(_handler=_cmd_run_wow)

    # --------------------- offline tooling ---------------------
    p_ds = sub.add_parser("export-dataset", help="Export a JSONL dataset of high-scoring iterations")
    p_ds.add_argument("--run-dir", required=True)
    p_ds.add_argument("--out-path", required=True)
    p_ds.add_argument("--min-score", type=float, default=0.7)
    p_ds.set_defaults(_handler=_cmd_export_dataset)

    p_dash = sub.add_parser("dashboard", help="Build a Markdown report for a run")
    p_dash.add_argument("--run-dir", required=True)
    p_dash.add_argument("--out-dir", default=None)
    p_dash.set_defaults(_handler=_cmd_dashboard)

    p_cal = sub.add_parser("calibrate-critic", help="Add a critic calibration observation")
    p_cal.add_argument("--run-dir", "--out", dest="run_dir", required=True)
    p_cal.add_argument("--critic", required=True)
    p_cal.add_argument("--success", choices=["0", "1"], required=True)
    p_cal.add_argument("--weight", type=float, default=1.0)
    p_cal.add_argument("--tag", action="append", default=[], help="Key=Value tags (repeatable)")
    p_cal.set_defaults(_handler=_cmd_calibrate_critic)

    p_bt = sub.add_parser("train-preference-rm", help="Train a lightweight Bradley–Terry reward model (CPU)")
    p_bt.add_argument("--pairs-jsonl", "--pairs", dest="pairs_jsonl", required=True)
    p_bt.add_argument("--out-path", "--out", dest="out_path", required=True)
    p_bt.add_argument("--bow-dim", type=int, default=64)
    p_bt.add_argument("--epochs", type=int, default=3)
    p_bt.add_argument("--lr", type=float, default=0.12)
    p_bt.add_argument("--l2", type=float, default=1e-3)
    p_bt.add_argument("--seed", type=int, default=0)
    p_bt.set_defaults(_handler=_cmd_train_preference_rm)

    p_lora = sub.add_parser("train-lora-rm", help="Train a LoRA text reward model (optional deps)")
    p_lora.add_argument("--pairs-jsonl", "--pairs", dest="pairs_jsonl", required=True)
    p_lora.add_argument("--base-model", default="distilbert-base-uncased")
    p_lora.add_argument("--out-dir", default="out_lora_reward_model")
    p_lora.add_argument("--epochs", type=int, default=1)
    p_lora.add_argument("--batch-size", type=int, default=8)
    p_lora.add_argument("--grad-accum", type=int, default=2)
    p_lora.add_argument("--lr", type=float, default=2e-5)
    p_lora.add_argument("--max-length", type=int, default=384)
    p_lora.set_defaults(_handler=_cmd_train_lora_rm)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)
    if not getattr(ns, "command", None):
        parser.print_help(sys.stdout)
        return 0
    handler = getattr(ns, "_handler", None)
    if handler is None:
        parser.print_help(sys.stdout)
        return 2
    try:
        return int(handler(ns))
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
