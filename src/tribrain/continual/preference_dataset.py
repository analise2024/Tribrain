from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class RunRecord:
    prompt: str
    goal_instruction: str
    artifact_path: str
    overall_score: float
    critic_scores: dict[str, float]
    meta: dict[str, Any]
    cost: dict[str, float]


def _load_run_jsonl(path: Path) -> list[RunRecord]:
    records: list[RunRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        meta = dict(obj.get("meta", {}) or {})
        records.append(
            RunRecord(
                prompt=str(obj.get("prompt", "")),
                goal_instruction=str(meta.get("goal_instruction", "")),
                artifact_path=str(obj.get("artifact_path", "")),
                overall_score=float(obj.get("overall_score", 0.0)),
                critic_scores={k: float(v) for k, v in (obj.get("critic_scores", {}) or {}).items()},
                meta=meta,
                cost={k: float(v) for k, v in (obj.get("cost", {}) or {}).items()},
            )
        )
    return records


@dataclass(frozen=True)
class PreferencePair:
    goal_instruction: str
    chosen_prompt: str
    rejected_prompt: str
    chosen_artifact: str
    rejected_artifact: str
    chosen_score: float
    rejected_score: float
    meta: dict[str, Any]


def build_preference_pairs(
    run_jsonl: Path,
    *,
    min_gap: float = 0.03,
    top_k: int = 1,
    strategy: Literal["best_vs_rest", "adjacent"] = "best_vs_rest",
) -> list[PreferencePair]:
    """Build pairwise preferences from a TriBrain run trace.

    - best_vs_rest: take the top_k candidates by overall_score as 'chosen' and pair against lower-score candidates.
    - adjacent: pair each iteration with the previous one if score improved by min_gap.

    Output is meant to train reward models (small BT or LoRA reward model) and/or preference-based policies.
    """
    records = _load_run_jsonl(run_jsonl)
    if not records:
        return []

    pairs: list[PreferencePair] = []

    if strategy == "adjacent":
        for a, b in zip(records[:-1], records[1:], strict=True):
            if (b.overall_score - a.overall_score) >= min_gap:
                pairs.append(
                    PreferencePair(
                        goal_instruction=b.goal_instruction or a.goal_instruction,
                        chosen_prompt=b.prompt,
                        rejected_prompt=a.prompt,
                        chosen_artifact=b.artifact_path,
                        rejected_artifact=a.artifact_path,
                        chosen_score=b.overall_score,
                        rejected_score=a.overall_score,
                        meta={"mode": "adjacent"},
                    )
                )
        return pairs

    # best_vs_rest
    ranked = sorted(records, key=lambda r: r.overall_score, reverse=True)
    chosen = ranked[: max(1, top_k)]
    rejected = ranked[max(1, top_k) :]
    for c in chosen:
        for r in rejected:
            if (c.overall_score - r.overall_score) < min_gap:
                continue
            pairs.append(
                PreferencePair(
                    goal_instruction=c.goal_instruction,
                    chosen_prompt=c.prompt,
                    rejected_prompt=r.prompt,
                    chosen_artifact=c.artifact_path,
                    rejected_artifact=r.artifact_path,
                    chosen_score=c.overall_score,
                    rejected_score=r.overall_score,
                    meta={"mode": "best_vs_rest"},
                )
            )
    return pairs


def export_preference_pairs_jsonl(
    run_jsonl: Path,
    out_path: Path,
    *,
    min_gap: float = 0.03,
    top_k: int = 1,
    strategy: Literal["best_vs_rest", "adjacent"] = "best_vs_rest",
) -> int:
    pairs = build_preference_pairs(run_jsonl, min_gap=min_gap, top_k=top_k, strategy=strategy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p.__dict__, ensure_ascii=False) + "\n")
            n += 1
    return n


def export_dpo_style_dataset(
    pairs_jsonl: Path,
    out_path: Path,
    *,
    system_prefix: str = "You are a refinement agent for a world model. Given an instruction, propose the best prompt.",
) -> int:
    """Export to a TRL/DPO-friendly JSONL format.

    This is for training the *text refiner* (Executive LLM) via preference learning.
    chosen/rejected are *prompt texts* in that setting.

    If you want reward-model training over *videos*, use export_preference_pairs_jsonl and a reward model that reads video metrics.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with pairs_jsonl.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            obj = json.loads(line)
            instr = str(obj.get("goal_instruction", ""))
            ex = {
                "prompt": system_prefix + "\nInstruction: " + instr,
                "chosen": str(obj.get("chosen_prompt", "")),
                "rejected": str(obj.get("rejected_prompt", "")),
                "meta": obj.get("meta", {}),
            }
            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            n += 1
    return n
