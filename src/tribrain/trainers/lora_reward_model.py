from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _lazy_import(module: str) -> Any:
    """Import optional heavy dependencies at runtime.

    We intentionally avoid importing HuggingFace/torch/datasets/peft at import time,
    so base installs and type-checking with only `.[dev]` do not break.
    """

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as e:  # pragma: no cover
        raise RuntimeError(
            "Missing optional dependencies. Install with: pip install -e '.[lora]'"
        ) from e


@dataclass(frozen=True)
class LoRATrainConfig:
    base_model: str = "distilbert-base-uncased"
    max_length: int = 384
    lr: float = 2e-5
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    output_dir: str = "out_lora_reward_model"
    seed: int = 7


def train_text_reward_model_lora(pairs_jsonl: Path, cfg: LoRATrainConfig) -> Path:
    """Train a text reward model with pairwise preference loss, optionally using LoRA.

    This trains a scalar reward head r(x). The loss is:
        L = -log sigmoid(r(chosen) - r(rejected))

    The resulting model is used to stabilize critic signals and to learn better refinement policies.

    Requires extras: torch, transformers, accelerate, peft, datasets
    """
    torch = _lazy_import("torch")
    datasets_mod = _lazy_import("datasets")
    peft_mod = _lazy_import("peft")
    transformers_mod = _lazy_import("transformers")

    Dataset = datasets_mod.Dataset
    LoraConfig = peft_mod.LoraConfig
    TaskType = peft_mod.TaskType
    get_peft_model = peft_mod.get_peft_model
    AutoModelForSequenceClassification = transformers_mod.AutoModelForSequenceClassification
    AutoTokenizer = transformers_mod.AutoTokenizer
    Trainer = transformers_mod.Trainer
    TrainingArguments = transformers_mod.TrainingArguments

    # Load pairs
    rows: list[dict[str, Any]] = []
    for line in pairs_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        instruction = str(obj.get("goal_instruction", ""))
        chosen = str(obj.get("chosen_prompt", ""))
        rejected = str(obj.get("rejected_prompt", ""))
        rows.append({"instruction": instruction, "chosen": chosen, "rejected": rejected})

    if not rows:
        raise ValueError("No preference pairs found.")

    tok = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)

    def encode(instr: str, cand: str) -> dict[str, Any]:
        text = f"Instruction:\n{instr}\nCandidate prompt:\n{cand}"
        enc = tok(text, truncation=True, max_length=cfg.max_length)
        # `enc` is usually a BatchEncoding (Mapping-like). Convert to a plain dict for type-checking.
        try:
            return dict(enc)
        except Exception:
            return {"text": text}

    # Build dataset of pairs (tokenize both sides)
    def map_fn(ex: dict[str, Any]) -> dict[str, Any]:
        a = encode(ex["instruction"], ex["chosen"])
        b = encode(ex["instruction"], ex["rejected"])
        out: dict[str, Any] = {}
        for k, v in a.items():
            out[f"chosen_{k}"] = v
        for k, v in b.items():
            out[f"rejected_{k}"] = v
        return out

    ds = Dataset.from_list(rows).map(map_fn, remove_columns=list(rows[0].keys()))

    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, num_labels=1)
    # LoRA config (for sequence classification)
    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        target_modules=None,  # let peft pick sensible defaults
    )
    model = get_peft_model(model, lora_cfg)

    def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
        # pad chosen and rejected separately
        chosen = [{k.replace("chosen_", ""): v for k, v in f.items() if k.startswith("chosen_")} for f in features]
        rejected = [{k.replace("rejected_", ""): v for k, v in f.items() if k.startswith("rejected_")} for f in features]
        batch_chosen = tok.pad(chosen, return_tensors="pt")
        batch_rejected = tok.pad(rejected, return_tensors="pt")
        return {"chosen": batch_chosen, "rejected": batch_rejected}

    class PairwiseRewardTrainer(Trainer):  # type: ignore[misc,valid-type]
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False) -> Any:
            chosen = inputs["chosen"]
            rejected = inputs["rejected"]
            r_c = model(**chosen).logits.squeeze(-1)
            r_r = model(**rejected).logits.squeeze(-1)
            loss = -torch.nn.functional.logsigmoid(r_c - r_r).mean()
            return (loss, {"r_c": r_c, "r_r": r_r}) if return_outputs else loss

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        logging_steps=20,
        save_steps=200,
        save_total_limit=2,
        seed=cfg.seed,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    trainer = PairwiseRewardTrainer(model=model, args=args, train_dataset=ds, data_collator=collate)
    trainer.train()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    return out_dir
