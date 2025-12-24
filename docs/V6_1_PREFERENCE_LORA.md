# v6.1 Preference Learning and LoRA-Ready Training

This release adds a *real* preference-learning path to TriBrain:

- Export pairwise preferences from run traces (best-vs-rest or adjacent improvements).
- Train a small, CPU-friendly Bradley–Terry reward model (logistic on feature differences).
- Train an optional LoRA text reward model using HF Transformers + PEFT.

## Why preference learning

Critic scores are noisy. Preference learning stabilizes signals by learning a reward function that:
- explains which outputs are *actually preferred* (by humans or validated proxies),
- generalizes across tasks via context features.

## Export preference pairs

A TriBrain run produces `trace.jsonl`. Export pairs:

```bash
python -m tribrain.cli export-preference-pairs --run-dir <RUN_DIR> --out <RUN_DIR>/pairs.jsonl
```

Options:
- `--strategy best_vs_rest` (default): best K candidates preferred over lower-scoring ones.
- `--strategy adjacent`: prefer iteration t+1 over t when score improves by `--min-gap`.

## Train small Bradley–Terry reward model (CPU)

```bash
python -m tribrain.cli train-preference-rm --pairs <RUN_DIR>/pairs.jsonl --out <RUN_DIR>/bt_reward.json
```

This model is cheap, interpretable, and good for:
- stabilizing online control,
- smoothing reward signals for hierarchical controller updates,
- calibrating which critics matter.

## Train LoRA text reward model (optional)

Install extras:

```bash
pip install -e ".[lora]"
```

Train:

```bash
python -m tribrain.cli train-lora-rm --pairs <RUN_DIR>/pairs.jsonl --base-model distilbert-base-uncased --out-dir <RUN_DIR>/lora_rm
```

This trains a scalar reward model with pairwise loss:
`-log sigmoid(r(chosen) - r(rejected))`.

## Connecting to WoW

TriBrain already exports:
- prompts,
- videos (artifact paths),
- critic signals,
- preference pairs.

Once WoW exposes a stable training entrypoint for the world model, you can:
1) use TriBrain traces to build caption datasets (prompt → video),
2) use preference pairs to train reward models or to filter data,
3) optionally use reward-model scores to drive data selection and continual fine-tuning.

TriBrain intentionally separates *data and control* from *model training* so that connecting to WoW is a wiring task, not a rewrite.

## Using the preference RM in TriBrain runs

Once trained:

```bash
python -m tribrain.cli run-wow ... \
  --preference-rm <RUN_DIR>/bt_reward.json \
  --preference-weight 0.25
```

This adds a preference-based stabilizer term to the combined reward used by the controller.
