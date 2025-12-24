# TriBrain for World Models (Executive + Bayesian Belief + Metacognition)

A **control layer** for world models that follows the SOPHIA-style loop:

> **generate → critic → update belief → metacognitive control → refine → repeat**

This repo is meant to be dropped **on top of** an existing world model (e.g., video diffusion) to make test-time refinement **more stable, more data-driven, and less wasteful**.

## Why this exists

Refinement loops often fail in predictable ways:
- the critic is noisy or inconsistent → the loop “thrashes”
- prompt drift improves one dimension while breaking another
- compute is wasted past the point of diminishing returns

TriBrain addresses this with:

- **Executive brain:** deterministic refiner policy + optional LLM refiner (budgeted + cached)
- **Bayesian belief brain:** belief over failure modes + critic reliability (calibration-ready)
- **Metacognitive brain:** stop/plateau/drift control + hard budgets (time/tokens/calls)
- **Ontology layer:** extract a task ontology and compress it into the prompt (token saver)
- **RL (lightweight):** Thompson-sampling bandit learns which refinement “focus” works best

## Architecture
<img width="876" height="437" alt="image" src="https://github.com/user-attachments/assets/740d9ae7-b1b2-4f33-84d2-05cdb51e8258" />



## Install

```bash
pip install -e ".[dev]"
# Optional: real VLM critic
pip install -e ".[dev,vlm]"
# Optional: preference learning + LoRA reward model trainer
pip install -e ".[dev,lora]"
```

## Quick start (generic external generator)

TriBrain can wrap any generator you can call from the shell.

```bash
python -m tribrain.cli run-subprocess \
  --cmd 'python your_generator.py --prompt "{prompt}" --out "{out_dir}" --seed {seed}' \
  --prompt "pick up the red cube and place it in the bowl" \
  --out outputs/run1 \
  --iters 3 \
  --use-ontology \
  --max-wall 900 \
  --llm-backend none
```

Outputs:
- `outputs/run1/trace.jsonl` – full iteration trace (prompt, scores, belief, meta decisions)
- `outputs/run1/best/` – best artifact(s) seen so far


## v6.2-final: Preference learning + LoRA-ready continual improvement

TriBrain can now export **pairwise preferences** from a run trace and train:
- a **small CPU Bradley–Terry reward model** (fast, stable)
- an optional **LoRA text reward model** (HF/PEFT)

Export pairs from a run:
```bash
python -m tribrain.cli export-preference-pairs --run-dir outputs/run1 --out outputs/run1/pairs.jsonl
```

Train small preference reward model (CPU):
```bash
python -m tribrain.cli train-preference-rm --pairs outputs/run1/pairs.jsonl --out outputs/run1/bt_reward.json
```

Train LoRA reward model (requires `.[lora]`):
```bash
python -m tribrain.cli train-lora-rm --pairs outputs/run1/pairs.jsonl --base-model distilbert-base-uncased --out-dir outputs/run1/lora_rm
```

These artifacts are meant to be plugged into your SOPHIA loop to reduce thrashing and improve *cost-aware convergence*.

## WoW integration (optional)

If you have the WoW repo locally, you can run the TriBrain loop around their inference scripts:

```bash
python -m tribrain.cli run-wow \
  --wow-repo /path/to/wow-world-model \
  --model dit-2b \
  --prompt "open the drawer, pick the spoon, place it on the table" \
  --out outputs/wow_run \
  --iters 4 \
  --use-ontology \
  --llm-backend ollama \
  --llm-model llama3.1:8b-instruct
```

TriBrain does **not** modify WoW code. It calls WoW inference scripts and evaluates outputs with built-in critics.
You can attach additional critics (e.g., a VLM critic) via the `Critic` interface.

## Calibration (turn “scores” into reliable signals)

TriBrain includes a SQLite calibration store (`tribrain.storage.calibration`) to record:
- critic scores
- a human/ground-truth label (success/fail or preference)
- context tags (task family, robot, camera, etc.)

This enables reliability-weighting of critics over time.

Add human feedback for a specific critic:

```bash
python -m tribrain.cli calibrate-critic \
  --run-dir outputs/wow_run \
  --critic FlickerCritic \
  --success 1 \
  --weight 1.0 \
  --tag task_family=drawer
```

## Continual improvement

TriBrain continuously improves *itself* with **episodic replay** (no regeneration needed):
- **Episodic memory** stores (context → action → outcome → cost) for every run.
- **Replay** updates the contextual controller and the metacognitive value model *offline* at the end of each run.
- A **meta-value model** learns the **marginal value of “one more iteration”** (expected improvement vs cost),
  so the system can stop early to save time/tokens/energy.

Create a run report (Markdown + optional plots):

```bash
python -m tribrain.cli dashboard --run-dir outputs/wow_run
```

Fine-tuning the **world model itself** is model-specific. TriBrain already exports traces and preference datasets,
so once WoW exposes a training entrypoint, you can attach a training hook (LoRA / DPO / reward-modeling) cleanly.

Export a dataset from a run (for labeling or training):

```bash
python -m tribrain.cli export-dataset   --run-dir outputs/wow_run   --out datasets/wow_run_good.jsonl   --min-score 0.75
```

## License
Apache 2.0
