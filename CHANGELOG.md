# Changelog

## 0.6.2 (v6.2-final)
- SQLite episodic migration rewritten as atomic, idempotent rebuild migration (no fragile ALTER TABLE loops).
- Added schema versioning, PRAGMA-based schema validation, and optional automatic backups on migration.
- Added regression tests that create legacy DBs (v1/v2) and migrate to latest, re-running migrations idempotently.
- CLI stabilized (python -m tribrain.cli works) and added `replay-update` command.
- Improved resilience: world-model failures stop cleanly with stop_reason and still write a run_result + dashboard.
- Critics are now failure-tolerant: optional VLM critic degrades gracefully on missing deps/runtime errors.

## 0.1.3 (v6.1)
- Preference dataset export (pairs) from run traces.
- Small Bradley–Terry preference reward model (CPU).
- Optional LoRA text reward model trainer (HF/PEFT).
- CLI commands: export-preference-pairs, train-preference-rm, train-lora-rm.


## v0.1.2 (2025-12-23)
- Added SQLite episodic memory (context → action → outcome → cost) for replay-driven continual improvement.
- Added ReplayEngine to update controller + meta-value model offline (saves time/tokens/energy).
- Added MetaValueModel (Bayesian) for learned early stopping based on marginal value vs cost.
- Added `tribrain dashboard` command producing a run report (Markdown + optional plots).
- Trace now records chosen arm, budgets, and meta-value signals for auditability.

## v0.1.1
- Added HierarchicalContextualTS controller (contextual hierarchical Bayesian Thompson sampling).
- Added BayesianRewardModel for stabilizing critic signals.
- Added HFVlmCritic (optional) for instruction-following + physics sanity using a real VLM.
- Added context feature extractor (ontology + critic history).

All notable changes to this project will be documented in this file.

The format is based on *Keep a Changelog* and the project adheres to *Semantic Versioning*.

## [0.1.0] - 2025-12-22
### Added
- TriBrain orchestrator (Executive + Bayesian Belief + Metacognitive control)
- Built-in video critics (flicker, flow smoothness, temporal jitter)
- JSONL tracing and SQLite calibration store
- WoW subprocess integration adapter (optional)
- CI (ruff + pytest) and packaging (pyproject)
