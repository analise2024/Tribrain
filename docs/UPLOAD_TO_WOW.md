# Uploading / Contributing this project into WoW

This document is practical guidance for integrating TriBrain into the WoW upstream repository as a contribution.

## Strategy (recommended)
1. Keep TriBrain as a **separate repo** (this one) for fast iteration.
2. Contribute to WoW as:
   - a new script under `scripts/` (e.g., `infer_wow_*_sophia3brain.py`), and/or
   - a small package folder (e.g., `sophia3brain/`) that is self-contained.

## PR checklist before you open a Pull Request to WoW
- Keep changes minimal and localized (ideally new files + small wiring)
- Provide a clear README section or docs file:
  - what it does
  - how to run it
  - expected outputs
- Add a JSONL trace output for debugging/repro
- Avoid adding heavyweight dependencies unless strictly required

## Typical GitHub PR flow
1. Fork the WoW repo
2. Create a feature branch
3. Implement changes + run tests / run a short demo
4. Open a Pull Request with:
   - what problem it solves
   - how to reproduce
   - links to any results (small samples or logs)

## If maintainers want it as “Phase 3 SOPHIA code”
Be ready to:
- keep APIs stable
- document interfaces for critic/refiner/belief/meta
- add at least one small regression-style test (if the repo supports it)
