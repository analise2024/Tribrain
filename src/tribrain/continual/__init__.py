"""Continual improvement utilities.

This package focuses on *operational* continual learning:
- export datasets from TriBrain traces (for human labeling or training)
- run external fine-tuning commands in a repeatable, auditable way

Note: Fine-tuning a specific world model (e.g. WoW DiT checkpoints) depends on the upstream
training pipeline. TriBrain deliberately keeps this modular.
"""

from .dataset import export_preference_dataset

__all__ = ["export_preference_dataset"]
