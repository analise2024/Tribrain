"""Reinforcement learning components.

TriBrain uses RL where it is cheap and immediately useful:
learning *refinement-control* policies (which constraints to add, when to stop, how to allocate compute).

We provide:
- PromptEditBandit: non-contextual Thompson sampling (baseline)
- HierarchicalContextualTS: contextual hierarchical controller (recommended)
"""

from .bandit import PromptEditBandit
from .hierarchical_contextual import HierarchicalContextualTS

__all__ = ["PromptEditBandit", "HierarchicalContextualTS"]
