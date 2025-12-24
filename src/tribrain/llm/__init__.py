"""LLM client adapters.

TriBrain does not hardcode a single provider. For cost/energy control, the orchestrator
calls the LLM only when necessary and under strict budgets.

The most practical open-source default is Ollama (local), but you can add more adapters.
"""

from .base import LLMClient
from .ollama import OllamaClient

__all__ = ["LLMClient", "OllamaClient"]
