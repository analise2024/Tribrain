from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    """Minimal chat interface.

    Implementations must be deterministic under temperature=0 and should support timeouts.
    """

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> str:
        ...
