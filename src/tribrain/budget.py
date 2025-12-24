from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(frozen=True)
class BudgetConfig:
    """Hard budgets to keep inference runs bounded.

    These are *mechanical* controls for cost/energy/time.
    """

    max_wall_time_s: float = 20 * 60
    max_iter_time_s: float = 10 * 60

    # LLM budgets (token/call caps). The project is designed to run even without an LLM,
    # but when present it must be bounded.
    llm_max_calls: int = 4
    llm_max_tokens: int = 1200


@dataclass
class BudgetState:
    start_time: float
    llm_calls: int = 0
    llm_tokens_used: int = 0
    total_iter_time_s: float = 0.0

    def to_dict(self) -> dict:
        return {
            "llm_calls": int(self.llm_calls),
            "llm_tokens_used": int(self.llm_tokens_used),
            "total_iter_time_s": float(self.total_iter_time_s),
        }

    @staticmethod
    def from_dict(d: dict, start_time: float | None = None) -> BudgetState:
        st = float(start_time if start_time is not None else time.time())
        return BudgetState(
            start_time=st,
            llm_calls=int(d.get("llm_calls", 0)) if isinstance(d, dict) else 0,
            llm_tokens_used=int(d.get("llm_tokens_used", 0)) if isinstance(d, dict) else 0,
            total_iter_time_s=float(d.get("total_iter_time_s", 0.0)) if isinstance(d, dict) else 0.0,
        )


class BudgetController:
    def __init__(self, cfg: BudgetConfig | None = None):
        self.cfg = cfg or BudgetConfig()

    def elapsed_wall_s(self, state: BudgetState) -> float:
        return float(time.time() - float(state.start_time))

    def should_stop(self, state: BudgetState) -> bool:
        return bool(
            (self.elapsed_wall_s(state) >= self.cfg.max_wall_time_s)
            or (state.llm_calls >= self.cfg.llm_max_calls)
            or (state.llm_tokens_used >= self.cfg.llm_max_tokens)
        )

    def iter_time_exceeded(self, iter_time_s: float) -> bool:
        return float(iter_time_s) >= self.cfg.max_iter_time_s

    def allow_llm_call(self, state: BudgetState, est_tokens: int = 256) -> bool:
        return bool(
            (state.llm_calls + 1 <= self.cfg.llm_max_calls)
            and (state.llm_tokens_used + max(0, int(est_tokens)) <= self.cfg.llm_max_tokens)
            and (self.elapsed_wall_s(state) < self.cfg.max_wall_time_s)
        )

    def note_llm_usage(self, state: BudgetState, tokens_used: int) -> None:
        state.llm_calls += 1
        state.llm_tokens_used += int(max(0, tokens_used))

    def note_iter_time(self, state: BudgetState, iter_time_s: float) -> None:
        state.total_iter_time_s += float(max(0.0, iter_time_s))
