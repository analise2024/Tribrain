from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ..budget import BudgetController, BudgetState
from ..llm.base import LLMClient
from ..types import FailureMode


@dataclass
class LLMRefinerConfig:
    cache_path: Path | None = None
    max_tokens: int = 350
    temperature: float = 0.0
    timeout_s: float = 20.0

    # Hard cap to keep outputs short and actionable.
    max_prompt_chars: int = 2000


class ExecutiveLLMRefiner:
    """LLM-backed executive refiner.

    This is where your "LLM brain" lives. It must be:
    - fast (timeouts),
    - bounded (token budgets),
    - conservative (doesn't rewrite the whole instruction unless necessary),
    - cacheable (avoid repeated calls for near-identical feedback).
    """

    def __init__(self, llm: LLMClient, cfg: LLMRefinerConfig | None = None):
        self.llm = llm
        self.cfg = cfg or LLMRefinerConfig()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        # Rough estimate: ~4 chars/token in English; keep conservative.
        return max(1, int(len(text) / 3.5))

    def refine(
        self,
        prompt: str,
        ontology_hint: str,
        belief_probs: Mapping[FailureMode, float],
        fused_scores: Mapping[FailureMode, float],
        drift_flag: bool,
        budget: BudgetController,
        budget_state: BudgetState,
    ) -> str:
        prompt = (prompt or "").strip()
        if not prompt:
            return prompt

        # Cache key captures the information that should change the rewrite.
        blob = {
            "prompt": prompt,
            "ontology": ontology_hint[:500],
            "belief": {k.name: float(v) for k, v in belief_probs.items()},
            "scores": {k.name: float(v) for k, v in fused_scores.items()},
            "drift": bool(drift_flag),
        }
        key = hashlib.sha256(json.dumps(blob, sort_keys=True).encode("utf-8")).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        # Budget check
        est = self._estimate_tokens(prompt + ontology_hint)
        if not budget.allow_llm_call(budget_state, est_tokens=min(self.cfg.max_tokens, est + 120)):
            return prompt

        system = (
            "You are an executive controller for a world model inference loop (SOPHIA-style). "
            "Your job: revise the instruction to improve physical plausibility, task completion, temporal coherence, "
            "and visual stability. You must be conservative: do not delete the original goal; only add or clarify constraints. "
            "Return STRICT JSON with keys: refined_prompt (string), edits (list of strings). No extra keys. "
            "Keep it short."
        )

        # Provide a *compressed* belief summary to keep tokens low.
        top_modes = sorted(
            [(m, float(belief_probs.get(m, 0.0))) for m in FailureMode if m != FailureMode.OTHER],
            key=lambda x: x[1],
            reverse=True,
        )[:3]
        belief_txt = ", ".join(f"{m.name}:{p:.2f}" for m, p in top_modes)
        score_txt = ", ".join(
            f"{m.name}:{float(fused_scores.get(m, 0.0)):.2f}" for m in (FailureMode.PHYSICS, FailureMode.TASK, FailureMode.SMOOTHNESS, FailureMode.QUALITY)
        )

        user = (
            f"CURRENT_INSTRUCTION:\n{prompt}\n\n"
            f"ONTOLOGY_HINT:\n{ontology_hint}\n\n"
            f"BELIEF_TOP:{belief_txt}\nSCORES:{score_txt}\nDRIFT_FLAG:{int(bool(drift_flag))}\n\n"
            "Return JSON now."
        )

        txt = self.llm.chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=int(self.cfg.max_tokens),
            temperature=float(self.cfg.temperature),
            timeout_s=float(self.cfg.timeout_s),
        )

        out = self._parse_refined_prompt(txt) or prompt
        out = out.strip()
        if len(out) > self.cfg.max_prompt_chars:
            out = out[: self.cfg.max_prompt_chars - 1].rstrip() + "…"

        # Note usage + cache.
        budget.note_llm_usage(budget_state, tokens_used=min(self.cfg.max_tokens, self._estimate_tokens(txt)))
        self._cache_set(key, out)
        return out

    def _parse_refined_prompt(self, txt: str) -> str | None:
        try:
            import re

            m = re.search(r"\{.*\}", txt.strip(), re.DOTALL)
            blob = m.group(0) if m else txt
            data = json.loads(blob)
            rp = data.get("refined_prompt")
            if isinstance(rp, str) and rp.strip():
                return rp
        except Exception:
            return None
        return None

    def _cache_get(self, key: str) -> str | None:
        if self.cfg.cache_path is None:
            return None
        p = Path(self.cfg.cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        f = p / f"{key}.txt"
        if not f.exists():
            return None
        try:
            return f.read_text(encoding="utf-8").strip()
        except Exception:
            return None

    def _cache_set(self, key: str, value: str) -> None:
        if self.cfg.cache_path is None:
            return
        p = Path(self.cfg.cache_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        f = p / f"{key}.txt"
        with suppress(Exception):
            f.write_text(value, encoding="utf-8")
