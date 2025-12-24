from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, cast

from .schema import Action, Entity, TaskOntology


class _ChatLLM(Protocol):
    """Minimal protocol for an LLM client used by OntologyExtractor.

    We intentionally keep this tiny to avoid binding the project to a specific SDK.
    """

    def chat(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        timeout_s: float,
    ) -> str: ...


@dataclass
class OntologyExtractorConfig:
    # If an LLM client is provided, we use it; otherwise we fall back to a rule-based extractor.
    max_llm_tokens: int = 350
    timeout_s: float = 12.0


class OntologyExtractor:
    """Extract a compact task ontology from a natural language instruction.

    The ontology is used as a *token saver* and as a shared representation across brains.
    """

    def __init__(self, llm: _ChatLLM | None = None, cfg: OntologyExtractorConfig | None = None):
        self.llm = llm
        self.cfg = cfg or OntologyExtractorConfig()

    def extract(self, instruction: str) -> TaskOntology:
        instruction = (instruction or "").strip()
        if not instruction:
            return TaskOntology(raw_instruction="")

        if self.llm is not None:
            try:
                return self._extract_with_llm(instruction)
            except Exception:
                # fall back
                pass
        return self._extract_rule_based(instruction)

    def _extract_with_llm(self, instruction: str) -> TaskOntology:
        system = (
            "You extract a compact task ontology from an instruction for a robot/world model. "
            "Return STRICT JSON with keys: entities (list), actions (list), constraints (list). "
            "Each entity: {name, kind, attrs}. Each action: {verb, target, tool, modifiers, preconditions, postconditions}. "
            "Do NOT include any extra keys. Keep lists short and high-signal."
        )
        user = f"Instruction: {instruction}"
        assert self.llm is not None
        txt = self.llm.chat(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=self.cfg.max_llm_tokens,
            temperature=0.0,
            timeout_s=self.cfg.timeout_s,
        )

        data = self._safe_json(txt)
        return self._from_dict(instruction, data)

    @staticmethod
    def _safe_json(txt: str) -> dict[str, Any]:
        # Try to locate the first JSON object in the string.
        txt = txt.strip()
        m = re.search(r"\{.*\}", txt, re.DOTALL)
        blob = m.group(0) if m else txt
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return cast(dict[str, Any], obj)
        return {}

    @staticmethod
    def _from_dict(instruction: str, data: dict[str, Any]) -> TaskOntology:
        ents = []
        for e in (data.get("entities") or []):
            if not isinstance(e, dict):
                continue
            name = str(e.get("name", "")).strip()
            if not name:
                continue
            ents.append(
                Entity(
                    name=name,
                    kind=str(e.get("kind", "object")),
                    attrs={str(k): str(v) for k, v in (e.get("attrs") or {}).items()} if isinstance(e.get("attrs"), dict) else {},
                )
            )

        acts = []
        for a in (data.get("actions") or []):
            if not isinstance(a, dict):
                continue
            verb = str(a.get("verb", "")).strip()
            if not verb:
                continue
            acts.append(
                Action(
                    verb=verb,
                    target=str(a.get("target", "")),
                    tool=str(a.get("tool", "")),
                    modifiers={str(k): str(v) for k, v in (a.get("modifiers") or {}).items()} if isinstance(a.get("modifiers"), dict) else {},
                    preconditions=[str(x) for x in (a.get("preconditions") or []) if str(x).strip()],
                    postconditions=[str(x) for x in (a.get("postconditions") or []) if str(x).strip()],
                )
            )

        constraints = [str(x) for x in (data.get("constraints") or []) if str(x).strip()]
        return TaskOntology(entities=ents, actions=acts, constraints=constraints, raw_instruction=instruction)

    @staticmethod
    def _extract_rule_based(instruction: str) -> TaskOntology:
        # Very small, fast extractor: verbs + naive noun phrases.
        verbs = [
            "grasp",
            "pick",
            "lift",
            "place",
            "move",
            "push",
            "pull",
            "open",
            "close",
            "insert",
            "remove",
            "rotate",
            "turn",
            "pour",
            "stack",
        ]
        lower = instruction.lower()
        acts: list[Action] = []
        for v in verbs:
            if re.search(rf"\b{re.escape(v)}\b", lower):
                acts.append(Action(verb=v))

        # Entities: grab simple color/object patterns
        entity_names: set[str] = set()
        for m in re.finditer(r"\b(red|blue|green|yellow|black|white)\s+([a-zA-Z][a-zA-Z0-9_-]{2,})\b", lower):
            entity_names.add(" ".join(m.groups()))
        # Also include the word after 'the'
        for m in re.finditer(r"\bthe\s+([a-zA-Z][a-zA-Z0-9_-]{2,})\b", lower):
            entity_names.add(m.group(1))

        ents = [Entity(name=n, kind="object") for n in sorted(entity_names)][:12]

        constraints = []
        if "avoid" in lower or "without" in lower:
            constraints.append("avoid violations stated in instruction")
        if "smooth" in lower:
            constraints.append("smooth motion")
        if "careful" in lower:
            constraints.append("careful handling")

        return TaskOntology(entities=ents, actions=acts, constraints=constraints, raw_instruction=instruction)
