from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Entity:
    """An object or actor referenced in a task instruction."""

    name: str
    kind: str = "object"  # object|tool|agent|surface|container|other
    attrs: dict[str, str] = field(default_factory=dict)


@dataclass
class Action:
    """An action primitive with optional preconditions/postconditions."""

    verb: str
    target: str = ""
    tool: str = ""
    modifiers: dict[str, str] = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    postconditions: list[str] = field(default_factory=list)


@dataclass
class TaskOntology:
    """Structured representation of an instruction."""

    entities: list[Entity] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    raw_instruction: str = ""
