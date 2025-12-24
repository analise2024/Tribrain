from __future__ import annotations

from .schema import TaskOntology


def compress_ontology(ont: TaskOntology, max_chars: int = 700) -> str:
    """Compress a TaskOntology into a token-light string.

    The goal is to keep it short enough to be appended to the instruction without exploding
    context length.
    """

    parts: list[str] = []
    if ont.entities:
        ents = "; ".join(e.name for e in ont.entities[:10])
        parts.append(f"Entities: {ents}")
    if ont.actions:
        acts = "; ".join(f"{a.verb}({a.target})" if a.target else a.verb for a in ont.actions[:10])
        parts.append(f"Actions: {acts}")
    if ont.constraints:
        cons = "; ".join(c for c in ont.constraints[:8])
        parts.append(f"Constraints: {cons}")

    out = " | ".join(parts).strip()
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out
