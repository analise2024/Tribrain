"""Lightweight ontology for world-model tasks.

This is not a "knowledge graph" for the internet. It's a *task ontology*:
objects, action primitives, and constraints extracted from instructions.

The goal is to:
1) standardize representations across brains/critics,
2) reduce prompt token waste by compressing repetitive info,
3) enable more stable refinement policies.
"""

from .compress import compress_ontology
from .extractor import OntologyExtractor
from .schema import Action, Entity, TaskOntology

__all__ = ["Action", "Entity", "TaskOntology", "OntologyExtractor", "compress_ontology"]
