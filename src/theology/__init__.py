# src/theology/__init__.py
"""
Theology module for Bible-AI.

Handles theological validation, doctrines, denominational variations, controversial topics,
and pastoral sensitivity to ensure theologically sound outputs.
"""

from .controversial import ControversialHandler
from .denominational import DenominationalAdjuster
from .doctrines import DoctrineChecker
from .pastoral import PastoralSensitivity
from .validator import TheologicalValidator

__all__ = [
    "TheologicalValidator",
    "DoctrineChecker",
    "DenominationalAdjuster",
    "ControversialHandler",
    "PastoralSensitivity",
]
