# src/theology/__init__.py
"""
Theology module for Bible-AI.

Handles theological validation, doctrines, denominational variations, controversial topics,
and pastoral sensitivity to ensure theologically sound outputs.
"""

from .validator import TheologicalValidator
from .doctrines import DoctrineChecker
from .denominational import DenominationalAdjuster
from .controversial import ControversialHandler
from .pastoral import PastoralSensitivity

__all__ = [
    "TheologicalValidator",
    "DoctrineChecker",
    "DenominationalAdjuster",
    "ControversialHandler",
    "PastoralSensitivity"
]