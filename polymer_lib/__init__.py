"""
polymer_lib — a library for building 3D polymer chains from monomer SMILES.

Main entry points:
    Monomer      — represents a single monomer unit with 3D geometry.
    Polymerizer  — orchestrates chain growth with retry/rollback logic.
    *Optimizer   — optional structure optimizers (MMFF, OpenFF).
"""

from .monomer import Monomer
from .polymerizer import Polymerizer
from .optimizers import (
    BaseOptimizer,
    MMFFOptimizer,
    OpenFFOptimizer,
    get_optimizer,
)

__all__ = [
    'Monomer',
    'Polymerizer',
    'BaseOptimizer',
    'MMFFOptimizer',
    'OpenFFOptimizer',
    'get_optimizer',
]