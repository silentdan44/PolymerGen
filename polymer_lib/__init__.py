"""
polymer_lib — a library for building 3D polymer chains from monomer SMILES.

Main entry points:
    Monomer      — represents a single monomer unit with 3D geometry.
    Polymerizer  — orchestrates chain growth with retry/rollback logic.
    *Optimizer   — optional structure optimizers (MMFF, OpenFF).
"""

from .monomer import Monomer
from .polymerizer import BuildConfig, BuildResult, Polymerizer, build_polymer
from .push_off import push_off_chain
from .api import API_VERSION, handle_request
from .optimizers import (
    BaseOptimizer,
    MMFFOptimizer,
    get_optimizer,
)

try:
    from .optimizers.openff import OpenFFOptimizer
except ImportError:
    OpenFFOptimizer = None

__all__ = [
    'Monomer',
    'Polymerizer',
    'BuildConfig',
    'BuildResult',
    'build_polymer',
    'push_off_chain',
    'API_VERSION',
    'handle_request',
    'BaseOptimizer',
    'MMFFOptimizer',
    'get_optimizer',
]

if OpenFFOptimizer is not None:
    __all__.append('OpenFFOptimizer')
