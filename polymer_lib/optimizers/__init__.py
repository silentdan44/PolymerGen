"""
Structure optimizers subpackage.

Provides a unified interface for geometry optimization via different
force fields (MMFF, OpenFF, ...). Use the `get_optimizer` factory
function to obtain an optimizer by name.
"""

from .base import BaseOptimizer
from .mmff import MMFFOptimizer
from .openff import OpenFFOptimizer

__all__ = ['BaseOptimizer', 'MMFFOptimizer', 'OpenFFOptimizer', 'get_optimizer']


def get_optimizer(name: str, **kwargs) -> BaseOptimizer:
    """
    Factory function for creating optimizer instances by name.

    Args:
        name: Optimizer identifier ('mmff' or 'openff').
        **kwargs: Additional arguments forwarded to the optimizer constructor.

    Returns:
        A configured BaseOptimizer instance.

    Raises:
        ValueError: If the name is not recognized.

    Examples:
        >>> opt = get_optimizer('mmff', max_iters=100)
        >>> opt = get_optimizer('openff', forcefield='openff-2.1.0.offxml')
    """
    name = name.lower()
    if name == 'mmff':
        return MMFFOptimizer(**kwargs)
    elif name == 'openff':
        return OpenFFOptimizer(**kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Available: 'mmff', 'openff'")