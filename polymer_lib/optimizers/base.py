"""
Abstract base class for structure optimizers.

All optimizers must implement the `optimize(mol)` method and expose a
`name` property for logging purposes.
"""

from abc import ABC, abstractmethod
from rdkit import Chem


class BaseOptimizer(ABC):
    """
    Abstract base class for 3D structure optimizers.

    Subclasses must implement:
        - optimize(mol, confId): perform the optimization in place.
        - name (property): a short human-readable identifier.
    """

    @abstractmethod
    def optimize(self, mol: Chem.Mol, confId: int = 0) -> Chem.Mol:
        """
        Optimize the 3D structure of a molecule.

        Args:
            mol: RDKit Mol object with 3D coordinates.
            confId: Conformer ID to optimize (default 0).

        Returns:
            The same Mol object with updated coordinates.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this optimizer (used in log messages)."""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}>"