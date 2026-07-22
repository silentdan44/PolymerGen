"""
MMFF94(s) optimizer based on RDKit.

Fast, requires no external dependencies beyond RDKit. Suitable for
rapid geometry relaxation during polymer chain growth.
"""

from rdkit.Chem import AllChem
from rdkit import Chem
from .base import BaseOptimizer


class MMFFOptimizer(BaseOptimizer):
    """
    Structure optimizer using the MMFF94s force field from RDKit.

    MMFF94s is the "small-molecule" variant of MMFF94, tuned for
    non-biological organic molecules.
    """

    def __init__(self, max_iters: int = 50, variant: str = 'MMFF94s',
                 non_bonded_thresh: float = 3.0):
        """
        Args:
            max_iters: Maximum number of optimization iterations.
                Use 0 for convergence-based termination.
            variant: Force field variant ('MMFF94' or 'MMFF94s').
            non_bonded_thresh: Distance cutoff (Angstrom) beyond which
                non-bonded interactions are ignored.
        """
        self.max_iters = max_iters
        self.variant = variant
        self.non_bonded_thresh = non_bonded_thresh

    def optimize(self, mol: Chem.Mol, confId: int = 0) -> Chem.Mol:
        """
        Run MMFF optimization on the given Mol.

        If MMFF parameters are unavailable for the molecule (e.g. unusual
        atom types), a warning is logged and the Mol is returned unchanged.

        Args:
            mol: RDKit Mol with 3D coordinates.
            confId: Conformer ID to optimize.

        Returns:
            The same Mol, with coordinates updated in place.
        """
        try:
            AllChem.MMFFOptimizeMolecule(
                mol,
                maxIters=self.max_iters,
                mmffVariant=self.variant,
                nonBondedThresh=self.non_bonded_thresh,
                confId=confId
            )
        except Exception as e:
            from ..utils import logger
            logger.warning(f'MMFF optimization failed: {e}')
        return mol

    @property
    def name(self) -> str:
        return f'mmff({self.variant}, iters={self.max_iters})'