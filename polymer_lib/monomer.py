"""
Monomer representation.

A Monomer wraps an RDKit Mol object together with its SMILES string and
pre-computed linker indices. It handles 3D coordinate generation and
provides a copy() method for safe reuse during polymerization.
"""

from . import utils
from . import poly


class Monomer:
    """
    A monomer unit with pre-computed 3D geometry and linker information.

    Attributes:
        smiles: The input SMILES string (with '*' for connection points).
        mol: RDKit Mol object with 3D coordinates and linker flags set.
    """

    def __init__(self, smiles: str):
        """
        Build a Monomer from a SMILES string.

        The SMILES must contain exactly two '*' atoms representing the
        head and tail connection points. 3D coordinates are generated
        via ETKDGv3, and the linker atoms are identified and flagged.

        Args:
            smiles: SMILES string with two '*' connection points.

        Raises:
            ValueError: If SMILES cannot be parsed or does not have
                exactly two linker atoms.
        """
        self.smiles = smiles
        self.mol = utils.mol_from_smiles(smiles, coord=True)
        if self.mol is None:
            raise ValueError(f"Cannot build monomer from SMILES: {smiles}")
        poly.set_linker_flag(self.mol)

    def copy(self):
        """
        Create a deep copy of this Monomer.

        The copy has its own independent Mol object with linker flags
        re-initialized, so it can be safely modified during polymerization.

        Returns:
            A new Monomer instance with a deep-copied Mol.
        """
        new_monomer = Monomer.__new__(Monomer)
        new_monomer.smiles = self.smiles
        new_monomer.mol = utils.deepcopy_mol(self.mol)
        poly.set_linker_flag(new_monomer.mol)
        return new_monomer