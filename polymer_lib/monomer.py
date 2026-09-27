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

    def __init__(self, smiles: str, seed=None, tacticity_center=None):
        """
        Build a Monomer from a SMILES string.

        The SMILES must contain exactly two '*' atoms representing the
        head and tail connection points. 3D coordinates are generated
        via ETKDGv3, and the linker atoms are identified and flagged.

        Args:
            smiles: SMILES string with two '*' connection points.
            seed: Optional ETKDG random seed.
            tacticity_center: Optional atom index for the atom that becomes
                the stereocenter during head-to-tail polymerization. Defaults
                to the atom next to the tail linker when it is an sp3 carbon.

        Raises:
            ValueError: If SMILES cannot be parsed or does not have
                exactly two linker atoms.
        """
        self.smiles = smiles
        self.mol = utils.mol_from_smiles(smiles, coord=True, seed=seed)
        if self.mol is None:
            raise ValueError(f"Cannot build monomer from SMILES: {smiles}")
        has_valid_linkers = poly.set_linker_flag(self.mol)
        linker_count = sum(atom.GetBoolProp('linker') for atom in self.mol.GetAtoms())
        if linker_count != 2 or not has_valid_linkers:
            raise ValueError(
                f"A polymerizable monomer must have exactly two valid connection points; "
                f"found {linker_count} in SMILES: {smiles}"
            )
        self.tacticity_center = self._resolve_tacticity_center(tacticity_center)

    def _resolve_tacticity_center(self, center):
        """Resolve an optional tacticity atom index in the monomer graph."""
        from rdkit import Chem

        if center is not None:
            if not isinstance(center, int) or not 0 <= center < self.mol.GetNumAtoms():
                raise ValueError('tacticity_center must be a valid monomer atom index')
            atom = self.mol.GetAtomWithIdx(center)
            if atom.GetSymbol() != 'C' or atom.GetHybridization() != Chem.HybridizationType.SP3:
                raise ValueError('tacticity_center must identify an sp3 carbon atom')
            return center

        # In this head-to-tail builder, the tail neighbor is the backbone
        # atom that receives the next unit's connection and becomes the
        # repeat-unit stereocenter (when chemically applicable).
        tail_neighbor = self.mol.GetIntProp('tail_ne_idx')
        atom = self.mol.GetAtomWithIdx(tail_neighbor)
        if atom.GetSymbol() == 'C' and atom.GetHybridization() == Chem.HybridizationType.SP3:
            return tail_neighbor
        return None

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
        new_monomer.tacticity_center = self.tacticity_center
        new_monomer.mol = utils.deepcopy_mol(self.mol)
        if not poly.set_linker_flag(new_monomer.mol):
            raise ValueError(f"Copied monomer has invalid connection points: {self.smiles}")
        return new_monomer
