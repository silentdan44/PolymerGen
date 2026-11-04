"""Pack chains with geometric checks before invoking Packmol."""

import logging
import math
import tempfile

import numpy as np
from rdkit import Chem
from scipy.spatial import cKDTree
from openff.toolkit import Molecule
from openff.units import unit
from openff.interchange.components._packmol import pack_box, UNIT_CUBE
from openff.interchange.exceptions import PACKMOLRuntimeError

logger = logging.getLogger(__name__)


def _check_chain_geometry(mol, index):
    """Detect severe non-local atomic clashes; bonded and 1-3/1-4 pairs are excluded."""
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError(f"Chain {index} must have 3D coordinates")
    coords = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    if not np.isfinite(coords).all():
        raise ValueError(f"Chain {index} has non-finite coordinates")
    table = Chem.GetPeriodicTable()
    radii = [table.GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()]
    if not radii:
        raise ValueError(f"Chain {index} has no atoms")
    for i, j in sorted(cKDTree(coords).query_pairs(max(radii))):
        # Atoms separated by at most three bonds have geometry governed by
        # local bonded terms; they must not be treated as chain self-clashes.
        visited = {i}
        frontier = {i}
        for _ in range(3):
            frontier = {n.GetIdx() for k in frontier for n in mol.GetAtomWithIdx(k).GetNeighbors()} - visited
            visited.update(frontier)
        if j in visited:
            continue
        distance = float(np.linalg.norm(coords[i] - coords[j]))
        threshold = 0.5 * (radii[i] + radii[j]) * 0.5
        if distance < threshold:
            raise ValueError(
                f"Chain {index} has an internal nonbonded overlap: atoms {i} and {j} "
                f"are {distance:.3f} Å apart (clash threshold {threshold:.3f} Å). "
                "Relax or regenerate this chain before packing."
            )
    # The diagonal bounds every interatomic distance even after rotation.
    return float(np.linalg.norm(np.ptp(coords, axis=0)))


class PackmolPacker:
    """Pack a set of chains at a geometry-adjusted initial density."""

    def __init__(self, density: float = 0.3, max_attempts: int = 5,
                 tolerance: float = 2.0):
        if isinstance(density, bool) or not isinstance(density, (int, float)) or not math.isfinite(density) or density <= 0:
            raise ValueError("density must be a positive finite number in g/mL")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("tolerance must be a positive finite number in Å")
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        self.density = density * unit.gram / unit.milliliter
        self.max_attempts = max_attempts
        self.tolerance = tolerance * unit.angstrom
        self.effective_density = density

    def _rdkit_to_openff(self, mol):
        mol = Chem.Mol(mol)
        try:
            Chem.AssignStereochemistryFrom3D(mol, confId=0, force=True, cleanIt=True)
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True, flagPossibleStereoCenters=True)
        except Exception:
            pass
        return Molecule.from_rdkit(mol, allow_undefined_stereo=True)

    def pack(self, chains):
        if not chains:
            raise ValueError("List of chains is empty")
        extents = [_check_chain_geometry(mol, i) for i, mol in enumerate(chains)]
        molecules = [self._rdkit_to_openff(mol) for mol in chains]
        # A cube wider than the diagonal of each axis-aligned bounding box
        # fits every chain in any orientation. Interchange reserves tolerance
        # at the box boundary; an extra Å provides a small numerical margin.
        side = max(extents) + self.tolerance.m_as(unit.angstrom) + 1.0
        total_mass = sum(atom.mass.m_as(unit.dalton) for mol in molecules for atom in mol.atoms)
        max_density = total_mass * 1.66053906660 / side**3
        requested = self.density.m_as(unit.gram / unit.milliliter)
        self.effective_density = min(requested, max_density)
        if self.effective_density < requested:
            logger.warning(
                "Requested initial density %.6g g/mL is too high for the chain dimensions; "
                "using %.6g g/mL (box side %.3f Å) before packing.",
                requested, self.effective_density, side,
            )
        actual_density = self.effective_density * unit.gram / unit.milliliter
        for attempt in range(1, self.max_attempts + 1):
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    return pack_box(
                        molecules=molecules, number_of_copies=[1] * len(molecules),
                        target_density=actual_density, tolerance=self.tolerance,
                        box_shape=UNIT_CUBE, working_directory=temp_dir,
                        retain_working_files=False,
                    )
            except PACKMOLRuntimeError as exc:
                logger.warning("Packmol attempt %d/%d failed: %s", attempt, self.max_attempts, exc)
                if attempt == self.max_attempts:
                    raise RuntimeError(
                        f"Packmol failed after {self.max_attempts} attempts at initial density "
                        f"{self.effective_density:.6g} g/mL: {exc}"
                    ) from exc


def pack_chains(chains, density=0.3, max_attempts=5, tolerance=2.0):
    """Pack RDKit polymer chains and return an OpenFF topology."""
    return PackmolPacker(density=density, max_attempts=max_attempts, tolerance=tolerance).pack(chains)
