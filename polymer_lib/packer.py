"""
Packmol wrapper for packing polymer chains into a simulation box.
Uses the native OpenFF Interchange pack_box implementation for maximum robustness.
"""

import tempfile
import logging
from rdkit import Chem
from openff.toolkit import Molecule
from openff.units import unit
from openff.interchange.components._packmol import pack_box, UNIT_CUBE
from openff.interchange.exceptions import PACKMOLRuntimeError

logger = logging.getLogger(__name__)


class PackmolPacker:
    """
    Wraps the native OpenFF pack_box to pack multiple polymer chains 
    into a periodic box at a target density, with automatic retry logic.
    """
    
    def __init__(self, density: float = 0.3, max_attempts: int = 5, 
                 tolerance: float = 2.0):
        """
        Args:
            density: Target density in g/mL (default 0.3).
            max_attempts: Number of retry attempts if Packmol fails.
            tolerance: Minimum distance between atoms of different molecules (Angstrom).
        """
        self.density = density * unit.gram / unit.milliliter
        self.max_attempts = max_attempts
        self.tolerance = tolerance * unit.angstrom

    def _rdkit_to_openff(self, mol: Chem.Mol) -> Molecule:
        """
        Convert an RDKit Mol to an OpenFF Molecule, handling stereochemistry.
        """
        # Assign stereochemistry from 3D coordinates to satisfy OpenFF's strict checks
        try:
            Chem.AssignStereochemistryFrom3D(mol, confId=0, force=True, cleanIt=True)
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True, flagPossibleStereoCenters=True)
        except Exception:
            pass  # Fallback to allow_undefined_stereo in from_rdkit
            
        return Molecule.from_rdkit(mol, allow_undefined_stereo=True)

    def pack(self, chains: list[Chem.Mol]):
        """
        Pack the given list of polymer chains into a single periodic box.
        
        Args:
            chains: List of RDKit Mol objects (polymer chains).
            
        Returns:
            An OpenFF Topology with the packed system and box vectors set.
        """
        if not chains:
            raise ValueError("List of chains is empty.")
            
        logger.info(f"Converting {len(chains)} RDKit molecules to OpenFF Molecules...")
        off_molecules = []
        for i, mol in enumerate(chains):
            try:
                off_mol = self._rdkit_to_openff(mol)
                off_molecules.append(off_mol)
            except Exception as e:
                logger.error(f"Failed to convert chain {i} to OpenFF Molecule: {e}")
                raise

        # We pack exactly 1 copy of each chain provided in the list.
        number_of_copies = [1] * len(off_molecules)
        
        logger.info(f"Packing {len(off_molecules)} chains at target density {self.density}...")
        
        for attempt in range(1, self.max_attempts + 1):
            logger.info(f"Packmol attempt {attempt}/{self.max_attempts}")
            try:
                # Create a temporary directory for packmol working files
                with tempfile.TemporaryDirectory() as temp_dir:
                    topology = pack_box(
                        molecules=off_molecules,
                        number_of_copies=number_of_copies,
                        target_density=self.density,
                        tolerance=self.tolerance,
                        box_shape=UNIT_CUBE,
                        working_directory=temp_dir,
                        retain_working_files=False,
                    )
                logger.info("Packmol succeeded!")
                return topology
                
            except PACKMOLRuntimeError as e:
                logger.warning(f"Packmol failed on attempt {attempt}: {e}")
                if attempt == self.max_attempts:
                    raise RuntimeError(
                        f"Packmol failed to pack the system after {self.max_attempts} attempts. "
                        "Check the logs for details."
                    ) from e
                    
        raise RuntimeError("Packmol failed unexpectedly.")