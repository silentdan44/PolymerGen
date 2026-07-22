"""
OpenFF + OpenMM optimizer.

Higher-quality force field than MMFF, with broad coverage of organic
chemistry via the OpenFF line of force fields (e.g. openff-2.1.0).
Uses OpenFF NAGL (graph neural network) for fast partial charge assignment,
avoiding the slow and fragile AmberTools/sqm pathway.
"""

import numpy as np
from rdkit import Chem
from .base import BaseOptimizer

# ============================================================
# Module-level imports (done ONCE, not on every optimize() call)
# ============================================================
from openff.toolkit import Molecule as OFFMolecule
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
from openff.nagl_models import list_available_nagl_models
from openff.interchange import Interchange
from openmm import unit, LocalEnergyMinimizer
import openmm
from rdkit import RDLogger

rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.ERROR)

# ============================================================
# Caches (module-level singletons)
# ============================================================
_FF_CACHE: dict = {}


# Default NAGL charge model. Other options:
DEFAULT_NAGL_MODEL = list_available_nagl_models()[-1]


class OpenFFOptimizer(BaseOptimizer):
    """
    Structure optimizer using OpenFF force fields via OpenMM.

    Performance notes:
        - ForceField instances are cached by name (parsed .offxml once).
        - All heavy imports happen at module load, not per call.
        - Partial charges are assigned via NAGL (GNN), which is ~100x
          faster than AM1BCC/sqm and does not require AmberTools.
    """

    def __init__(self,
                 forcefield: str = 'openff-2.1.0.offxml',
                 max_iters: int = 50,
                 platform: str = 'CPU',
                 tolerance: float = 10.0,
                 charge_method: str = DEFAULT_NAGL_MODEL):
        """
        Args:
            forcefield: Name of the OpenFF force field file.
            max_iters: Maximum number of minimization iterations.
            platform: OpenMM platform ('CPU', 'CUDA', or 'OpenCL').
            tolerance: Convergence tolerance (kJ/mol/nm).
            charge_method: Partial charge method. Use a NAGL .pt model
                for speed; 'am1bcc' is NOT recommended (falls back to
                slow AmberTools).
        """
        self.forcefield = forcefield
        self.max_iters = max_iters
        self.platform = platform
        self.tolerance = tolerance
        self.charge_method = charge_method
        self._check_dependencies()

    def _check_dependencies(self):
        """Verify that openff-toolkit, openff-interchange, openff-nagl
        and openmm are importable."""
        try:
            import openff.nagl            # noqa: F401
            import openff.interchange     # noqa: F401
            import openmm                 # noqa: F401
        except ImportError as e:
            raise ImportError(
                f"OpenFFOptimizer requires openff-toolkit, openff-interchange, "
                f"openff-nagl and openmm. Install with:\n"
                f"  conda install -c conda-forge openff-toolkit "
                f"openff-interchange openff-nagl openff-nagl-models openmm\n"
                f"Original error: {e}"
            )

    def _get_ff(self) -> ForceField:
        """Return a cached ForceField instance (parsed once per name)."""
        if self.forcefield not in _FF_CACHE:
            _FF_CACHE[self.forcefield] = ForceField(self.forcefield)
        return _FF_CACHE[self.forcefield]

    @staticmethod
    def _assign_stereo_from_3d(mol: Chem.Mol, confId: int = 0) -> None:
        """Infer R/S tags from 3D coordinates so OpenFF accepts the Mol."""
        try:
            Chem.AssignStereochemistryFrom3D(
                mol, confId=confId, force=True, cleanIt=True
            )
            Chem.AssignStereochemistry(
                mol, force=True, cleanIt=True, flagPossibleStereoCenters=True
            )
        except Exception:
            pass

    def optimize(self, mol: Chem.Mol, confId: int = 0) -> Chem.Mol:
        """
        Run OpenFF/OpenMM energy minimization on the given Mol.

        Workflow:
            1. Assign stereochemistry from 3D coordinates.
            2. Convert RDKit Mol -> OpenFF Molecule.
            3. Assign partial charges via NAGL (fast, no sqm needed).
            4. Parameterize with the chosen force field.
            5. Build an OpenMM System via Interchange.
            6. Run LocalEnergyMinimizer.
            7. Write the minimized coordinates back into the RDKit Mol.
        """
        # 1. Stereo
        self._assign_stereo_from_3d(mol, confId=confId)

        # 2. RDKit -> OpenFF (suppress "Omitted undefined stereo" warnings)
        #    First try strict conversion; fall back to permissive if it fails.
        try:
            off_mol = OFFMolecule.from_rdkit(mol)
        except Exception:
            off_mol = OFFMolecule.from_rdkit(mol, allow_undefined_stereo=True)

        # 3. Assign partial charges via NAGL (fast GNN, ~milliseconds)
        off_mol.assign_partial_charges(
            partial_charge_method=self.charge_method,
            toolkit_registry=GLOBAL_TOOLKIT_REGISTRY,
        )

        # 4. Cached ForceField
        ff = self._get_ff()

        # 5. Build Interchange, reusing the pre-assigned NAGL charges
        interchange = Interchange.from_smirnoff(
            force_field=ff,
            topology=[off_mol],
            charge_from_molecules=[off_mol],
        )
        system = interchange.to_openmm_system()

        # 6. Set initial positions
        conf = mol.GetConformer(confId)
        positions = np.array(conf.GetPositions()) * unit.angstrom

        # 7. Build OpenMM Context
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName(self.platform)
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)

        # 8. Minimize
        LocalEnergyMinimizer.minimize(
            context,
            tolerance=self.tolerance * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=self.max_iters,
        )

        # 9. Retrieve optimized positions
        state = context.getState(getPositions=True)
        new_positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

        # 10. Write back to RDKit Mol
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, new_positions[i].tolist())

        del context, integrator
        return mol

    @property
    def name(self) -> str:
        return (f'openff({self.forcefield}, iters={self.max_iters}, '
                f'charges={self.charge_method})')