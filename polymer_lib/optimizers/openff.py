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
                 charge_method: str = DEFAULT_NAGL_MODEL,
                 push_off_steps: int = 0,
                 push_off_stages: int = 10,
                 push_off_temperature: float = 300.0,
                 push_off_timestep: float = 1.0,
                 push_off_friction: float = 1.0,
                 push_off_amplitude: float = 25.0,
                 push_off_cutoff_scale: float = 1.5,
                 push_off_seed: int = None):
        """
        Args:
            forcefield: Name of the OpenFF force field file.
            max_iters: Maximum number of minimization iterations.
            platform: OpenMM platform ('CPU', 'CUDA', or 'OpenCL').
            tolerance: Convergence tolerance (kJ/mol/nm).
            charge_method: Partial charge method. Use a NAGL .pt model
                for speed; 'am1bcc' is NOT recommended (falls back to
                slow AmberTools).
            push_off_steps: Number of Langevin MD steps for soft-core push-off.
                Zero disables the procedure. Steps are split across stages
                that gradually raise the soft repulsion.
            push_off_stages: Number of repulsion increments.
            push_off_temperature: Langevin thermostat temperature (K).
            push_off_timestep: MD timestep (fs).
            push_off_friction: Langevin friction coefficient (1/ps).
            push_off_amplitude: Final maximum of the finite soft pair potential
                (kJ/mol per pair at complete overlap).
            push_off_cutoff_scale: Soft-potential cutoff as a multiple of the
                Lorentz-mixed Lennard-Jones sigma.
            push_off_seed: Optional random seed for Langevin velocities.
        """
        if not isinstance(push_off_steps, int) or push_off_steps < 0:
            raise ValueError('push_off_steps must be a nonnegative integer')
        if not isinstance(push_off_stages, int) or push_off_stages < 1:
            raise ValueError('push_off_stages must be a positive integer')
        if push_off_steps and push_off_stages > push_off_steps:
            raise ValueError('push_off_stages cannot exceed push_off_steps')
        if push_off_temperature <= 0 or push_off_timestep <= 0 or push_off_friction <= 0:
            raise ValueError('push-off temperature, timestep, and friction must be positive')
        if push_off_amplitude <= 0 or push_off_cutoff_scale <= 0:
            raise ValueError('push-off amplitude and cutoff scale must be positive')
        if push_off_seed is not None and (
            not isinstance(push_off_seed, int) or not 0 <= push_off_seed <= 2**31 - 1
        ):
            raise ValueError('push_off_seed must be between 0 and 2**31 - 1')
        self.forcefield = forcefield
        self.max_iters = max_iters
        self.platform = platform
        self.tolerance = tolerance
        self.charge_method = charge_method
        self.push_off_steps = push_off_steps
        self.push_off_stages = push_off_stages
        self.push_off_temperature = push_off_temperature
        self.push_off_timestep = push_off_timestep
        self.push_off_friction = push_off_friction
        self.push_off_amplitude = push_off_amplitude
        self.push_off_cutoff_scale = push_off_cutoff_scale
        self.push_off_seed = push_off_seed
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

        # 7. Resolve severe initial overlaps with finite soft repulsion and
        # Langevin dynamics before restoring the physical nonbonded force.
        if self.push_off_steps:
            positions = self._run_push_off(system, positions)

        # 8. Build final-force-field OpenMM Context
        integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
        platform = openmm.Platform.getPlatformByName(self.platform)
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)

        # 9. Minimize with the original OpenFF nonbonded interactions restored.
        LocalEnergyMinimizer.minimize(
            context,
            tolerance=self.tolerance * unit.kilojoules_per_mole / unit.nanometer,
            maxIterations=self.max_iters,
        )

        # 10. Retrieve optimized positions
        state = context.getState(getPositions=True)
        new_positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)

        # 11. Write back to RDKit Mol
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, new_positions[i].tolist())

        del context, integrator
        return mol

    def _run_push_off(self, system, positions):
        """Run finite-core overlap removal while retaining bonded forces.

        The OpenFF NonbondedForce (both Lennard-Jones and electrostatics) is
        temporarily replaced with a cosine soft-core repulsion. Its amplitude
        is increased over short Langevin segments, then the original force is
        restored before the normal force-field minimization.
        """
        nonbonded_indices = [
            i for i in range(system.getNumForces())
            if isinstance(system.getForce(i), openmm.NonbondedForce)
        ]
        if len(nonbonded_indices) != 1:
            raise ValueError(
                'soft push-off requires exactly one OpenMM NonbondedForce; '
                f'found {len(nonbonded_indices)}'
            )
        nonbonded = system.getForce(nonbonded_indices[0])

        soft_force = openmm.CustomNonbondedForce(
            'step(rc-r)*0.5*push_off_amplitude*(1+cos(pi*r/rc)); '
            'rc=push_off_cutoff_scale*0.5*(sigma1+sigma2)'
        )
        soft_force.addGlobalParameter('push_off_amplitude', 0.0)
        soft_force.addGlobalParameter('push_off_cutoff_scale', self.push_off_cutoff_scale)
        soft_force.addPerParticleParameter('sigma')
        soft_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffNonPeriodic)
        soft_force.setUseLongRangeCorrection(False)

        sigmas = []
        for particle_index in range(nonbonded.getNumParticles()):
            _, sigma, _ = nonbonded.getParticleParameters(particle_index)
            sigma_nm = max(sigma.value_in_unit(unit.nanometer), 1.0e-6)
            sigmas.append(sigma_nm)
            soft_force.addParticle([sigma_nm])
        max_sigma = max(sigmas, default=0.0)
        if max_sigma <= 0:
            raise ValueError('soft push-off needs positive Lennard-Jones sigma values')
        soft_force.setCutoffDistance(
            self.push_off_cutoff_scale * max_sigma * unit.nanometer
        )

        # The original NonbondedForce exceptions include bonded exclusions
        # and scaled 1-4 pairs. Keep all of them out of the temporary soft term.
        for exception_index in range(nonbonded.getNumExceptions()):
            atom1, atom2, _, _, _ = nonbonded.getExceptionParameters(exception_index)
            soft_force.addExclusion(atom1, atom2)

        original_particles = [
            nonbonded.getParticleParameters(i)
            for i in range(nonbonded.getNumParticles())
        ]
        original_exceptions = [
            nonbonded.getExceptionParameters(i)
            for i in range(nonbonded.getNumExceptions())
        ]
        context = None
        integrator = None
        soft_index = None
        try:
            # Keep the force in the System but switch off its physical LJ and
            # Coulomb terms. It can then be restored exactly after MD without
            # relying on force-object ownership after System.removeForce().
            for i, (charge, sigma, _) in enumerate(original_particles):
                nonbonded.setParticleParameters(
                    i, charge * 0, sigma, 0 * unit.kilojoules_per_mole
                )
            for i, (atom1, atom2, charge_prod, sigma, _) in enumerate(original_exceptions):
                nonbonded.setExceptionParameters(
                    i, atom1, atom2, charge_prod * 0, sigma,
                    0 * unit.kilojoules_per_mole,
                )

            soft_index = system.addForce(soft_force)
            integrator = openmm.LangevinMiddleIntegrator(
                self.push_off_temperature * unit.kelvin,
                self.push_off_friction / unit.picosecond,
                self.push_off_timestep * unit.femtoseconds,
            )
            if self.push_off_seed is not None:
                integrator.setRandomNumberSeed(self.push_off_seed)
            platform = openmm.Platform.getPlatformByName(self.platform)
            context = openmm.Context(system, integrator, platform)
            context.setPositions(positions)
            if self.push_off_seed is None:
                context.setVelocitiesToTemperature(self.push_off_temperature * unit.kelvin)
            else:
                context.setVelocitiesToTemperature(
                    self.push_off_temperature * unit.kelvin,
                    self.push_off_seed,
                )

            steps_per_stage, extra_steps = divmod(
                self.push_off_steps, self.push_off_stages
            )
            for stage in range(self.push_off_stages):
                amplitude = self.push_off_amplitude * (stage + 1) / self.push_off_stages
                context.setParameter('push_off_amplitude', amplitude)
                context.step(steps_per_stage + (stage < extra_steps))

            state = context.getState(getPositions=True)
            positions = state.getPositions(asNumpy=True)
        finally:
            if context is not None:
                del context
            if integrator is not None:
                del integrator
            try:
                if soft_index is not None:
                    system.removeForce(soft_index)
            finally:
                # Always restore the physical force field, including when
                # Context construction or soft-force cleanup itself fails.
                for i, parameters in enumerate(original_particles):
                    nonbonded.setParticleParameters(i, *parameters)
                for i, parameters in enumerate(original_exceptions):
                    nonbonded.setExceptionParameters(i, *parameters)
        return positions

    @property
    def name(self) -> str:
        push_off = f', push-off={self.push_off_steps} steps' if self.push_off_steps else ''
        return (f'openff({self.forcefield}, iters={self.max_iters}, '
                f'charges={self.charge_method}{push_off})')
