"""Post-build soft push-off relaxation for a complete polymer chain."""

import numpy as np
from rdkit import Chem


def push_off_chain(
    mol: Chem.Mol,
    *,
    forcefield: str = 'openff-2.1.0.offxml',
    charge_method: str = None,
    steps: int = 1000,
    stages: int = 10,
    temperature: float = 300.0,
    timestep: float = 1.0,
    friction: float = 1.0,
    amplitude: float = 25.0,
    cutoff_scale: float = 1.5,
    seed: int = None,
    platform: str = 'CPU',
    soft_minimize: bool = True,
    soft_minimize_max_iters: int = 200,
    soft_minimize_tolerance: float = 10.0,
    minimize: bool = True,
    max_iters: int = 200,
    tolerance: float = 10.0,
) -> Chem.Mol:
    """Relax a completed chain with staged OpenMM soft-core dynamics.

    The molecule is converted to an OpenFF force-field system. During the
    push-off, bonded terms remain active while Lennard-Jones and electrostatic
    terms are temporarily replaced by a finite cosine repulsion. Original
    nonbonded parameters are restored before the optional physical minimization.
    The input molecule is copied; its conformer coordinates are not mutated.

    Args:
        mol: Completed RDKit polymer with explicit hydrogens and a 3D conformer.
        forcefield: OpenFF force field name.
        charge_method: NAGL charge model; defaults to the installed NAGL model.
        steps: Total Langevin steps, divided among stages.
        stages: Number of gradual repulsion increments.
        temperature: Langevin temperature in kelvin.
        timestep: Integrator timestep in femtoseconds.
        friction: Langevin friction in inverse picoseconds.
        amplitude: Final soft repulsion maximum in kJ/mol per pair.
        cutoff_scale: Pair cutoff as a multiple of mixed Lennard-Jones sigma.
        seed: Optional random seed for initial velocities and Langevin dynamics.
        platform: OpenMM platform name, such as ``CPU`` or ``CUDA``.
        soft_minimize: Minimize using the soft potential before Langevin MD.
        soft_minimize_max_iters: Maximum minimizer iterations for the soft system.
        soft_minimize_tolerance: Soft minimizer force tolerance in kJ/(mol nm).
        minimize: Run physical OpenFF energy minimization after push-off.
        max_iters: Maximum minimizer iterations when ``minimize`` is true.
        tolerance: Minimizer force tolerance in kJ/(mol nm).

    Returns:
        A copy of ``mol`` with relaxed coordinates.
    """
    if mol is None or mol.GetNumConformers() == 0:
        raise ValueError('mol must contain a 3D conformer')
    if not isinstance(steps, int) or steps < 1:
        raise ValueError('steps must be a positive integer')
    if not isinstance(stages, int) or stages < 1 or stages > steps:
        raise ValueError('stages must be between 1 and steps')
    if min(temperature, timestep, friction, amplitude, cutoff_scale) <= 0:
        raise ValueError('temperature, timestep, friction, amplitude, and cutoff_scale must be positive')
    if not isinstance(max_iters, int) or max_iters < 0:
        raise ValueError('max_iters must be a nonnegative integer')
    if not isinstance(soft_minimize_max_iters, int) or soft_minimize_max_iters < 0:
        raise ValueError('soft_minimize_max_iters must be a nonnegative integer')
    if soft_minimize_tolerance <= 0:
        raise ValueError('soft_minimize_tolerance must be positive')
    if seed is not None and (not isinstance(seed, int) or not 0 <= seed <= 2**31 - 1):
        raise ValueError('seed must be between 0 and 2**31 - 1')

    try:
        from openff.toolkit import Molecule as OFFMolecule
        from openff.toolkit.typing.engines.smirnoff import ForceField
        from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
        from openff.nagl_models import list_available_nagl_models
        from openff.interchange import Interchange
        from openmm import LocalEnergyMinimizer, unit
        import openmm
    except ImportError as exc:
        raise ImportError(
            'push_off_chain requires OpenFF Toolkit, Interchange, NAGL, and OpenMM'
        ) from exc

    if charge_method is None:
        models = list_available_nagl_models()
        if not models:
            raise ImportError('No OpenFF NAGL charge models are installed')
        charge_method = models[-1]

    working_mol = Chem.Mol(mol)
    try:
        off_mol = OFFMolecule.from_rdkit(working_mol)
    except Exception:
        off_mol = OFFMolecule.from_rdkit(working_mol, allow_undefined_stereo=True)
    off_mol.assign_partial_charges(
        partial_charge_method=charge_method,
        toolkit_registry=GLOBAL_TOOLKIT_REGISTRY,
    )
    interchange = Interchange.from_smirnoff(
        force_field=ForceField(forcefield),
        topology=[off_mol],
        charge_from_molecules=[off_mol],
    )
    system = interchange.to_openmm_system()
    nonbonded = _get_nonbonded_force(system, openmm)
    conf = working_mol.GetConformer()
    positions = np.array(conf.GetPositions()) * unit.angstrom
    positions = _run_soft_push_off(
        system=system,
        nonbonded=nonbonded,
        positions=positions,
        openmm=openmm,
        unit=unit,
        steps=steps,
        stages=stages,
        temperature=temperature,
        timestep=timestep,
        friction=friction,
        amplitude=amplitude,
        cutoff_scale=cutoff_scale,
        seed=seed,
        platform=platform,
        soft_minimize=soft_minimize,
        soft_minimize_max_iters=soft_minimize_max_iters,
        soft_minimize_tolerance=soft_minimize_tolerance,
    )

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName(platform)
    )
    try:
        context.setPositions(positions)
        if minimize:
            LocalEnergyMinimizer.minimize(
                context,
                tolerance=tolerance * unit.kilojoules_per_mole / unit.nanometer,
                maxIterations=max_iters,
            )
        final_positions = context.getState(getPositions=True).getPositions(
            asNumpy=True
        ).value_in_unit(unit.angstrom)
    finally:
        del context, integrator

    for atom_idx, xyz in enumerate(final_positions):
        conf.SetAtomPosition(atom_idx, xyz.tolist())
    return working_mol


def _get_nonbonded_force(system, openmm):
    forces = [
        system.getForce(i)
        for i in range(system.getNumForces())
        if isinstance(system.getForce(i), openmm.NonbondedForce)
    ]
    if len(forces) != 1:
        raise ValueError(
            'soft push-off requires exactly one OpenMM NonbondedForce; '
            f'found {len(forces)}'
        )
    return forces[0]


def _run_soft_push_off(
    *, system, nonbonded, positions, openmm, unit, steps, stages,
    temperature, timestep, friction, amplitude, cutoff_scale, seed, platform,
    soft_minimize, soft_minimize_max_iters, soft_minimize_tolerance,
):
    """Run soft dynamics and restore every physical nonbonded parameter."""
    soft_force = openmm.CustomNonbondedForce(
        'step(rc-r)*0.5*push_off_amplitude*(1+cos(3.141592653589793*r/rc)); '
        'rc=push_off_cutoff_scale*0.5*(sigma1+sigma2)'
    )
    soft_force.addGlobalParameter('push_off_amplitude', 0.0)
    soft_force.addGlobalParameter('push_off_cutoff_scale', cutoff_scale)
    soft_force.addPerParticleParameter('sigma')
    nonbonded_method = nonbonded.getNonbondedMethod()
    if nonbonded_method == openmm.NonbondedForce.NoCutoff:
        soft_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)
    elif nonbonded_method == openmm.NonbondedForce.CutoffNonPeriodic:
        soft_force.setNonbondedMethod(
            openmm.CustomNonbondedForce.CutoffNonPeriodic
        )
    else:
        # PME/Ewald/LJPME and CutoffPeriodic all use periodic pair distances
        # in OpenMM. The temporary soft term follows their existing cutoff.
        soft_force.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    soft_force.setUseLongRangeCorrection(False)

    sigmas = []
    for atom_idx in range(nonbonded.getNumParticles()):
        _, sigma, _ = nonbonded.getParticleParameters(atom_idx)
        sigma_nm = max(sigma.value_in_unit(unit.nanometer), 1.0e-6)
        sigmas.append(sigma_nm)
        soft_force.addParticle([sigma_nm])
    if not sigmas or max(sigmas) <= 0:
        raise ValueError('soft push-off needs positive Lennard-Jones sigma values')
    if nonbonded_method != openmm.NonbondedForce.NoCutoff:
        soft_force.setCutoffDistance(nonbonded.getCutoffDistance())

    original_particles = [
        nonbonded.getParticleParameters(i)
        for i in range(nonbonded.getNumParticles())
    ]
    original_exceptions = [
        nonbonded.getExceptionParameters(i)
        for i in range(nonbonded.getNumExceptions())
    ]
    for atom1, atom2, *_ in original_exceptions:
        soft_force.addExclusion(atom1, atom2)

    context = None
    integrator = None
    soft_index = None
    try:
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
            temperature * unit.kelvin,
            friction / unit.picosecond,
            timestep * unit.femtoseconds,
        )
        if seed is not None:
            integrator.setRandomNumberSeed(seed)
        context = openmm.Context(
            system, integrator, openmm.Platform.getPlatformByName(platform)
        )
        context.setPositions(positions)
        initial_amplitude = amplitude / stages
        context.setParameter('push_off_amplitude', initial_amplitude)
        if soft_minimize:
            openmm.LocalEnergyMinimizer.minimize(
                context,
                tolerance=(
                    soft_minimize_tolerance
                    * unit.kilojoules_per_mole
                    / unit.nanometer
                ),
                maxIterations=soft_minimize_max_iters,
            )
        if seed is None:
            context.setVelocitiesToTemperature(temperature * unit.kelvin)
        else:
            context.setVelocitiesToTemperature(temperature * unit.kelvin, seed)

        per_stage, remainder = divmod(steps, stages)
        for stage in range(stages):
            context.setParameter(
                'push_off_amplitude', amplitude * (stage + 1) / stages
            )
            integrator.step(per_stage + (stage < remainder))
        return context.getState(getPositions=True).getPositions(asNumpy=True)
    finally:
        if context is not None:
            del context
        if integrator is not None:
            del integrator
        try:
            if soft_index is not None:
                system.removeForce(soft_index)
        finally:
            for i, params in enumerate(original_particles):
                nonbonded.setParticleParameters(i, *params)
            for i, params in enumerate(original_exceptions):
                nonbonded.setExceptionParameters(i, *params)
