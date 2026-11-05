"""End-to-end smoke test for OpenFF push-off and MD restart artifacts."""

import numpy as np
import pytest


pytest.importorskip('openmm')
pytest.importorskip('openff.toolkit')
pytest.importorskip('openff.interchange')
pytest.importorskip('openff.nagl_models')

import openmm
from openmm import unit
from openmm.app import PDBFile, Simulation
from openff.interchange import Interchange
from openff.nagl_models import list_available_nagl_models
from openff.toolkit import ForceField, Molecule
from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY

from polymer_lib import BuildConfig, build_polymer, push_off_chain


def test_build_push_off_serialize_reload_and_continue_md(tmp_path):
    result = build_polymer(
        '*CC(c1ccccc1)*', units=2, optimizer=None,
        config=BuildConfig(retry=20, retry_step=100, rollback=2), seed=17,
        push_off_options={"steps": 4, "stages": 2, "soft_minimize_max_iters": 5, "max_iters": 5},
    )
    assert result.success, result.failure_reason
    original_positions = np.array(result.molecule.GetConformer().GetPositions())

    relaxed = push_off_chain(
        result.molecule,
        steps=4,
        stages=2,
        seed=17,
        soft_minimize_max_iters=5,
        max_iters=5,
    )
    assert relaxed is not result.molecule
    assert np.array_equal(
        original_positions, result.molecule.GetConformer().GetPositions()
    )
    assert np.isfinite(relaxed.GetConformer().GetPositions()).all()

    try:
        off_mol = Molecule.from_rdkit(relaxed)
    except Exception:
        off_mol = Molecule.from_rdkit(relaxed, allow_undefined_stereo=True)
    off_mol.assign_partial_charges(
        partial_charge_method=list_available_nagl_models()[-1],
        toolkit_registry=GLOBAL_TOOLKIT_REGISTRY,
    )
    interchange = Interchange.from_smirnoff(
        force_field=ForceField('openff-2.1.0.offxml'),
        topology=[off_mol],
        charge_from_molecules=[off_mol],
    )
    system = interchange.to_openmm_system()
    topology = interchange.to_openmm_topology()
    positions = interchange.positions.to_openmm()
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtoseconds
    )
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 17)

    (tmp_path / 'system.xml').write_text(openmm.XmlSerializer.serialize(system))
    (tmp_path / 'integrator.xml').write_text(
        openmm.XmlSerializer.serialize(integrator)
    )
    state = simulation.context.getState(getPositions=True, getVelocities=True)
    (tmp_path / 'state.xml').write_text(openmm.XmlSerializer.serialize(state))
    with (tmp_path / 'chain.pdb').open('w') as pdb_file:
        PDBFile.writeFile(topology, positions, pdb_file)
    (tmp_path / 'interchange.json').write_text(interchange.model_dump_json())
    simulation.saveCheckpoint(str(tmp_path / 'checkpoint.chk'))

    loaded_interchange = Interchange.model_validate_json(
        (tmp_path / 'interchange.json').read_text()
    )
    assert loaded_interchange.to_openmm_system().getNumParticles() == system.getNumParticles()

    pdb = PDBFile(str(tmp_path / 'chain.pdb'))
    loaded_system = openmm.XmlSerializer.deserialize(
        (tmp_path / 'system.xml').read_text()
    )
    loaded_integrator = openmm.XmlSerializer.deserialize(
        (tmp_path / 'integrator.xml').read_text()
    )
    resumed = Simulation(pdb.topology, loaded_system, loaded_integrator)
    resumed.loadState(str(tmp_path / 'state.xml'))
    resumed.step(2)
    assert resumed.currentStep == 2

    checkpoint_resume = Simulation(
        pdb.topology,
        openmm.XmlSerializer.deserialize((tmp_path / 'system.xml').read_text()),
        openmm.XmlSerializer.deserialize((tmp_path / 'integrator.xml').read_text()),
    )
    checkpoint_resume.loadCheckpoint(str(tmp_path / 'checkpoint.chk'))
    checkpoint_resume.step(1)


def test_generate_and_pack_runs_push_off_once():
    from polymer_lib.packer import PackmolPacker

    result = build_polymer(
        '*CC*', units=2, optimizer=None, seed=7,
        push_off_options={'steps': 4, 'stages': 2,
                          'soft_minimize_max_iters': 5, 'max_iters': 5},
    )
    assert result.success
    assert result.raw_molecule is not result.molecule
    packer = PackmolPacker(density=0.3, max_attempts=1)
    topology = packer.pack([result.molecule])
    assert topology.n_atoms == result.molecule.GetNumAtoms()
    assert 0 < packer.effective_density <= 0.3
