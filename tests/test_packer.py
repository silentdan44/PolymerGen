import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from openff.units import unit

from polymer_lib.packer import PackmolPacker, _check_chain_geometry


def molecule(smiles='CCCCCC'):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=17) == 0
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_detects_internal_overlap_before_packmol(monkeypatch):
    mol = molecule()
    conf = mol.GetConformer()
    conf.SetAtomPosition(5, conf.GetAtomPosition(0))
    monkeypatch.setattr('polymer_lib.packer._run_packmol_attempt', lambda *args: pytest.fail('Packmol called'))
    with pytest.raises(ValueError, match=r'Chain 0 has an internal nonbonded overlap: atoms 0 and 5'):
        PackmolPacker().pack([mol])


def test_bonded_neighbours_do_not_trigger_overlap():
    mol = molecule('CC')
    mol.GetConformer().SetAtomPosition(1, mol.GetConformer().GetAtomPosition(0))
    _check_chain_geometry(mol, 0)


def test_density_is_reduced_before_packing(monkeypatch, caplog):
    mol = molecule()
    # Stretch without introducing non-local overlaps.
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(pos.x * 10, pos.y, pos.z))
    received = []
    monkeypatch.setattr('polymer_lib.packer._run_packmol_attempt', lambda *args: received.append(args) or object())
    packer = PackmolPacker(density=1.0)
    packer.pack([mol])
    assert len(received) == 1
    assert received[0][1] == pytest.approx(packer.effective_density)
    assert 0 < packer.effective_density < 1.0
    assert 'using' in caplog.text


def test_pack_real_molecule_with_geometry_adjusted_density():
    mol = molecule('CC')
    packer = PackmolPacker(density=0.3, max_attempts=1)
    topology = packer.pack([mol])
    assert topology.n_atoms == mol.GetNumAtoms()
    assert np.isfinite(topology.get_positions().m).all()


def test_json_api_reports_internal_overlap():
    from polymer_lib.api import handle_request

    mol = molecule()
    mol.GetConformer().SetAtomPosition(5, mol.GetConformer().GetAtomPosition(0))
    response = handle_request({
        'api_version': '1.0', 'action': 'pack',
        'input': {'molecules': [{'format': 'mol', 'data': Chem.MolToMolBlock(mol)}]},
    })
    assert not response['ok']
    assert 'internal nonbonded overlap' in response['error']['message']
    assert 'atoms 0 and 5' in response['error']['message']


def test_packmol_timeout_retries_with_new_reproducible_seeds(monkeypatch):
    import subprocess
    mol = molecule('CC')
    seeds = []
    def attempt(molecules, density, tolerance, seed, timeout, disable_movebad):
        seeds.append(seed)
        assert density > 0 and timeout == 0.2
        if len(seeds) % 3:
            raise subprocess.TimeoutExpired('packmol', timeout)
        return 'success'
    monkeypatch.setattr('polymer_lib.packer._run_packmol_attempt', attempt)
    packer = PackmolPacker(max_attempts=3, timeout=0.2, seed=10)
    assert packer.pack([mol]) == 'success'
    assert len(set(seeds)) == 3
    assert packer.pack([mol]) == 'success'
    assert seeds[:3] == seeds[3:]


def test_exhausted_packmol_timeout_uses_periodic_fallback(monkeypatch):
    import subprocess
    mol = molecule('CC')
    def attempt(*args):
        raise subprocess.TimeoutExpired('packmol', 0.1)
    monkeypatch.setattr('polymer_lib.packer._run_packmol_attempt', attempt)
    packer = PackmolPacker(max_attempts=2, timeout=0.1, seed=10)
    topology = packer.pack([mol])
    assert topology.n_atoms == mol.GetNumAtoms()
    assert np.isfinite(topology.get_positions().m).all()
    assert 0 < packer.effective_density


def test_large_dense_box_packs_dilute_then_compresses_to_target(monkeypatch):
    from types import SimpleNamespace
    from polymer_lib import _rigid_compression
    # Only the orchestration is under test; geometries and conversion are stubbed.
    chain = Chem.RWMol()
    for _ in range(1000):
        chain.AddAtom(Chem.Atom('C'))
    for i in range(999):
        chain.AddBond(i, i + 1, Chem.BondType.SINGLE)
    chain = chain.GetMol()
    off = SimpleNamespace(atoms=[SimpleNamespace(mass=1*unit.dalton)]*1000)
    calls = []
    monkeypatch.setattr('polymer_lib.packer._check_chain_geometry', lambda mol, i: 1.0)
    monkeypatch.setattr(PackmolPacker, '_rdkit_to_openff', lambda self, mol: off)
    def pack_attempt(molecules, density, tolerance, seed, timeout, disable_movebad):
        calls.append(('pack', density))
        return 'dilute-topology'
    def compress(topology, chains, start_density, target_density, tolerance, **kwargs):
        calls.append(('compress', start_density, target_density))
        assert topology == 'dilute-topology'
        return 'compressed-topology'
    monkeypatch.setattr('polymer_lib.packer._run_packmol_attempt', pack_attempt)
    monkeypatch.setattr(_rigid_compression, 'compress_topology', compress)
    packer = PackmolPacker(density=0.3, seed=4)
    assert packer.pack([chain]) == 'compressed-topology'
    assert calls == [('pack', 0.05), ('compress', 0.05, 0.3)]
    assert packer.initial_pack_density == 0.05
    assert packer.effective_density == 0.3


def test_real_worker_timeout_is_bounded():
    import subprocess
    from openff.toolkit import Molecule
    from polymer_lib.packer import _run_packmol_attempt
    mol = Molecule.from_rdkit(molecule('CC'))
    with pytest.raises(subprocess.TimeoutExpired):
        _run_packmol_attempt([mol], 0.1, 2.0, 42, 0.01, False)


def test_packmol_exhaustion_returns_safe_random_periodic_fallback(monkeypatch):
    mol = molecule('CC')
    monkeypatch.setattr(
        'polymer_lib.packer._run_packmol_attempt',
        lambda *args: (_ for _ in ()).throw(RuntimeError('simulated Packmol failure')),
    )
    monkeypatch.setattr(
        'polymer_lib._rigid_compression.compress_topology',
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('simulated box push-off failure')),
    )
    packer = PackmolPacker(density=0.3, max_attempts=1, timeout=0.01, seed=11)
    topology = packer.pack([mol, molecule('CC')])
    coords = topology.get_positions().m_as(unit.angstrom)
    box = topology.box_vectors.m_as(unit.angstrom)

    assert topology.n_atoms == 2 * mol.GetNumAtoms()
    assert np.isfinite(coords).all()
    assert np.allclose(box, np.eye(3) * box[0, 0])
    assert 0 < packer.effective_density < 0.3
    assert packer.initial_pack_density == pytest.approx(packer.effective_density)
