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
    monkeypatch.setattr('polymer_lib.packer.pack_box', lambda **kwargs: pytest.fail('Packmol called'))
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
    monkeypatch.setattr('polymer_lib.packer.pack_box', lambda **kw: received.append(kw) or object())
    packer = PackmolPacker(density=1.0)
    packer.pack([mol])
    assert len(received) == 1
    assert received[0]['target_density'].m_as(unit.gram / unit.milliliter) == pytest.approx(packer.effective_density)
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
