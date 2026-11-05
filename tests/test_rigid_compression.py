import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from rdkit import Chem
from rdkit.Chem import AllChem
from openff.toolkit import Molecule, Topology
from openff.units import unit

from polymer_lib._rigid_compression import (
    _inter_pairs, _rigid_energy_gradient, compress_topology,
)


def test_rigid_gradient_matches_finite_difference_with_periodic_contacts():
    index = np.array([0, 0, 1, 1])
    local = np.array([[-0.3, 0.2, 0], [0.3, -0.2, 0],
                      [-0.2, 0, 0.3], [0.2, 0, -0.3]])
    base = np.array([[0.4, 1., 1.], [9.5, 1., 1.]])
    x = np.array([.02, .04, -.02, -.03, .03, .01,
                  .11, -.07, .05, -.1, .05, .07])
    fun = lambda v: _rigid_energy_gradient(v, local, base, index, 10., 2.)[0]
    energy, analytic = _rigid_energy_gradient(x, local, base, index, 10., 2.)
    numerical = approx_derivative(fun, x, method='3-point')
    assert energy > 0
    assert np.allclose(analytic, numerical, atol=1e-6, rtol=1e-5)


def test_compression_preserves_chains_and_reaches_target_density():
    mols=[]
    for smiles,seed in [('CC',4),('CO',5)]:
        m=Chem.AddHs(Chem.MolFromSmiles(smiles))
        assert AllChem.EmbedMolecule(m,randomSeed=seed)==0
        AllChem.MMFFOptimizeMolecule(m)
        mols.append(m)
    off=[Molecule.from_rdkit(m) for m in mols]
    topology=Topology.from_molecules(off)
    coords=np.concatenate([np.array(m.GetConformer().GetPositions()) + [3+7*i,5,5]
                           for i,m in enumerate(mols)])
    topology.set_positions(coords*unit.angstrom)
    topology.box_vectors=np.eye(3)*20*unit.angstrom
    result=compress_topology(topology,mols,0.05,0.1,2.0,seed=7,timeout=10)
    actual=np.asarray(result.get_positions().m_as(unit.angstrom))
    offset=0
    for m in mols:
        n=m.GetNumAtoms();original=coords[offset:offset+n];new=actual[offset:offset+n]
        assert np.allclose(np.linalg.norm(original[:,None]-original[None,:],axis=2),
                           np.linalg.norm(new[:,None]-new[None,:],axis=2),atol=1e-7)
        offset+=n
    side=result.box_vectors.m_as(unit.angstrom)[0,0]
    assert side==pytest.approx(20*(0.05/0.1)**(1/3))
    index=np.repeat([0,1],[m.GetNumAtoms() for m in mols])
    pairs,_=_inter_pairs(actual,index,side,1.999)
    assert len(pairs)==0


def test_compression_does_not_modify_topology_on_failure():
    m=Chem.AddHs(Chem.MolFromSmiles('CC'))
    assert AllChem.EmbedMolecule(m,randomSeed=3)==0
    topology=Topology.from_molecules([Molecule.from_rdkit(m)])
    coords=np.asarray(m.GetConformer().GetPositions())
    topology.set_positions(coords*unit.angstrom)
    topology.box_vectors=np.eye(3)*20*unit.angstrom
    before=np.array(topology.get_positions().m_as(unit.angstrom))
    with pytest.raises(TimeoutError,match='timed out'):
        compress_topology(topology,[m],0.05,0.3,2.0,timeout=0)
    assert np.array_equal(topology.get_positions().m_as(unit.angstrom),before)
    assert topology.box_vectors.m_as(unit.angstrom)[0,0]==pytest.approx(20)
