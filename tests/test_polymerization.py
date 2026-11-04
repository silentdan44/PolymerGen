import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from polymer_lib import BuildConfig, Monomer, Polymerizer, build_polymer
from polymer_lib.poly import check_3d_bond_length, check_3d_structure_poly, connect_mols


@pytest.fixture(autouse=True)
def isolate_chain_growth_from_push_off(monkeypatch):
    # The growth tests validate geometry and stereochemistry independently.
    monkeypatch.setattr('polymer_lib.push_off.push_off_chain', lambda mol, **kwargs: mol)


POLYSTYRENE = '*CC(c1ccccc1)*'


def build(smiles, count, *, seed=19, optimizer=None):
    monomer = Monomer(smiles, seed=seed)
    return Polymerizer(
        [monomer.copy() for _ in range(count)],
        optimizer=optimizer,
        seed=seed,
        retry=100,
        retry_step=200,
        rollback=5,
    ).build()


def assert_valid_chain(result, expected_atoms):
    assert result.success, result.failure_reason
    mol = result.molecule
    assert mol.GetNumAtoms() == expected_atoms
    assert len(Chem.GetMolFrags(mol)) == 1
    assert Chem.SanitizeMol(Chem.Mol(mol), catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE

    coords = np.asarray(mol.GetConformer().GetPositions())
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        length = np.linalg.norm(coords[a] - coords[b])
        assert 0.9 <= length <= 1.9, (a, b, length)

    adjacency = [[] for _ in range(mol.GetNumAtoms())]
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[a].append(b)
        adjacency[b].append(a)
    heavy_indices = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() != 'H']
    for start in heavy_indices:
        distances = {start: 0}
        queue = [start]
        for atom_idx in queue:
            if distances[atom_idx] == 3:
                continue
            for neighbor in adjacency[atom_idx]:
                if neighbor not in distances:
                    distances[neighbor] = distances[atom_idx] + 1
                    queue.append(neighbor)
        for other in heavy_indices:
            if other <= start:
                continue
            if other not in distances:
                separation = np.linalg.norm(coords[start] - coords[other])
                assert separation >= 1.8, (start, other, separation)


@pytest.mark.parametrize('count', [1, 2, 5])
def test_polymer_chain_is_connected_and_has_valid_geometry(count):
    monomer = Monomer(POLYSTYRENE)
    expected_atoms = count * monomer.mol.GetNumAtoms() - 2 * (count - 1)
    assert_valid_chain(build(POLYSTYRENE, count), expected_atoms)


def test_seed_reproduces_coordinates_for_same_monomer_geometry():
    monomer = Monomer(POLYSTYRENE, seed=123)
    monomers = [monomer.copy() for _ in range(4)]

    first = Polymerizer(monomers, optimizer=None, seed=123).build()
    second = Polymerizer(monomers, optimizer=None, seed=123).build()
    assert first.success and second.success
    assert np.allclose(
        first.molecule.GetConformer().GetPositions(),
        second.molecule.GetConformer().GetPositions(),
        atol=1e-10,
    )


def test_twenty_unit_chain_passes_global_nonbonded_check():
    monomer = Monomer(POLYSTYRENE, seed=42)
    result = Polymerizer(
        [monomer.copy() for _ in range(20)],
        optimizer=None,
        seed=42,
        dist_min=1.8,
        retry=30,
        retry_step=100,
        rollback=5,
    ).build()
    expected_atoms = 20 * monomer.mol.GetNumAtoms() - 2 * 19
    assert_valid_chain(result, expected_atoms)


def test_one_call_api_seed_reproduces_initial_and_growth_geometry():
    first = build_polymer(POLYSTYRENE, units=3, optimizer=None, seed=456)
    second = build_polymer(POLYSTYRENE, units=3, optimizer=None, seed=456)
    assert first.success and second.success
    assert np.allclose(
        first.molecule.GetConformer().GetPositions(),
        second.molecule.GetConformer().GetPositions(),
        atol=1e-10,
    )


@pytest.mark.parametrize(
    ('tacticity', 'expected'),
    [
        ('isotactic', [Chem.ChiralType.CHI_TETRAHEDRAL_CCW] * 3),
        ('syndiotactic', [
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ]),
    ],
)
def test_tacticity_assigns_expected_backbone_stereochemistry(tacticity, expected):
    monomer = Monomer(POLYSTYRENE, seed=19)
    polymerizer = Polymerizer(
        [monomer.copy() for _ in range(4)],
        optimizer=None, seed=19, tacticity=tacticity,
    )
    result = polymerizer.build()
    assert result.success, result.failure_reason
    assert [
        result.molecule.GetAtomWithIdx(index).GetChiralTag()
        for index in polymerizer._tacticity_atom_indices(result.molecule)
    ] == expected


def test_tacticity_rejects_repeat_unit_without_identifiable_backbone_centers():
    with pytest.raises(ValueError, match='not stereogenic'):
        build_polymer('*CC*', units=3, optimizer=None, seed=19, tacticity='isotactic')


def test_config_can_select_tacticity():
    monomer = Monomer(POLYSTYRENE, seed=19)
    polymerizer = Polymerizer(
        [monomer.copy() for _ in range(3)],
        optimizer=None, seed=19,
        config=BuildConfig(tacticity='syndiotactic'),
    )
    result = polymerizer.build()
    assert result.success
    assert [
        result.molecule.GetAtomWithIdx(index).GetChiralTag()
        for index in polymerizer._tacticity_atom_indices(result.molecule)
    ] == [Chem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.ChiralType.CHI_TETRAHEDRAL_CW]


def test_tacticity_accepts_explicit_configuration_sequence():
    monomer = Monomer(POLYSTYRENE, seed=19)
    polymerizer = Polymerizer(
        [monomer.copy() for _ in range(4)], optimizer=None, seed=19,
        tacticity=['+', '-', '+'],
    )
    result = polymerizer.build()
    assert [
        result.molecule.GetAtomWithIdx(index).GetChiralTag()
        for index in polymerizer._tacticity_atom_indices(result.molecule)
    ] == [
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    ]


def test_tacticity_accepts_numpy_array_and_tags_match_3d_geometry():
    monomer = Monomer(POLYSTYRENE, seed=19)
    tacticity = np.array(['+', '-'], dtype=str)
    polymerizer = Polymerizer(
        [monomer.copy() for _ in range(3)],
        optimizer=None,
        seed=19,
        tacticity=tacticity,
    )
    result = polymerizer.build()
    assert result.success, result.failure_reason
    mol = result.molecule
    indices = polymerizer._tacticity_atom_indices(mol)
    graph_tags = [mol.GetAtomWithIdx(i).GetChiralTag() for i in indices]
    geometry_mol = Chem.Mol(mol)
    Chem.AssignAtomChiralTagsFromStructure(geometry_mol, replaceExistingTags=True)
    assert [geometry_mol.GetAtomWithIdx(i).GetChiralTag() for i in indices] == graph_tags
    assert graph_tags[0] != graph_tags[1]


def test_tacticity_sequence_length_must_match_stereocenters():
    with pytest.raises(ValueError, match='must have 2 entries'):
        build_polymer(
            POLYSTYRENE, units=3, optimizer=None, seed=19,
            tacticity=['+'],
        )


def test_tacticity_targets_marked_backbone_center_and_preserves_side_stereocenters():
    smiles = '*C[C@H](F)C(c1ccccc1)*'
    monomer = Monomer(smiles, seed=19)
    polymerizer = Polymerizer(
        [monomer.copy() for _ in range(3)],
        optimizer=None,
        seed=19,
        tacticity='syndiotactic',
        config=BuildConfig(tacticity='syndiotactic', dist_min=1.4),
    )
    result = polymerizer.build()
    assert result.success, result.failure_reason
    mol = result.molecule
    marked_indices = polymerizer._tacticity_atom_indices(mol)
    assert [mol.GetAtomWithIdx(i).GetChiralTag() for i in marked_indices] == [
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    ]
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    side_center_codes = [
        atom.GetProp('_CIPCode')
        for atom in mol.GetAtoms()
        if atom.HasProp('_CIPCode')
        and any(neighbor.GetSymbol() == 'F' for neighbor in atom.GetNeighbors())
    ]
    assert side_center_codes == ['S', 'S', 'S']


def test_target_atoms_rounds_up_to_whole_repeat_units():
    result = build_polymer(POLYSTYRENE, target_atoms=20, optimizer=None, seed=19)
    assert result.success, result.failure_reason
    assert result.molecule.GetNumAtoms() == 34


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param({}, id='neither'),
        pytest.param({'units': 2, 'target_atoms': 20}, id='both-values'),
        pytest.param({'units': None, 'target_atoms': 20}, id='both-names'),
    ],
)
def test_build_polymer_requires_exactly_one_chain_length_mode(kwargs):
    with pytest.raises(ValueError):
        build_polymer(
            POLYSTYRENE, optimizer=None, seed=19, **kwargs,
        )


@pytest.mark.parametrize('pattern', ['small-large-small', 'large-small-large'])
def test_mixed_monomer_sizes_build_a_valid_chain(pattern):
    small = Monomer('*CC*', seed=19)
    large = Monomer(POLYSTYRENE, seed=19)
    monomers = (
        [small.copy(), large.copy(), small.copy()]
        if pattern == 'small-large-small'
        else [large.copy(), small.copy(), large.copy()]
    )
    expected_atoms = sum(m.mol.GetNumAtoms() for m in monomers) - 2 * (len(monomers) - 1)
    result = Polymerizer(
        monomers, optimizer=None, seed=19, retry=100, retry_step=200, rollback=5
    ).build()
    assert_valid_chain(result, expected_atoms)


def test_default_mmff_converges_on_two_repeat_units():
    monomer = Monomer(POLYSTYRENE, seed=19)
    result = Polymerizer(
        [monomer.copy(), monomer.copy()],
        optimizer='mmff',
        seed=19,
        retry=3,
        retry_step=10,
        rollback=1,
    ).build()
    assert_valid_chain(result, 2 * monomer.mol.GetNumAtoms() - 2)


def test_new_interunit_bond_matches_requested_length():
    first = Monomer(POLYSTYRENE, seed=19).mol
    second = Monomer(POLYSTYRENE, seed=19).mol
    tail = first.GetIntProp('tail_idx')
    tail_neighbor = first.GetIntProp('tail_ne_idx')
    head = second.GetIntProp('head_idx')
    head_neighbor = second.GetIntProp('head_ne_idx')

    result = connect_mols(first, second, bond_length=1.5, random_rot=False)
    atom_a = tail_neighbor - (tail_neighbor > tail)
    atom_b = first.GetNumAtoms() - 1 + head_neighbor - (head_neighbor > head)
    coords = np.asarray(result.GetConformer().GetPositions())
    assert np.linalg.norm(coords[atom_a] - coords[atom_b]) == pytest.approx(1.5, abs=1e-8)


def test_bond_length_check_uses_bond_vectors_and_detects_stretching():
    mol = Monomer(POLYSTYRENE, seed=19).mol
    assert check_3d_bond_length(mol)
    conf = mol.GetConformer()
    bond = mol.GetBondWithIdx(0)
    atom_idx = bond.GetEndAtomIdx()
    old = conf.GetAtomPosition(atom_idx)
    conf.SetAtomPosition(atom_idx, Point3D(old.x + 10.0, old.y, old.z))
    assert not check_3d_bond_length(mol)


def test_sparse_contact_check_matches_expected_for_large_graph():
    atom_count = 600
    editable = Chem.RWMol()
    for _ in range(atom_count):
        editable.AddAtom(Chem.Atom('C'))
    for idx in range(atom_count - 1):
        editable.AddBond(idx, idx + 1, Chem.BondType.SINGLE)
    mol = editable.GetMol()
    conformer = Chem.Conformer(atom_count)
    for idx in range(atom_count):
        conformer.SetAtomPosition(idx, Point3D(idx * 3.0, 0.0, 0.0))
    mol.AddConformer(conformer)

    assert check_3d_structure_poly(mol, mol, dist_min=1.8)
    mol.GetConformer().SetAtomPosition(atom_count - 1, Point3D(0.0, 0.0, 0.0))
    assert not check_3d_structure_poly(mol, mol, dist_min=1.8)


def test_build_runs_push_off_once_with_options(monkeypatch):
    calls = []
    def relax(mol, **kwargs):
        calls.append(kwargs)
        return Chem.Mol(mol)
    monkeypatch.setattr('polymer_lib.push_off.push_off_chain', relax)
    result = build_polymer('*CC*', units=2, optimizer=None, seed=19,
                           push_off_options={'steps': 4})
    assert result.success
    assert result.raw_molecule is not result.molecule
    assert calls == [{'steps': 4, 'seed': 19}]


def test_conformer_pools_are_reproducible_and_preserve_input():
    monomer = Monomer(POLYSTYRENE, seed=42)
    original = np.array(monomer.mol.GetConformer().GetPositions())
    first = Polymerizer([monomer, monomer], seed=42, conformer_count=4)
    second = Polymerizer([monomer, monomer], seed=42, conformer_count=4)
    pools = first._prepare_conformer_pools()
    repeated = second._prepare_conformer_pools()
    assert len(pools[0]) == 4
    assert np.array_equal(pools[0][0], original)
    assert all(np.array_equal(a, b) for a, b in zip(pools[0], repeated[0]))
    assert not np.allclose(pools[0][0], pools[0][1])
    assert np.array_equal(monomer.mol.GetConformer().GetPositions(), original)


def test_failed_attempt_reuses_conformer_for_same_growth_step(monkeypatch):
    from polymer_lib import poly
    monomer = Monomer(POLYSTYRENE, seed=42)
    seen = []
    connect = poly.connect_mols
    check = poly.check_3d_structure_poly
    def record(first, second, **kwargs):
        seen.append(np.array(second.GetConformer().GetPositions()))
        return connect(first, second, **kwargs)
    def reject_first(*args, **kwargs):
        if len(seen) == 1:
            return False
        return check(*args, **kwargs)
    monkeypatch.setattr(poly, 'connect_mols', record)
    monkeypatch.setattr(poly, 'check_3d_structure_poly', reject_first)
    result = Polymerizer([monomer, monomer], seed=42, optimizer=None,
                         conformer_count=4).build()
    assert result.success
    assert len(seen) >= 2
    assert np.allclose(seen[0], seen[1])


def test_rollback_to_start_draws_initial_monomer_again(monkeypatch):
    from polymer_lib import poly
    monomer = Monomer(POLYSTYRENE, seed=42)
    starts = []
    connect = poly.connect_mols
    def record(first, second, **kwargs):
        starts.append(np.array(first.GetConformer().GetPositions()))
        return connect(first, second, **kwargs)
    monkeypatch.setattr(poly, 'connect_mols', record)
    monkeypatch.setattr(poly, 'check_3d_structure_poly', lambda *args, **kwargs: len(starts) > 1)
    result = Polymerizer([monomer, monomer], seed=42, optimizer=None,
                         conformer_count=4, retry_step=1, retry=1).build()
    assert result.success
    assert not np.allclose(starts[0], starts[1])


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_conformer_count_must_be_positive_integer(value):
    with pytest.raises(ValueError, match='conformer_count'):
        BuildConfig(conformer_count=value)


def test_one_conformer_uses_original_geometry_only():
    monomer = Monomer(POLYSTYRENE, seed=42)
    polymerizer = Polymerizer([monomer, monomer], seed=42, conformer_count=1)
    pools = polymerizer._prepare_conformer_pools()
    assert all(len(pool) == 1 for pool in pools)
    assert np.array_equal(pools[0][0], monomer.mol.GetConformer().GetPositions())


def test_alternative_embedding_failure_keeps_supplied_geometry(monkeypatch):
    from rdkit.Chem import AllChem
    monomer = Monomer(POLYSTYRENE, seed=42)
    monkeypatch.setattr(AllChem, 'EmbedMolecule', lambda *args, **kwargs: -1)
    polymerizer = Polymerizer([monomer, monomer], seed=42, conformer_count=3)
    pools = polymerizer._prepare_conformer_pools()
    assert len(pools[0]) == 1
    assert np.array_equal(pools[0][0], monomer.mol.GetConformer().GetPositions())


def test_random_conformer_is_selected_at_every_growth_step(monkeypatch):
    from polymer_lib import poly
    monomer = Monomer(POLYSTYRENE, seed=42)
    polymerizer = Polymerizer([monomer] * 6, seed=42, optimizer=None,
                             conformer_count=4)
    pools = polymerizer._prepare_conformer_pools()
    observed = []
    original_connect = poly.connect_mols

    def record(first, second, **kwargs):
        coords = np.array(second.GetConformer().GetPositions())
        candidates = pools[len(observed) + 1]
        matches = [i for i, candidate in enumerate(candidates)
                   if np.allclose(coords, candidate)]
        assert len(matches) == 1
        observed.append(matches[0])
        return original_connect(first, second, **kwargs)

    monkeypatch.setattr(poly, 'connect_mols', record)
    monkeypatch.setattr(poly, 'check_3d_structure_poly', lambda *args, **kwargs: True)
    assert polymerizer.build().success
    assert observed == [1, 3, 2, 3, 0]
