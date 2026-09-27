"""
Core polymerization routines.
Based on RadonPy 0.2.9
"""

import numpy as np
from rdkit import Chem
from rdkit import Geometry as Geom
from . import calc
from . import utils
from .utils import logger


def set_linker_flag(mol, reverse=False, label=1):
    linker_indices = []
    mol.SetIntProp('head_idx', -1)
    mol.SetIntProp('tail_idx', -1)
    mol.SetIntProp('head_ne_idx', -1)
    mol.SetIntProp('tail_ne_idx', -1)
    for atom in mol.GetAtoms():
        atom.SetBoolProp('linker', False)
        atom.SetBoolProp('head', False)
        atom.SetBoolProp('tail', False)
        atom.SetBoolProp('head_neighbor', False)
        atom.SetBoolProp('tail_neighbor', False)
        if (atom.GetSymbol() == "H" and atom.GetIsotope() == label + 2) or \
           (atom.GetSymbol() == "*" and atom.HasProp('ff_type') and atom.GetProp('ff_type') != "MTIP") or \
           (atom.GetSymbol() == "*" and not atom.HasProp('ff_type')) or \
           (atom.HasProp('terminal') and atom.GetBoolProp('terminal')):
            atom.SetBoolProp('linker', True)
            linker_indices.append(atom.GetIdx())
    if not linker_indices:
        return False
    # Assign the first/last linker as head/tail, preserving atom ordering.
    mol_head_idx = linker_indices[0]
    mol_tail_idx = linker_indices[-1]
    if reverse and len(linker_indices) > 1:
        mol_head_idx, mol_tail_idx = mol_tail_idx, mol_head_idx
    if len(linker_indices) < 2 or \
       len(mol.GetAtomWithIdx(mol_head_idx).GetNeighbors()) != 1 or \
       len(mol.GetAtomWithIdx(mol_tail_idx).GetNeighbors()) != 1:
        return False
    mol.SetIntProp('head_idx', mol_head_idx)
    mol.GetAtomWithIdx(mol_head_idx).SetBoolProp('head', True)
    mol.SetIntProp('tail_idx', mol_tail_idx)
    mol.GetAtomWithIdx(mol_tail_idx).SetBoolProp('tail', True)
    head_ne_atom = mol.GetAtomWithIdx(mol_head_idx).GetNeighbors()[0]
    mol.SetIntProp('head_ne_idx', head_ne_atom.GetIdx())
    head_ne_atom.SetBoolProp('head_neighbor', True)
    tail_ne_atom = mol.GetAtomWithIdx(mol_tail_idx).GetNeighbors()[0]
    mol.SetIntProp('tail_ne_idx', tail_ne_atom.GetIdx())
    tail_ne_atom.SetBoolProp('tail_neighbor', True)
    return True


def combine_mols(mol1, mol2, res_name_1='RU0', res_name_2='RU0'):
    mol = Chem.rdmolops.CombineMols(mol1, mol2)
    mol1_n = mol1.GetNumAtoms()
    mol2_n = mol2.GetNumAtoms()

    resid = []
    if mol1.HasProp('num_units'):
        max_resid = mol1.GetIntProp('num_units')
    else:
        for i in range(mol1_n):
            atom = mol.GetAtomWithIdx(i)
            atom_name = atom.GetProp('ff_type') if atom.HasProp('ff_type') else atom.GetSymbol()
            if atom.GetPDBResidueInfo() is None:
                atom.SetMonomerInfo(
                    Chem.AtomPDBResidueInfo(
                        atom_name,
                        residueName=res_name_1,
                        residueNumber=1,
                        isHeteroAtom=False
                    )
                )
                resid.append(1)
            else:
                atom.GetPDBResidueInfo().SetName(atom_name)
                resid1 = atom.GetPDBResidueInfo().GetResidueNumber()
                resid.append(resid1)
        max_resid = max(resid) if len(resid) > 0 else 0

    for i in range(mol2_n):
        atom = mol.GetAtomWithIdx(i + mol1_n)
        atom_name = atom.GetProp('ff_type') if atom.HasProp('ff_type') else atom.GetSymbol()
        if atom.GetPDBResidueInfo() is None:
            atom.SetMonomerInfo(
                Chem.AtomPDBResidueInfo(
                    atom_name,
                    residueName=res_name_2,
                    residueNumber=1 + max_resid,
                    isHeteroAtom=False
                )
            )
            resid.append(1 + max_resid)
        else:
            atom.GetPDBResidueInfo().SetName(atom_name)
            resid2 = atom.GetPDBResidueInfo().GetResidueNumber()
            atom.GetPDBResidueInfo().SetResidueNumber(resid2 + max_resid)
            resid.append(resid2 + max_resid)

    max_resid = max(resid) if len(resid) > 0 else 0
    mol.SetIntProp('num_units', max_resid)
    return mol


def connect_mols(mol1, mol2, bond_length=1.5, dihedral=np.pi, random_rot=False,
                 set_linker=True, label1=1, label2=1,
                 confId1=0, confId2=0,
                 res_name_1='RU0', res_name_2='RU0', rng=None):
    """
    Connect tail atom in mol1 to head atom in mol2.
    Uses PROVEN WORKING LOGIC from RadonPy 0.2.9.
    """
    if mol1 is None: return mol2
    if mol2 is None: return mol1

    # Initialize linkers
    if set_linker:
        set_linker_flag(mol1, label=label1)
        set_linker_flag(mol2, label=label2)

    if mol1.GetIntProp('tail_idx') < 0 or mol1.GetIntProp('tail_ne_idx') < 0:
        logger.warning('Cannot connect_mols because mol1 does not have a tail linker atom.')
        return mol1
    elif mol2.GetIntProp('head_idx') < 0 or mol2.GetIntProp('head_ne_idx') < 0:
        logger.warning('Cannot connect_mols because mol2 does not have a head linker atom.')
        return mol1

    mol1_n = mol1.GetNumAtoms()
    mol1_coord = mol1.GetConformer(confId1).GetPositions()
    mol2_coord = mol2.GetConformer(confId2).GetPositions()

    mol1_tail_vec = mol1_coord[mol1.GetIntProp('tail_ne_idx')] - mol1_coord[mol1.GetIntProp('tail_idx')]
    mol2_head_vec = mol2_coord[mol2.GetIntProp('head_ne_idx')] - mol2_coord[mol2.GetIntProp('head_idx')]

    # Rotation mol2 to align bond vectors
    angle = calc.angle_vec(mol1_tail_vec, mol2_head_vec, rad=True)
    center = mol2_coord[mol2.GetIntProp('head_ne_idx')]
    if angle == 0:
        mol2_coord_rot = (mol2_coord - center) * -1 + center
    elif angle == np.pi:
        mol2_coord_rot = mol2_coord
    else:
        vcross = np.cross(mol1_tail_vec, mol2_head_vec)
        mol2_coord_rot = calc.rotate_rod(mol2_coord, vcross, (np.pi - angle), center=center)

    # Translation mol2
    # tail_vec points from the tail linker back into the existing chain, so
    # continue growth in the opposite direction.
    trans = mol1_coord[mol1.GetIntProp('tail_ne_idx')] - (
        bond_length * mol1_tail_vec / np.linalg.norm(mol1_tail_vec)
    )
    mol2_coord_rot = mol2_coord_rot + trans - mol2_coord_rot[mol2.GetIntProp('head_ne_idx')]

    # Rotation mol2 around new bond
    if random_rot:
        rng = np.random.default_rng() if rng is None else rng
        dih = rng.uniform(-np.pi, np.pi)
    else:
        dih = calc.dihedral_coord(
            mol1_coord[mol1.GetIntProp('head_idx')],
            mol1_coord[mol1.GetIntProp('tail_ne_idx')],
            mol2_coord_rot[mol2.GetIntProp('head_ne_idx')],
            mol2_coord_rot[mol2.GetIntProp('tail_idx')],
            rad=True
        )
    mol2_coord_rot = calc.rotate_rod(
        mol2_coord_rot, -mol1_tail_vec, (dihedral - dih),
        center=mol2_coord_rot[mol2.GetIntProp('head_ne_idx')]
    )

    # Combining mol1 and mol2
    mol = combine_mols(mol1, mol2, res_name_1=res_name_1, res_name_2=res_name_2)

    # Set atomic coordinate
    for i in range(mol2.GetNumAtoms()):
        mol.GetConformer(0).SetAtomPosition(
            i + mol1_n,
            Geom.Point3D(mol2_coord_rot[i, 0], mol2_coord_rot[i, 1], mol2_coord_rot[i, 2])
        )

    # ============================================================
    # Delete linker atoms and bonds
    # RADONPY 0.2.9 LOGIC
    # ============================================================
    mol = utils.remove_atom(mol, mol2.GetIntProp('head_idx') + mol1_n)
    mol = utils.remove_atom(mol, mol1.GetIntProp('tail_idx'))

    # Add a new bond
    tail_ne_idx = mol1.GetIntProp('tail_ne_idx')
    head_ne_idx = mol1_n - 1 + mol2.GetIntProp('head_ne_idx')
    if mol1.GetIntProp('tail_ne_idx') > mol1.GetIntProp('tail_idx'):
        tail_ne_idx -= 1
    if mol2.GetIntProp('head_ne_idx') > mol2.GetIntProp('head_idx'):
        head_ne_idx -= 1
    mol = utils.add_bond(mol, tail_ne_idx, head_ne_idx, order=Chem.rdchem.BondType.SINGLE)

    # Finalize
    Chem.SanitizeMol(mol)
    set_linker_flag(mol)

    return mol


def check_3d_proximity(coord1, coord2=None, dist_min=1.5, ignore_rad=3, dmat=None):
    if coord2 is not None:
        dist_matrix = calc.distance_matrix(coord1, coord2)
    else:
        dist_matrix = calc.distance_matrix(coord1)
        np.fill_diagonal(dist_matrix, np.nan)

    if dmat is not None:
        imat = np.where(dmat <= ignore_rad, np.nan, 1)
        dist_matrix = dist_matrix * imat
    finite = dist_matrix[~np.isnan(dist_matrix)]
    return bool(finite.size == 0 or np.min(finite) > dist_min)


def check_3d_bond_length(mol, confId=0, bond_s=2.7, bond_a=1.9, bond_d=1.8, bond_t=1.4):
    coord = np.array(mol.GetConformer(confId).GetPositions())
    dist_matrix = calc.distance_matrix(coord)
    check = True

    for b in mol.GetBonds():
        bond_l = dist_matrix[b.GetBeginAtom().GetIdx(), b.GetEndAtom().GetIdx()]
        if b.GetBondTypeAsDouble() == 1.0 and bond_l > bond_s:
            check = False
            break
        elif b.GetBondTypeAsDouble() == 1.5 and bond_l > bond_a:
            check = False
            break
        elif b.GetBondTypeAsDouble() == 2.0 and bond_l > bond_d:
            check = False
            break
        elif b.GetBondTypeAsDouble() == 3.0 and bond_l > bond_t:
            check = False
            break
    return check


def check_3d_structure_poly(poly, mon, poly_dmat=None, dist_min=1.0, ignore_rad=3,
                            check_bond_length=False, previous_atom_count=None,
                            added_atom_indices=None, removed_head_idx=None):
    """
    Improved 3D structure check with self-intersection detection.
    """
    n_mon = mon.GetNumAtoms()
    n_poly = poly.GetNumAtoms()
    if n_mon == 0 or n_poly < n_mon:
        return False
    coord = np.array(poly.GetConformer(0).GetPositions())
    if coord.shape[0] != n_poly:
        return False

    if added_atom_indices is None:
        if previous_atom_count is None:
            # Compatibility for direct callers with equal-sized repeat units.
            n_previous = n_poly - (n_mon - 2)
        else:
            n_previous = previous_atom_count - 1
        if n_previous < 0 or n_previous > n_poly:
            return False
        added_atom_indices = list(range(n_previous, n_poly))
    added_atom_indices = sorted(set(added_atom_indices))
    if not added_atom_indices or added_atom_indices[0] < 0 or added_atom_indices[-1] >= n_poly:
        return False
    added_set = set(added_atom_indices)
    previous_indices = [i for i in range(n_poly) if i not in added_set]
    p_coord = coord[previous_indices]
    m_coord = coord[added_atom_indices]
    
    # Ignore only pairs connected by a short covalent path. A fixed index
    # radius can hide clashes when atom order differs from chain order.
    ignore_mask = np.ones((p_coord.shape[0], len(added_atom_indices)), dtype=bool)
    adjacency = [[] for _ in range(n_poly)]
    for bond in poly.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[a].append(b)
        adjacency[b].append(a)
    atom_to_row = {atom_idx: row for row, atom_idx in enumerate(previous_indices)}
    atom_to_col = {atom_idx: col for col, atom_idx in enumerate(added_atom_indices)}
    for atom_idx, i in atom_to_row.items():
        seen = {atom_idx}
        frontier = {atom_idx}
        for _ in range(ignore_rad):
            frontier = {j for k in frontier for j in adjacency[k] if j not in seen}
            seen.update(frontier)
        for j in seen:
            if j in atom_to_col:
                ignore_mask[i, atom_to_col[j]] = False
    distances = calc.distance_matrix(p_coord, m_coord)
    distances[~ignore_mask] = np.inf
    check = bool(np.all(distances > dist_min))

    # Check nonbonded contacts within the added monomer too. The molecular
    # graph, rather than atom count, determines which pairs are excluded.
    if check:
        mon_distances = calc.distance_matrix(m_coord)
        # New unit has lost its head linker; convert original atom indices
        # into product-local indices before excluding bonded pairs.
        if removed_head_idx is None:
            removed_head_idx = mon.GetIntProp('head_idx') if mon.HasProp('head_idx') else None
        original_to_local = {}
        local_idx = 0
        for atom_idx in range(n_mon):
            if atom_idx == removed_head_idx:
                continue
            original_to_local[atom_idx] = local_idx
            local_idx += 1
        # The incoming head linker is removed during connection, so the
        # remaining atoms map in order to the added coordinate block.
        if local_idx != len(added_atom_indices):
            return False
        for bond in mon.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if a in original_to_local and b in original_to_local:
                a, b = original_to_local[a], original_to_local[b]
                mon_distances[a, b] = np.inf
                mon_distances[b, a] = np.inf
        # Exclude 1-3 bonded-neighbor pairs, which are constrained by angles.
        for atom in mon.GetAtoms():
            i_original = atom.GetIdx()
            if i_original not in original_to_local:
                continue
            i = original_to_local[i_original]
            for neighbor in atom.GetNeighbors():
                for second in neighbor.GetNeighbors():
                    j_original = second.GetIdx()
                    if j_original in original_to_local:
                        j = original_to_local[j_original]
                        mon_distances[i, j] = np.inf
        np.fill_diagonal(mon_distances, np.inf)
        check = bool(np.all(mon_distances > dist_min))
    
    # Check 3: Bond lengths
    if check and check_bond_length:
        check = check_3d_bond_length(poly)
    
    return check
